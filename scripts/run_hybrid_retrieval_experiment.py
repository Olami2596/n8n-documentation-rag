"""One-run dense+sparse hybrid retrieval experiment for n8n Documentation RAG.

This script preserves the existing n8n-documentation index and creates a separate
n8n-documentation-hybrid index with dotproduct metric. It corrects the V5
baseline's document-level aggregation, builds TF-IDF sparse vectors, uploads
hybrid vectors, retrieves the same 30 benchmark questions, and evaluates them
using the existing V5 Groq/OpenRouter judge module.

Run from the project root:
    python -m scripts.run_hybrid_retrieval_experiment

Required existing project files:
    data/processed/n8n_full_embedded_chunks.json
    tests/evaluation_questions_v3.json
    data/processed/retrieval_benchmark_report_v5.json
    scripts/run_retrieval_benchmark_v5_with_fallback_provider_tracking.py
"""

import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from sklearn.feature_extraction.text import TfidfVectorizer

from vectorstore.embeddings import EmbeddingGenerator

ROOT = Path.cwd()
CORPUS_FILE = ROOT / "data/processed/n8n_full_embedded_chunks.json"
QUESTIONS_FILE = ROOT / "tests/evaluation_questions_v3.json"
BASELINE_REPORT_FILE = ROOT / "data/processed/retrieval_benchmark_report_v5.json"
BASELINE_SUMMARY_FILE = (
    ROOT / "data/processed/retrieval_benchmark_summary_v5_corrected.json"
)

HYBRID_INDEX_NAME = "n8n-documentation-hybrid"
NAMESPACE = "n8n-full-corpus"
DIMENSION = 384
TOP_K = 5
BATCH_SIZE = 100
ALPHA = 0.70  # dense weight; sparse query receives 1 - ALPHA

VECTORIZER_FILE = ROOT / "data/processed/hybrid_tfidf_vectorizer.joblib"
HYBRID_RETRIEVAL_FILE = (
    ROOT / "data/processed/retrieval_evaluation_results_hybrid_v1.json"
)
HYBRID_REPORT_FILE = ROOT / "data/processed/retrieval_benchmark_report_hybrid_v1.json"
HYBRID_CSV_FILE = ROOT / "data/processed/retrieval_benchmark_results_hybrid_v1.csv"
HYBRID_SUMMARY_FILE = ROOT / "data/processed/retrieval_benchmark_summary_hybrid_v1.json"
COMPARISON_FILE = ROOT / "data/processed/retrieval_benchmark_comparison_hybrid_v1.json"

RELEVANCE_THRESHOLD = 2
INTER_REQUEST_DELAY_SECONDS = 2.5


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def metric_values(results, grade_key="relevance"):
    """Calculate P@5, MRR@5, NDCG@5, Hit@5. Zero-IDCG returns 0, not 1."""
    grades = [int(item.get(grade_key, 0)) for item in results[:TOP_K]]
    grades += [0] * (TOP_K - len(grades))
    relevant = [grade >= RELEVANCE_THRESHOLD for grade in grades]
    dcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )
    ideal = sorted(grades, reverse=True)
    idcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal, start=1)
    )
    return {
        "precision_at_5": sum(relevant) / TOP_K,
        "mrr_at_5": next(
            (1 / rank for rank, hit in enumerate(relevant, 1) if hit), 0.0
        ),
        "ndcg_at_5": dcg / idcg if idcg > 0 else 0.0,
        "hit_at_5": int(any(relevant)),
    }


def aggregate_documents_max_grade(results):
    """Keep first-seen URL order, but retain the best grade across all its chunks."""
    by_url = {}
    order = []
    for item in results:
        url = item.get("url") or f"__missing_url__:{item.get('chunk_id', len(order))}"
        if url not in by_url:
            by_url[url] = dict(item)
            order.append(url)
        else:
            if int(item.get("relevance", 0)) > int(by_url[url].get("relevance", 0)):
                # Preserve the page's first position while using its best chunk grade.
                by_url[url].update(
                    {
                        "relevance": item.get("relevance", 0),
                        "judge_reason": item.get("judge_reason", ""),
                        "judge_evidence": item.get("judge_evidence", ""),
                        "evidence_verified": item.get("evidence_verified"),
                    }
                )
    return [by_url[url] for url in order][:TOP_K]


def corrected_baseline():
    print("\n[STAGE 1/4] Correcting the existing V5 baseline metrics...")
    report = load_json(BASELINE_REPORT_FILE)
    metric_names = ["precision_at_5", "mrr_at_5", "ndcg_at_5", "hit_at_5"]
    chunk_rows, document_rows = [], []
    per_question = []

    for item in report:
        chunks = item.get("results", [])
        chunk_metrics = metric_values(chunks)
        docs = aggregate_documents_max_grade(chunks)
        doc_metrics = metric_values(docs)
        chunk_rows.append(chunk_metrics)
        document_rows.append(doc_metrics)
        per_question.append(
            {
                "question_id": item.get("question_id"),
                "question": item.get("question"),
                "chunk_level": chunk_metrics,
                "document_level": doc_metrics,
                "unique_document_count": len(
                    {r.get("url") for r in chunks if r.get("url")}
                ),
            }
        )

    summary = {
        "source_report": str(BASELINE_REPORT_FILE),
        "questions_evaluated": len(report),
        "method": "Corrected document aggregation: maximum relevance grade across all retrieved chunks for each URL; zero-IDCG NDCG is 0.",
        "chunk_level": {
            m: float(np.mean([r[m] for r in chunk_rows])) for m in metric_names
        },
        "document_level": {
            m: float(np.mean([r[m] for r in document_rows])) for m in metric_names
        },
        "per_question": per_question,
    }
    save_json(BASELINE_SUMMARY_FILE, summary)
    print("Corrected baseline saved:", BASELINE_SUMMARY_FILE)
    print("Corrected baseline document-level:", summary["document_level"])
    return summary


def chunk_text(chunk):
    return "\n".join(str(chunk.get(k) or "") for k in ("title", "content", "category"))


def pinecone_client():
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing from your .env file.")
    return Pinecone(api_key=api_key)


def ensure_hybrid_index(pc):
    print(
        "\n[STAGE 2/4] Preparing sparse TF-IDF vectors and a separate hybrid index..."
    )
    names = [item.name for item in pc.indexes.list()]
    schema = {
        "fields": {
            "_values": {
                "type": "dense_vector",
                "dimension": DIMENSION,
                "metric": "dotproduct",
            },
            "_sparse_values": {"type": "sparse_vector"},
        }
    }
    if HYBRID_INDEX_NAME not in names:
        print(f"Creating new index: {HYBRID_INDEX_NAME}")
        pc.indexes.create(
            name=HYBRID_INDEX_NAME,
            schema=schema,
            deployment={
                "deployment_type": "managed",
                "cloud": "aws",
                "region": "us-east-1",
            },
            deletion_protection="enabled",
        )
    else:
        desc = pc.indexes.describe(HYBRID_INDEX_NAME)
        fields = desc.schema.fields
        dense = fields.get("_values")
        sparse = fields.get("_sparse_values")
        if (
            not dense
            or not sparse
            or dense.metric != "dotproduct"
            or dense.dimension != DIMENSION
        ):
            raise RuntimeError(
                f"Existing {HYBRID_INDEX_NAME} index is not compatible. "
                "The script will not delete or modify it. Inspect it manually or choose another index name."
            )
        print(f"Reusing compatible index: {HYBRID_INDEX_NAME}")

    desc = pc.indexes.describe(HYBRID_INDEX_NAME)
    while not desc.status.ready:
        time.sleep(3)
        desc = pc.indexes.describe(HYBRID_INDEX_NAME)
    return pc.Index(HYBRID_INDEX_NAME)


def sparse_payload(matrix_row):
    row = matrix_row.tocsr()
    return {
        "indices": row.indices.astype(np.int32).tolist(),
        "values": row.data.astype(float).tolist(),
    }


def build_and_upload_sparse_vectors(chunks, index):
    texts = [chunk_text(c) for c in chunks]
    if VECTORIZER_FILE.exists():
        print("Loading saved TF-IDF vectorizer...")
        vectorizer = joblib.load(VECTORIZER_FILE)
        matrix = vectorizer.transform(texts)
    else:
        print("Fitting TF-IDF vocabulary on the complete chunk corpus...")
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_features=250_000,
            sublinear_tf=True,
            norm="l2",
            token_pattern=r"(?u)\b[\w./:-]+\b",
        )
        matrix = vectorizer.fit_transform(texts)
        joblib.dump(vectorizer, VECTORIZER_FILE)

    if matrix.shape[0] != len(chunks):
        raise RuntimeError(
            "Sparse matrix row count does not match the corpus chunk count."
        )

    print(f"Sparse vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"Sparse non-zero values: {matrix.nnz:,}")
    print(f"Uploading {len(chunks):,} dense+sparse vectors to {HYBRID_INDEX_NAME}...")

    total = 0
    for start in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[start : start + BATCH_SIZE]
        vectors = []
        for offset, chunk in enumerate(batch_chunks):
            row_index = start + offset
            embedding = chunk.get("embedding")
            if not isinstance(embedding, list) or len(embedding) != DIMENSION:
                raise ValueError(
                    f"Invalid dense embedding for chunk {chunk.get('chunk_id')}"
                )
            metadata = {
                key: chunk.get(key)
                for key in (
                    "document_id",
                    "title",
                    "url",
                    "category",
                    "source",
                    "content",
                    "chunk_index",
                    "start_index",
                    "status",
                )
                if chunk.get(key) is not None
            }
            vectors.append(
                {
                    "id": chunk["chunk_id"],
                    "values": embedding,
                    "sparse_values": sparse_payload(matrix[row_index]),
                    "metadata": metadata,
                }
            )
        response = index.upsert(vectors=vectors, namespace=NAMESPACE)
        total += response.upserted_count
        print(f"  Upserted {total:,}/{len(chunks):,}")
    if total != len(chunks):
        raise RuntimeError(
            f"Expected {len(chunks)} upserts but Pinecone reported {total}."
        )
    print("Hybrid corpus upload complete.")
    return vectorizer


def retrieve_hybrid_questions(questions, vectorizer, index):
    print("\n[STAGE 3/4] Running hybrid retrieval for the 30 benchmark questions...")
    embedder = EmbeddingGenerator()
    retrieval_dataset = []
    for position, qdata in enumerate(questions, start=1):
        question = qdata["question"]
        dense = np.asarray(embedder.generate_embedding(question), dtype=np.float32)
        if dense.shape[0] != DIMENSION:
            raise ValueError(
                f"Question embedding dimension was {dense.shape[0]}, expected {DIMENSION}."
            )
        sparse = vectorizer.transform([question]).tocsr()
        sparse_row = sparse_payload(sparse)
        # Dot-product index: scale the query-side dense and sparse vectors to blend scores.
        dense_weighted = (dense * ALPHA).tolist()
        sparse_weighted = {
            "indices": sparse_row["indices"],
            "values": [v * (1.0 - ALPHA) for v in sparse_row["values"]],
        }
        response = index.query(
            vector=dense_weighted,
            sparse_vector=sparse_weighted,
            top_k=TOP_K,
            namespace=NAMESPACE,
            include_metadata=True,
        )
        results = []
        for rank, match in enumerate(response.matches, start=1):
            metadata = match.metadata or {}
            results.append(
                {
                    "rank": rank,
                    "chunk_id": match.id,
                    "title": metadata.get("title"),
                    "url": metadata.get("url"),
                    "category": metadata.get("category"),
                    "status": metadata.get("status", "unspecified"),
                    "score": float(match.score),
                    "content": metadata.get("content"),
                    "document_id": metadata.get("document_id"),
                }
            )
        retrieval_dataset.append({**qdata, "results": results})
        print(f"  [{position}/{len(questions)}] {question} -> {len(results)} results")
    save_json(HYBRID_RETRIEVAL_FILE, retrieval_dataset)
    print("Hybrid retrieval results saved:", HYBRID_RETRIEVAL_FILE)
    return retrieval_dataset


def evaluate_hybrid(retrieval_dataset):
    print(
        "\n[STAGE 4/4] Evaluating hybrid retrieval with the existing V5 relevance judge..."
    )
    # Reuse the user's existing judge, including Groq -> OpenRouter fallback and
    # independent quote verification. No new judging methodology is introduced.
    sys.path.insert(0, str(ROOT))
    from scripts.run_retrieval_benchmark_v5_with_fallback_provider_tracking import (
        create_groq_client,
        judge_question,
    )

    client = create_groq_client()
    report = []
    if HYBRID_REPORT_FILE.exists():
        report = load_json(HYBRID_REPORT_FILE)
        if len(report) > len(retrieval_dataset):
            raise RuntimeError(
                "Hybrid report has more records than the current question dataset."
            )
        for i, saved in enumerate(report):
            if saved.get("question") != retrieval_dataset[i].get("question"):
                raise RuntimeError(
                    "Existing hybrid report does not match the current question set."
                )
        print(
            f"Resuming hybrid evaluation at question {len(report) + 1}/{len(retrieval_dataset)}"
        )

    for position, qdata in enumerate(
        retrieval_dataset[len(report) :], start=len(report) + 1
    ):
        print(f"\n  Judging [{position}/{len(retrieval_dataset)}]: {qdata['question']}")
        if not qdata.get("results"):
            judged = []
        else:
            judgements = judge_question(client, qdata)
            judged = []
            for rank, result in enumerate(qdata["results"], start=1):
                item = judgements[rank]
                judged.append(
                    {
                        **result,
                        "relevance": item["relevance"],
                        "judge_reason": item["reason"],
                        "judge_evidence": item["evidence"],
                        "evidence_verified": item["evidence_verified"],
                        "judge_provider": item["judge_provider"],
                        "judge_model": item["judge_model"],
                    }
                )
                print(
                    f"    rank {rank}: grade {item['relevance']} - {result.get('title')}"
                )
        docs = aggregate_documents_max_grade(judged)
        report.append(
            {
                **qdata,
                "results": judged,
                "metrics": metric_values(judged),
                "unique_document_count": len(
                    {r.get("url") for r in judged if r.get("url")}
                ),
                "document_metrics": metric_values(docs),
            }
        )
        save_json(HYBRID_REPORT_FILE, report)
        time.sleep(INTER_REQUEST_DELAY_SECONDS)

    metric_names = ["precision_at_5", "mrr_at_5", "ndcg_at_5", "hit_at_5"]
    summary = {
        "index": HYBRID_INDEX_NAME,
        "namespace": NAMESPACE,
        "alpha_dense_weight": ALPHA,
        "questions_evaluated": len(report),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "evaluation_type": "AI-assisted graded relevance; same V5 judge; independent evidence verification",
        "chunk_level": {},
        "document_level": {},
        "judge_provider_counts": {},
        "evidence_audit": {},
    }
    for level, field in (
        ("chunk_level", "metrics"),
        ("document_level", "document_metrics"),
    ):
        for metric in metric_names:
            vals = [r[field][metric] for r in report]
            summary[level][metric] = float(np.mean(vals)) if vals else 0.0

    all_results = [r for q in report for r in q.get("results", [])]
    grade2plus = [
        r for r in all_results if r.get("relevance", 0) >= RELEVANCE_THRESHOLD
    ]
    verified = [r for r in grade2plus if r.get("evidence_verified") is True]
    unverified = [r for r in grade2plus if r.get("evidence_verified") is False]
    for r in all_results:
        provider = r.get("judge_provider", "unknown")
        summary["judge_provider_counts"][provider] = (
            summary["judge_provider_counts"].get(provider, 0) + 1
        )
    summary["evidence_audit"] = {
        "grade_2_or_3_judgements": len(grade2plus),
        "verified_quotes": len(verified),
        "unverified_or_missing_quotes": len(unverified),
        "verification_rate": len(verified) / len(grade2plus) if grade2plus else 0.0,
        "note": "Evidence verification does not alter relevance grades or retrieval metrics.",
    }
    save_json(HYBRID_SUMMARY_FILE, summary)

    with HYBRID_CSV_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "question_id",
            "category",
            "question",
            "rank",
            "title",
            "url",
            "score",
            "relevance",
            "judge_reason",
            "judge_evidence",
            "evidence_verified",
            "judge_provider",
            "judge_model",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for q in report:
            for r in q.get("results", []):
                writer.writerow(
                    {
                        "question_id": q.get("question_id"),
                        "category": q.get("category"),
                        "question": q.get("question"),
                        "rank": r.get("rank"),
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "score": r.get("score"),
                        "relevance": r.get("relevance"),
                        "judge_reason": r.get("judge_reason"),
                        "judge_evidence": r.get("judge_evidence"),
                        "evidence_verified": r.get("evidence_verified"),
                        "judge_provider": r.get("judge_provider"),
                        "judge_model": r.get("judge_model"),
                    }
                )

    return summary


def compare_summaries(baseline, hybrid):
    comparison = {
        "baseline": {
            "name": "V5 dense-only baseline (document metrics corrected)",
            "chunk_level": baseline["chunk_level"],
            "document_level": baseline["document_level"],
        },
        "hybrid": {
            "name": "TF-IDF sparse + SentenceTransformer dense",
            "chunk_level": hybrid["chunk_level"],
            "document_level": hybrid["document_level"],
        },
        "delta_hybrid_minus_baseline": {"chunk_level": {}, "document_level": {}},
        "caveat": "The baseline and hybrid runs use the same question set and V5 judge, but the hybrid results are newly judged and AI-graded relevance remains an approximation. Compare per-question results as well as aggregate metrics.",
    }
    for level in ("chunk_level", "document_level"):
        for metric in baseline[level]:
            comparison["delta_hybrid_minus_baseline"][level][metric] = round(
                hybrid[level][metric] - baseline[level][metric], 6
            )
    save_json(COMPARISON_FILE, comparison)
    return comparison


def main():
    print("=" * 78)
    print("N8N DOCUMENTATION RAG — FOUR-STAGE HYBRID RETRIEVAL EXPERIMENT")
    print("=" * 78)
    required = [CORPUS_FILE, QUESTIONS_FILE, BASELINE_REPORT_FILE]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required project files are missing:\n- " + "\n- ".join(missing)
        )

    chunks = load_json(CORPUS_FILE)
    questions = load_json(QUESTIONS_FILE)
    if not chunks or not questions:
        raise RuntimeError("Corpus or evaluation question set is empty.")
    print(f"Corpus chunks: {len(chunks):,}")
    print(f"Evaluation questions: {len(questions)}")
    print(f"New index: {HYBRID_INDEX_NAME} (dotproduct; existing index is untouched)")

    confirmation = input(
        "\nThis will create/use a separate Pinecone index, upload a second copy of "
        f"{len(chunks):,} vectors, and run {len(questions)} AI-judged questions. "
        "Type RUN HYBRID EXPERIMENT to proceed: "
    )
    if confirmation != "RUN HYBRID EXPERIMENT":
        print("Cancelled. No changes made.")
        return

    baseline = corrected_baseline()
    pc = pinecone_client()
    index = ensure_hybrid_index(pc)
    vectorizer = build_and_upload_sparse_vectors(chunks, index)
    retrieval_dataset = retrieve_hybrid_questions(questions, vectorizer, index)
    hybrid = evaluate_hybrid(retrieval_dataset)
    comparison = compare_summaries(baseline, hybrid)

    print("\n" + "=" * 78)
    print("FOUR-STAGE EXPERIMENT COMPLETED")
    print("=" * 78)
    print("Corrected dense-only baseline:", baseline["document_level"])
    print("Hybrid document-level:", hybrid["document_level"])
    print(
        "Hybrid minus baseline:",
        comparison["delta_hybrid_minus_baseline"]["document_level"],
    )
    print("\nGenerated files:")
    for path in (
        BASELINE_SUMMARY_FILE,
        VECTORIZER_FILE,
        HYBRID_RETRIEVAL_FILE,
        HYBRID_REPORT_FILE,
        HYBRID_CSV_FILE,
        HYBRID_SUMMARY_FILE,
        COMPARISON_FILE,
    ):
        print(" -", path)


if __name__ == "__main__":
    main()
