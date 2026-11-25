import csv
import json
import math
import os
import re
import time
import unicodedata
import requests
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(".env"))
MODEL = "openai/gpt-oss-20b"
INPUT_FILE = Path("data/processed/retrieval_evaluation_results_v3.json")
REPORT_FILE = Path("data/processed/retrieval_benchmark_report_v5.json")
CSV_FILE = Path("data/processed/retrieval_benchmark_results_v5.csv")
SUMMARY_FILE = Path("data/processed/retrieval_benchmark_summary_v5.json")
RELEVANCE_THRESHOLD = 2
MAX_RETRIES = 3
EVALUATOR_VERSION = (
    "v5-independent-evidence-validation-resumable-openrouter-nemotron-fallback"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
INTER_REQUEST_DELAY_SECONDS = 2.5
RATE_LIMIT_WAIT_SECONDS = 65
MAX_CONTENT_CHARS = 4000


def create_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY was not found.")
    return Groq(api_key=api_key)


def call_openrouter(prompt):
    """Call the same model through OpenRouter when Groq is quota-limited."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Groq quota was reached, but OPENROUTER_API_KEY is not configured. "
            "Add it to .env and rerun; the saved report will resume safely."
        )

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "N8N Documentation RAG Retrieval Benchmark",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(
            f"OpenRouter request failed ({response.status_code}): {response.text[:1200]}"
        )
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def extract_json(response_text):
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def normalize_quote(value):
    value = unicodedata.normalize("NFKC", value or "").casefold()
    return " ".join(value.split())


def judge_question(client, question_data):
    """Score relevance independently; quote checking never changes the score."""
    question = question_data["question"]
    documents = question_data.get("results", [])
    source_contents = {}
    document_context = []

    for i, document in enumerate(documents, start=1):
        content = (document.get("content") or "")[:MAX_CONTENT_CHARS]
        source_contents[i] = content
        document_context.append(
            {
                "result_number": i,
                "title": document.get("title"),
                "url": document.get("url"),
                "category": document.get("category"),
                "content": content,
            }
        )

    prompt = f"""You are a strict information-retrieval evaluator for technical documentation.
Evaluate each retrieved chunk independently against the user's exact task.

USER QUESTION:
{question}

EXPECTED TOPIC (secondary guidance only):
{question_data.get('expected_topic', '')}

EXPECTED KEYWORDS (secondary guidance only):
{json.dumps(question_data.get('expected_keywords', []), ensure_ascii=False)}

RETRIEVED DOCUMENTATION:
{json.dumps(document_context, ensure_ascii=False)}

RELEVANCE SCALE:
0 = Irrelevant; does not help answer the specific question.
1 = Topically related, but gives no concrete task-useful information.
2 = Relevant supporting information; concrete, task-specific information
    that helps answer part of the question (setting, prerequisite, parameter,
    or procedural step).
3 = Directly relevant; explicitly documents the requested feature, operation,
    configuration, API endpoint, or procedure.

STRICT RULES:
1. Identify the precise operation requested, not just the general topic.
2. Distinguish similar operations: receiving a request with a Webhook trigger
   differs from sending a response with Respond to Webhook; connecting n8n to
   Google Sheets differs from deploying n8n to Google Cloud Run; stopping
   executions through the API differs from stopping them through the UI;
   listing executions differs from retrying or stopping them.
3. Shared product, integration, node family, API, or technology alone does
   not justify grade 2 or 3.
4. Grade 3 requires explicit, direct relevance. Grade 2 requires concrete
   supporting information for this exact task.
5. Use chunk content as primary evidence. Titles, URLs, categories, expected
   topic and keywords are navigation clues only.
6. Do not infer missing steps, endpoints, parameters, or capabilities. Do not
   assume the full page contains information absent from this chunk.
7. Grade every result independently. Expected topic/keywords never override
   the question or actual content.

EVIDENCE IS A SEPARATE AUDIT, NOT A CONDITION FOR THE SCORE:
- First assign relevance based only on the content and task.
- For grades 2 and 3, provide a short exact quotation if you can identify one.
  If unsure, use an empty string. For grades 0 and 1, use an empty string.
- Never lower or raise a relevance grade because a quotation is difficult to
  copy. Python checks quotations separately after scoring.
- Explain in 'reason' why the chunk does or does not support the task.

Return ONLY valid JSON in this structure:
{{"judgements":[{{"result_number":1,"relevance":0,
"reason":"Brief task-specific explanation","evidence":""}}]}}
Return exactly one judgement per result. Relevance must be an integer 0-3."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            judge_provider = "groq"
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    include_reasoning=False,
                )
                response_text = completion.choices[0].message.content
            except Exception as groq_error:
                groq_text = str(groq_error).lower()
                groq_rate_limited = (
                    getattr(groq_error, "status_code", None) == 429
                    or "429" in groq_text
                    or "rate_limit" in groq_text
                    or "tokens per day" in groq_text
                )
                if not groq_rate_limited:
                    raise
                print("  Groq rate limit detected. Switching to OpenRouter backup...")
                response_text = call_openrouter(prompt)
                judge_provider = "openrouter"

            response = extract_json(response_text)
            judgements = response.get("judgements", [])
            if not isinstance(judgements, list) or len(judgements) != len(documents):
                raise ValueError(
                    "The judge returned an incorrect number of judgements."
                )

            validated = {}
            for item in judgements:
                number = item.get("result_number")
                score = item.get("relevance")
                reason = item.get("reason")
                evidence = item.get("evidence", "")
                if type(number) is not int or not 1 <= number <= len(documents):
                    raise ValueError("Invalid result number returned.")
                if type(score) is not int or score not in (0, 1, 2, 3):
                    raise ValueError("Invalid relevance score returned.")
                if number in validated:
                    raise ValueError("Duplicate judgement returned.")
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(f"Missing explanation for result {number}.")
                if not isinstance(evidence, str):
                    raise ValueError(f"Invalid evidence field for result {number}.")

                evidence = evidence.strip()
                if score >= RELEVANCE_THRESHOLD:
                    quote = normalize_quote(evidence)
                    content = normalize_quote(source_contents[number])
                    evidence_verified = bool(quote) and quote in content
                else:
                    evidence_verified = None

                validated[number] = {
                    "relevance": score,
                    "reason": reason.strip(),
                    "evidence": evidence,
                    "evidence_verified": evidence_verified,
                    "judge_provider": judge_provider,
                    "judge_model": (
                        MODEL if judge_provider == "groq" else OPENROUTER_MODEL
                    ),
                }

            if len(validated) != len(documents):
                raise ValueError("Missing or duplicate judgements.")
            return validated

        except Exception as error:
            error_text = str(error).lower()
            is_rate_limit = (
                getattr(error, "status_code", None) == 429
                or "429" in error_text
                or "rate_limit" in error_text
            )
            print(f"  Judge attempt {attempt}/{MAX_RETRIES} failed: {error}")
            if attempt == MAX_RETRIES:
                raise
            if is_rate_limit:
                # Prefer Groq's own reset estimate, e.g. "try again in 10m4.8s".
                retry_match = re.search(
                    r"try again in (?:(\d+)m)?([\d.]+)s",
                    str(error),
                    flags=re.IGNORECASE,
                )
                if retry_match:
                    minutes = int(retry_match.group(1) or 0)
                    seconds = float(retry_match.group(2) or 0)
                    wait = int(minutes * 60 + seconds) + 15
                else:
                    wait = RATE_LIMIT_WAIT_SECONDS * attempt
            else:
                wait = 5 * attempt

            print(f"  Retrying in {wait} seconds.")
            time.sleep(wait)


def calculate_metrics(results):
    if not results:
        return {"precision_at_5": 0.0, "mrr_at_5": 0.0, "ndcg_at_5": 0.0, "hit_at_5": 0}
    grades = [item["relevance"] for item in results[:5]]
    relevant = [grade >= RELEVANCE_THRESHOLD for grade in grades]
    precision = sum(relevant) / 5
    mrr = next((1 / rank for rank, flag in enumerate(relevant, 1) if flag), 0.0)
    dcg = sum(
        (2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1)
    )
    ideal = sorted(grades, reverse=True)
    idcg = sum(
        (2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1)
    )
    return {
        "precision_at_5": precision,
        "mrr_at_5": mrr,
        "ndcg_at_5": dcg / idcg if idcg else 0.0,
        "hit_at_5": int(any(relevant)),
    }


def deduplicate_by_url(results):
    unique, seen = [], set()
    for item in results:
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(item)
    return unique[:5]


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Evaluation results not found: {INPUT_FILE}")
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    print("=" * 75)
    print("N8N DOCUMENTATION RAG - FINAL RELEVANCE BENCHMARK V5")
    print("=" * 75)
    print(f"Questions: {len(questions)} | Model: {MODEL}")
    print(f"Input: {INPUT_FILE}")
    print("Evidence verification is independent of relevance scoring.\n")

    client = create_groq_client()

    # Resume from the saved report. Each completed question is checkpointed.
    report = []
    if REPORT_FILE.exists():
        with REPORT_FILE.open("r", encoding="utf-8") as file:
            report = json.load(file)

        if not isinstance(report, list) or len(report) > len(questions):
            raise ValueError(
                "The existing V5 report is invalid or contains more questions "
                "than the current evaluation dataset."
            )

        for index, saved_item in enumerate(report):
            if saved_item.get("question") != questions[index].get("question"):
                raise ValueError(
                    "The saved V5 report does not match the current question set. "
                    "Do not overwrite it; move/rename the report if you intend "
                    "to start a completely new evaluation."
                )

        print(
            f"Resuming from checkpoint: {len(report)} of {len(questions)} "
            "questions already completed.\\n"
        )

    for position, question_data in enumerate(
        questions[len(report) :], start=len(report) + 1
    ):
        print(f'\\n[{position}/{len(questions)}] {question_data["question"]}')
        results = question_data.get("results", [])
        if not results:
            print("  No retrieved results.")
            report.append(
                {
                    **question_data,
                    "results": [],
                    "metrics": calculate_metrics([]),
                    "unique_document_count": 0,
                    "document_metrics": calculate_metrics([]),
                }
            )
            save_json(REPORT_FILE, report)
            continue

        judgements = judge_question(client, question_data)
        judged_results = []
        for rank, result in enumerate(results, start=1):
            judgement = judgements[rank]
            judged = {
                **result,
                "rank": rank,
                "relevance": judgement["relevance"],
                "judge_reason": judgement["reason"],
                "judge_evidence": judgement["evidence"],
                "evidence_verified": judgement["evidence_verified"],
                "judge_provider": judgement["judge_provider"],
                "judge_model": judgement["judge_model"],
            }
            judged_results.append(judged)
            verified = judgement["evidence_verified"]
            label = (
                "quote verified"
                if verified is True
                else "quote not verified" if verified is False else "quote not required"
            )
            print(
                f'  Rank {rank}: Grade {judgement["relevance"]} - '
                f'{result.get("title")} [{label}]'
            )

        unique_docs = deduplicate_by_url(judged_results)
        report.append(
            {
                **question_data,
                "results": judged_results,
                "metrics": calculate_metrics(judged_results),
                "unique_document_count": len(unique_docs),
                "document_metrics": calculate_metrics(unique_docs),
            }
        )
        save_json(REPORT_FILE, report)  # Save after every question.
        time.sleep(INTER_REQUEST_DELAY_SECONDS)

    metric_names = ["precision_at_5", "mrr_at_5", "ndcg_at_5", "hit_at_5"]
    summary = {
        "model": MODEL,
        "primary_provider": "Groq",
        "primary_model": MODEL,
        "backup_provider": "OpenRouter",
        "backup_model": OPENROUTER_MODEL,
        "questions_evaluated": len(report),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "evaluation_type": "AI-assisted graded relevance; independent evidence verification",
        "evaluator_version": EVALUATOR_VERSION,
        "input_file": str(INPUT_FILE),
        "chunk_level": {},
        "document_level": {},
        "evidence_audit": {},
        "judge_provider_counts": {},
    }
    for level, field in [
        ("chunk_level", "metrics"),
        ("document_level", "document_metrics"),
    ]:
        for metric in metric_names:
            values = [item[field][metric] for item in report if field in item]
            summary[level][metric] = sum(values) / len(values) if values else 0.0

    # Backfill provider metadata for the completed V5 run. The final question was
    # visibly evaluated through OpenRouter after Groq's daily quota was exhausted.
    # Earlier saved results predate provider fields and were evaluated by Groq.
    for question_item in report:
        question_text = question_item.get("question", "")
        inferred_provider = (
            "openrouter"
            if question_text
            == "How do I troubleshoot an AI agent that produces incorrect results?"
            else "groq"
        )
        for result_item in question_item.get("results", []):
            result_item.setdefault("judge_provider", inferred_provider)
            result_item.setdefault(
                "judge_model",
                OPENROUTER_MODEL if inferred_provider == "openrouter" else MODEL,
            )

    all_results = [result for item in report for result in item.get("results", [])]
    grade_2_3 = [
        item for item in all_results if item.get("relevance", 0) >= RELEVANCE_THRESHOLD
    ]
    verified = [item for item in grade_2_3 if item.get("evidence_verified") is True]
    unverified = [item for item in grade_2_3 if item.get("evidence_verified") is False]
    for item in all_results:
        provider = item.get("judge_provider", "groq")
        summary["judge_provider_counts"][provider] = (
            summary["judge_provider_counts"].get(provider, 0) + 1
        )

    summary["evidence_audit"] = {
        "grade_2_or_3_judgements": len(grade_2_3),
        "verified_quotes": len(verified),
        "unverified_or_missing_quotes": len(unverified),
        "verification_rate": len(verified) / len(grade_2_3) if grade_2_3 else 0.0,
        "note": "Evidence verification does not alter relevance grades or retrieval metrics.",
    }

    fields = [
        "question_id",
        "category",
        "question",
        "rank",
        "title",
        "url",
        "similarity_score",
        "relevance",
        "judge_reason",
        "judge_evidence",
        "evidence_verified",
        "judge_provider",
        "judge_model",
    ]
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for question in report:
            for result in question.get("results", []):
                writer.writerow(
                    {
                        "question_id": question.get("question_id"),
                        "category": question.get("category"),
                        "question": question.get("question"),
                        "rank": result.get("rank"),
                        "title": result.get("title"),
                        "url": result.get("url"),
                        "similarity_score": result.get("score"),
                        "relevance": result.get("relevance"),
                        "judge_reason": result.get("judge_reason"),
                        "judge_evidence": result.get("judge_evidence"),
                        "evidence_verified": result.get("evidence_verified"),
                        "judge_provider": result.get("judge_provider"),
                        "judge_model": result.get("judge_model"),
                    }
                )
    save_json(SUMMARY_FILE, summary)

    print("\n" + "=" * 75 + "\nV5 BENCHMARK COMPLETED\n" + "=" * 75)
    print(f'Questions evaluated: {summary["questions_evaluated"]}')
    for label in ("chunk_level", "document_level"):
        print(f"\n{label.upper()}")
        for metric, value in summary[label].items():
            print(f"{metric}: {value:.4f}")
    print("\nJUDGE PROVIDERS")
    print(summary["judge_provider_counts"])
    print("\nEVIDENCE AUDIT")
    for metric, value in summary["evidence_audit"].items():
        print(
            f"{metric}: {value:.4f}"
            if isinstance(value, float)
            else f"{metric}: {value}"
        )
    print("\nFILES CREATED")
    print(f"Detailed report: {REPORT_FILE}")
    print(f"CSV results: {CSV_FILE}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
