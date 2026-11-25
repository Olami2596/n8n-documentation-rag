import os
import logging
import re
import streamlit as st
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

logger = logging.getLogger(__name__)

def get_secret(key):
    """
    Retrieve credentials from Streamlit Secrets
    or environment variables.
    """

    try:
        value = st.secrets.get(key)
        if value:
            return value
    except Exception:
        pass

    return os.getenv(key)

class AnswerGenerator:

    def __init__(self, model="openai/gpt-oss-20b"):

        load_dotenv(Path(".env"))

        # Primary model: Groq
        groq_api_key = get_secret("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY was not found.")

        self.client = Groq(api_key=groq_api_key)
        self.model = model

        # Backup model: OpenRouter
        self.fallback_model = "nvidia/nemotron-3-ultra-550b-a55b:free"

        openrouter_api_key = get_secret("OPENROUTER_API_KEY")

        self.fallback_client = None

        if openrouter_api_key:
            self.fallback_client = OpenAI(
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )

    # --------------------------------------------------
    # 1. MODEL COMPLETION WITH AUTOMATIC FALLBACK
    # --------------------------------------------------

    def _create_completion(self, **kwargs):
        """
        Try Groq first.

        If Groq returns a rate-limit error (HTTP 429),
        automatically retry the request using OpenRouter.
        """

        try:

            return self.client.chat.completions.create(**kwargs)

        except Exception as error:

            status_code = getattr(error, "status_code", None)
            error_code = getattr(error, "code", None)

            is_rate_limited = status_code == 429 or error_code == "rate_limit_exceeded"

            if not is_rate_limited:
                raise

            logger.warning(
                "Groq rate limit reached. " "Attempting OpenRouter fallback."
            )

            if not self.fallback_client:

                logger.error(
                    "OpenRouter fallback is unavailable. "
                    "OPENROUTER_API_KEY is not configured."
                )

                raise

            fallback_kwargs = kwargs.copy()

            # Remove Groq-specific parameters.
            fallback_kwargs.pop("include_reasoning", None)

            # Switch to the backup model.
            fallback_kwargs["model"] = self.fallback_model

            try:

                completion = self.fallback_client.chat.completions.create(
                    **fallback_kwargs
                )

                logger.info("Request successfully processed using OpenRouter.")

                return completion

            except Exception:

                logger.exception("OpenRouter fallback request failed.")

                raise

    # --------------------------------------------------
    # 2. QUESTION REWRITING
    # --------------------------------------------------

    def rewrite_question(self, question, conversation_history=None):
        """
        Convert a follow-up question into a standalone
        question suitable for document retrieval.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # No conversation history means no rewriting is needed.
        if not conversation_history:
            return question

        recent_history = conversation_history[-6:]

        history_text = "\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in recent_history
            if message.get("role") in ("user", "assistant") and message.get("content")
        )

        prompt = f"""
You are a search-query rewriting assistant for an n8n
documentation RAG system.

Your task is to rewrite the user's latest question into
a standalone search query that can be understood without
reading the conversation history.

CONVERSATION HISTORY:
{history_text}

LATEST USER QUESTION:
{question}

INSTRUCTIONS:

1. Resolve references such as "it", "that", "this", and
   "the second option" using the conversation history.
2. Preserve the user's original intent.
3. Include relevant technical terms where appropriate.
4. Do not answer the question.
5. Do not introduce unrelated topics.
6. If the question is already standalone, return it unchanged.
7. Return ONLY the rewritten question.
"""

        try:

            completion = self._create_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                include_reasoning=False,
            )

            rewritten_question = completion.choices[0].message.content

            if not rewritten_question or not rewritten_question.strip():
                return question

            return rewritten_question.strip()

        except Exception:

            logger.exception(
                "Question rewriting failed. " "Using the original question."
            )

            return question

    # --------------------------------------------------
    # 3. CITATION VALIDATION
    # --------------------------------------------------

    def validate_citations(self, answer, sources):
        """
        Remove citation references that do not correspond
        to available source references.
        """

        valid_references = {str(source["reference"]) for source in sources}

        def replace_citation(match):

            reference = match.group(1)

            if reference in valid_references:
                return match.group(0)

            return ""

        return re.sub(r"\[(\d+)\]", replace_citation, answer)

    # --------------------------------------------------
    # 4. ANSWER GENERATION
    # --------------------------------------------------

    def generate_answer(self, question, retrieval_response, conversation_history=None):

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        results = retrieval_response.get("results", [])

        deprecated_sources = retrieval_response.get("deprecated_sources", [])

        # Exclude deprecated and unusable documents.
        active_results = [
            result
            for result in results
            if result.get("status") != "deprecated"
            and result.get("url")
            and isinstance(result.get("content"), str)
            and result["content"].strip()
        ]

        # Handle cases where no usable documentation exists.
        if not active_results:

            if deprecated_sources:

                answer = (
                    "I found relevant documentation, but it is marked "
                    "as deprecated. I cannot use it to provide current "
                    "instructions. Please consult the latest n8n "
                    "documentation."
                )

            else:

                answer = (
                    "I couldn't find relevant information in the n8n "
                    "documentation to answer that question."
                )

            return {
                "answer": answer,
                "sources": [],
                "has_deprecated_sources": bool(deprecated_sources),
                "deprecated_sources": deprecated_sources,
            }

        # Prepare documentation context and source references.
        sources = []
        context_parts = []
        seen_urls = {}

        for result in active_results:

            url = result["url"]

            if url not in seen_urls:

                reference = len(sources) + 1
                seen_urls[url] = reference

                sources.append(
                    {
                        "reference": reference,
                        "title": result.get("title") or "Untitled document",
                        "url": url,
                        "status": result.get("status", "unspecified"),
                    }
                )

            source_number = seen_urls[url]

            context_parts.append(f"""
[Source {source_number}]
Title: {result.get('title')}
URL: {url}
Content:
{result['content']}
""")

        if not context_parts:

            return {
                "answer": (
                    "I couldn't find usable documentation sources "
                    "to answer that question."
                ),
                "sources": [],
                "has_deprecated_sources": bool(deprecated_sources),
                "deprecated_sources": deprecated_sources,
            }

        context = "\n\n".join(context_parts)

        system_prompt = """
You are an expert n8n documentation assistant.

Answer questions using ONLY the documentation provided
as factual evidence.

Follow these rules:

1. Use only the supplied documentation as factual evidence.
2. Never invent features, instructions, or technical details.
3. If the documentation does not contain enough information,
   clearly explain what is missing.
4. Explain technical concepts in simple, beginner-friendly language.
5. Use numbered steps when explaining processes.
6. Cite supporting documentation using [1], [2], etc.
7. Only use source numbers provided in the current context.
8. Never invent URLs or citations.
9. Treat conversation history as context for understanding
   the user's question, not as factual documentation.
10. Do not claim that unspecified documents are verified as current.
11. Distinguish documented facts from explanations.
12. Do not cite a source unless it supports the statement.

Be accurate, helpful, and concise.
"""

        messages = [{"role": "system", "content": system_prompt}]

        # Include recent conversation history.
        if conversation_history:

            recent_history = conversation_history[-10:]

            for message in recent_history:

                role = message.get("role")
                content = message.get("content")

                if role in ("user", "assistant") and content:

                    messages.append({"role": role, "content": content})

        # Add current question and retrieved documentation.
        user_prompt = f"""
DOCUMENTATION CONTEXT:

{context}

CURRENT USER QUESTION:

{question}

Use the conversation history to understand the user's intent
and references to earlier messages.

However, use ONLY the documentation context above as
factual evidence for your answer.

Include numbered source citations where appropriate.
Only cite sources that directly support your statements.
"""

        messages.append({"role": "user", "content": user_prompt})

        # Generate answer using Groq or OpenRouter fallback.
        completion = self._create_completion(
            model=self.model,
            messages=messages,
            temperature=0.2,
            include_reasoning=False,
        )

        answer = completion.choices[0].message.content

        if not answer or not answer.strip():

            raise ValueError("The language model returned an empty answer.")

        # Validate citation numbers against available sources.
        answer = self.validate_citations(answer, sources)

        return {
            "answer": answer,
            "sources": sources,
            "has_deprecated_sources": bool(deprecated_sources),
            "deprecated_sources": deprecated_sources,
        }
