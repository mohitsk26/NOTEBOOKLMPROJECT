import os
from typing import List
from config import LLM_MODEL  # centralized model config

# ---------------------------
# Groq Import with Safe Stub
# ---------------------------
try:
    from groq import Groq
except Exception:
    # Fallback stub if Groq not installed
    class _ChatCompletions:
        def create(self, model, messages, temperature=0.0):
            raise RuntimeError(
                "Groq package not installed. Run: pip install groq"
            )

    class _Chat:
        def __init__(self):
            self.completions = _ChatCompletions()

    class Groq:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.chat = _Chat()


# ---------------------------
# Generator Class
# ---------------------------
class GroqGenerator:
    def __init__(self):
        """
        Initialize LLM client
        """

        # STEP 1: Load API key
        api_key = os.getenv("GROQ_API_KEY")

        # STEP 2: Validate key (fail-fast)
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        # STEP 3: Initialize client
        self.client = Groq(api_key=api_key)

        # STEP 4: Use config-based model
        self.qa_model = LLM_MODEL
        self.summary_model = LLM_MODEL

    # ---------------------------
    # Query Rewriting (Agentic step)
    # ---------------------------
    def rewrite_query(self, query: str) -> str:
        """
        Improve user query for better retrieval
        """

        prompt = f"""
Rewrite the following question to be precise and optimized
for retrieving relevant information from a document.

Question:
{query}

Rewritten question:
"""

        try:
            response = self.client.chat.completions.create(
                model=self.qa_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()

        except Exception:
            # fallback → return original query
            return query

    # ---------------------------
    # Grounded Answer Generator
    # ---------------------------
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate answer using retrieved context
        """

        if not context_chunks:
            return "No relevant information found."

        # STEP 1: Limit context (token safety)
        context_chunks = context_chunks[:5]

        # STEP 2: Label chunks
        labeled_context = ""
        for i, chunk in enumerate(context_chunks, 1):
            labeled_context += f"[Chunk {i}] {chunk}\n\n"

        # STEP 3: Build grounded prompt
        prompt = f"""
You are a document-grounded AI assistant.

RULES:
- Answer ONLY using the provided context.
- Do NOT use external knowledge.
- If the answer is not present, say:
  "Information not found in the uploaded documents."
- Mention chunk numbers used at the end.

Context:
{labeled_context}

Question:
{query}

Answer:
"""

        try:
            response = self.client.chat.completions.create(
                model=self.qa_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            return response.choices[0].message.content.strip()

        except Exception:
            return "Error generating response."

    # ---------------------------
    # Hierarchical Summarization
    # ---------------------------
    def summarize_document(self, context_chunks: List[str]) -> str:
        """
        Token-safe summarization using batching
        """

        if not context_chunks:
            return "No content available for summarization."

        partial_summaries = []

        # STEP 1: Process chunks in batches
        for i in range(0, len(context_chunks), 5):
            batch = context_chunks[i:i + 5]
            context = "\n\n".join(batch)

            prompt = f"""
Summarize the following content in concise bullet points.
Focus only on key ideas.

Content:
{context}

Summary:
"""

            try:
                response = self.client.chat.completions.create(
                    model=self.summary_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )

                partial_summaries.append(
                    response.choices[0].message.content.strip()
                )

            except Exception:
                continue

        # STEP 2: Combine summaries
        combined_text = "\n\n".join(partial_summaries)

        final_prompt = f"""
Combine the following summaries into a single,
clear and concise document summary.

Summaries:
{combined_text}

Final Summary:
"""

        try:
            final_response = self.client.chat.completions.create(
                model=self.summary_model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.3,
            )

            return final_response.choices[0].message.content.strip()

        except Exception:
            return "Error generating summary."

    # ---------------------------
    # Question Generation
    # ---------------------------
    def generate_questions(self, context_chunks: List[str]) -> List[str]:
        """
        Generate useful questions from document
        """

        if not context_chunks:
            return []

        context = "\n\n".join(context_chunks[:5])

        prompt = f"""
Based on the following document content,
generate 5 useful questions a student might ask.

Content:
{context}

Questions:
"""

        try:
            response = self.client.chat.completions.create(
                model=self.qa_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )

            questions_text = response.choices[0].message.content.strip()

            return [
                q.strip("- ").strip()
                for q in questions_text.split("\n")
                if q.strip()
            ]

        except Exception:
            return []
