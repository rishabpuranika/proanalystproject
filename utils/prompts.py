"""
Prompts Utility Module
=======================

Centralises all prompt templates, constants, and formatting logic
used by the RAG pipeline.

Separation of prompts into their own module allows for:
    - Easy iteration on prompt engineering without touching pipeline code.
    - Clear documentation of the system's behavioural constraints.
    - Reusable formatting functions for context assembly.
"""

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
# This prompt establishes the LLM's persona and hard constraints.
# Every rule is designed to minimise hallucination and keep answers
# grounded strictly in the retrieved Upwork API documentation.

SYSTEM_PROMPT: str = """You are a Senior Upwork API Consultant with deep expertise in the Upwork API ecosystem.

Rules:
1. Answer ONLY using the retrieved documentation provided in the context.
2. Never use outside knowledge or make assumptions beyond the documentation.
3. Never fabricate or hallucinate information.
4. If the information is not found in the provided context, respond exactly:
   "I'm sorry, but the provided documentation does not contain that information."
5. Quote relevant documentation sections where useful, using exact text from the context.
6. Keep answers concise, technical, and well-structured.
7. Mention OAuth details, API limitations, rate limits, and permissions exactly as stated in the documentation.
8. When referencing specific sections, cite the page number from the metadata provided.
"""

# ---------------------------------------------------------------------------
# Hallucination Guard
# ---------------------------------------------------------------------------
# This exact message is returned when retrieval confidence is too low
# to generate a reliable answer.  Using a constant ensures consistency
# across the guard check in rag.py and the system prompt instruction.

HALLUCINATION_GUARD_MESSAGE: str = (
    "I'm sorry, but the provided documentation does not contain that information."
)

# ---------------------------------------------------------------------------
# Similarity Threshold
# ---------------------------------------------------------------------------
# ChromaDB returns L2 (Euclidean) distances.  With normalised embeddings,
# L2 distance ranges from 0 (identical) to 2 (opposite).
#
# A threshold of 0.6 means: if the *best* retrieved chunk has a distance
# greater than this value, the query is considered out-of-scope and the
# hallucination guard fires instead of calling the LLM.
#
# Tuning notes:
#   - Lower threshold (e.g. 0.4) = stricter, rejects more queries.
#   - Higher threshold (e.g. 0.8) = more permissive, risks lower-quality answers.
#   - 0.6 was chosen as a balanced default after testing with representative queries.

SIMILARITY_THRESHOLD: float = 1.2


# ---------------------------------------------------------------------------
# Prompt Formatting
# ---------------------------------------------------------------------------

def build_user_prompt(context_chunks: list[dict], question: str) -> str:
    """
    Assemble the user-facing prompt that combines retrieved context with
    the user's question.

    The prompt is structured so the LLM sees:
        1. A clearly delineated CONTEXT block with numbered sources.
        2. The user's QUESTION.
        3. Explicit instructions to answer only from the context.

    Args:
        context_chunks: List of dicts, each with keys:
            - content (str):  The chunk text.
            - page (int):     The source page number.
            - score (float):  The similarity distance.
        question: The user's natural-language question.

    Returns:
        A formatted prompt string ready to send to the LLM.
    """
    # Format each retrieved chunk with its source metadata
    context_parts: list[str] = []
    for idx, chunk in enumerate(context_chunks, start=1):
        page_num = chunk.get("page", "unknown")
        score = chunk.get("score", "N/A")
        content = chunk.get("content", "")
        context_parts.append(
            f"--- Source {idx} (Page {page_num}, Relevance Distance: {score}) ---\n"
            f"{content}"
        )

    # Join all context chunks with clear separators
    formatted_context: str = "\n\n".join(context_parts)

    # Build the final prompt with explicit grounding instructions
    prompt: str = f"""Context:
{formatted_context}

Question:
{question}

Instructions:
Answer ONLY using the context above. Structure your answer clearly.
If the answer cannot be found in the context, respond exactly:
"I'm sorry, but the provided documentation does not contain that information."
"""
    return prompt
