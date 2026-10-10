"""Build the prompt, keeping retrieved context inside a token budget."""

from app.data import Document

SYSTEM_PROMPT = (
    "You answer questions about a forest and carbon dataset. "
    "Use ONLY the dataset sources provided. Cite every figure you use with its source "
    "id in the form [source N], and always state the unit (e.g. 1000 HA means thousand "
    "hectares). If the sources do not contain the answer, say "
    "\"I don't have that in the dataset.\" Keep answers under 150 words."
)


def estimate_tokens(text: str) -> int:
    """Rough estimate (about 4 characters per token for English text).

    Good enough for budgeting before the call; the provider's response gives
    the exact count afterwards, which is what we record in the metrics.
    """
    return max(1, len(text) // 4)


def build_context(docs: list[Document], max_tokens: int) -> tuple[str, list[int]]:
    """Add sources in relevance order until the token budget is used up."""
    lines, used_ids, used = [], [], 0
    for doc in docs:
        line = f"[source {doc.source_id}] {doc.text}"
        cost = estimate_tokens(line)
        if used + cost > max_tokens:
            break
        lines.append(line)
        used_ids.append(doc.source_id)
        used += cost
    return "\n".join(lines), used_ids


def build_user_message(question: str, context: str) -> str:
    return f"Dataset sources:\n{context or '(no matching sources)'}\n\nQuestion: {question}"
