"""Check AI output before showing it: is the answer grounded in the data we sent?

LLMs can invent numbers. We ask the model to cite sources as [source N], then verify
that every cited source was actually in the context. Answers with no citations,
or citations to sources we never sent, are flagged as unverified.
"""

import re
from dataclasses import dataclass

CITATION = re.compile(r"\[source (\d+)\]")
NO_ANSWER = "i don't have that in the dataset"


@dataclass(frozen=True)
class Validation:
    grounded: bool
    cited_sources: list[int]
    invalid_citations: list[int]
    reason: str


def validate_answer(answer: str, context_ids: list[int]) -> Validation:
    cited = sorted({int(n) for n in CITATION.findall(answer)})
    allowed = set(context_ids)
    invalid = [s for s in cited if s not in allowed]

    if NO_ANSWER in answer.lower():
        return Validation(True, cited, invalid, "Model declined: answer not in dataset.")
    if not cited:
        return Validation(False, cited, invalid, "No citations: figures cannot be checked.")
    if invalid:
        return Validation(False, cited, invalid, "Cites sources that were not provided.")
    return Validation(True, cited, invalid, "All citations match provided sources.")
