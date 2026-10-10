"""Retrieval step of RAG: find the dataset sources most relevant to a question.

Three stages:
1. Metadata filter: values from the data named in the question (a country, an
   indicator, a unit) narrow the candidates. Within one column any named value
   matches ("Ireland and Finland"); across columns all must match ("Ireland" AND
   "Forest area"). This stops "Ireland forest area in 2010" returning every
   country's 2010 row. Sources without that column (e.g. rankings, which cover
   many countries) are kept.
2. Intent: questions about change ("how did X change") put summaries first;
   questions asking for the most/least put rankings first.
3. TF-IDF ranking within the candidates, so years and other words still count.
   Words are split on punctuation (forest_area_pct -> forest, area, pct).

TF-IDF needs no API key or vector database and is easy to explain. A production
version could swap in embeddings + a vector store without changing the rest of the app.
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app.data import Document

SUPERLATIVE = re.compile(r"\b(most|least|largest|biggest|smallest|highest|lowest|top|rank\w*)\b", re.I)
TREND = re.compile(
    r"\b(change\w*|trend\w*|increas\w*|decreas\w*|grow\w*|grew|gain\w*|los[st]\w*|declin\w*|since|between|over time)\b",
    re.I,
)


def _label_pattern(value: str) -> re.Pattern:
    """Match a value as a whole phrase, allowing a plural (country/countries, forest/forests)."""
    if value.endswith("y") and value[-2:-1] not in "aeiou":
        body = re.escape(value[:-1]) + "(?:y|ies)"
    else:
        body = re.escape(value) + "(?:s|es)?"
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


def _priority(question: str) -> dict[str, int]:
    if SUPERLATIVE.search(question):
        return {"ranking": 2, "summary": 1, "row": 0}
    if TREND.search(question):
        return {"summary": 2, "ranking": 1, "row": 0}
    return {}


class Retriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(token_pattern=r"[a-z0-9]+", lowercase=True)
        self.matrix = self.vectorizer.fit_transform(d.text for d in documents)
        self.all_labels = {label for d in documents for label in d.labels}
        self._patterns = {
            label: _label_pattern(label[1])
            for label in self.all_labels
        }

    def labels_in(self, question: str) -> dict[str, set[str]]:
        """Values from the data that the question names, grouped by column."""
        q = question.lower()
        wanted: dict[str, set[str]] = {}
        for (col, value), pattern in self._patterns.items():
            if pattern.search(q):
                wanted.setdefault(col, set()).add(value)
        return wanted

    @staticmethod
    def _matches(doc: Document, wanted: dict[str, set[str]]) -> bool:
        doc_cols = {col for col, _ in doc.labels}
        return all(
            col not in doc_cols or any((col, v) in doc.labels for v in values)
            for col, values in wanted.items()
        )

    def search(self, question: str, top_k: int) -> list[Document]:
        scores = linear_kernel(self.vectorizer.transform([question]), self.matrix).ravel()
        wanted = self.labels_in(question)
        candidates = [i for i, d in enumerate(self.documents) if self._matches(d, wanted)]
        priority = _priority(question)
        ranked = sorted(
            candidates,
            key=lambda i: (priority.get(self.documents[i].kind, 0), scores[i]),
            reverse=True,
        )[:top_k]
        return [self.documents[i] for i in ranked if scores[i] > 0]
