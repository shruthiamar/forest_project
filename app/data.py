"""Load the CSV and turn it into short text "sources" the retriever can search.

Three kinds of source, built from a long-format CSV (one value per row):
- Row: one fact, e.g. "country: Ireland; indicator: Forest area; year: 2010; value: 703.2"
- Summary: one per series (e.g. Ireland + Forest area), first vs last year with % change,
  so "how did X change?" questions have a direct answer.
- Ranking: one per measure (e.g. Forest area for countries), listing the largest
  gains and losses, so "which country lost the most?" questions can be answered
  without sending hundreds of rows to the model.

Each source has an id for citations: rows keep their CSV row number (1, 2, 3 ...),
and summaries and rankings are numbered after the last row.
"""

from dataclasses import dataclass, field

import pandas as pd

RANKING_SIZE = 10


@dataclass(frozen=True)
class Document:
    source_id: int
    text: str
    kind: str = "row"  # "row", "summary" or "ranking"
    # (column, lower-cased value) pairs used for metadata filtering, e.g. ("country", "ireland").
    labels: frozenset[tuple[str, str]] = field(default_factory=frozenset)


def _labels(values: dict) -> frozenset[tuple[str, str]]:
    """Each text value, plus a short alias for official names like 'China, P.R.: Mainland'."""
    pairs = set()
    for col, val in values.items():
        if pd.isna(val):
            continue
        name = str(val).strip().lower()
        pairs.add((col, name))
        short = name.split(",")[0].strip()
        if short != name and len(short) >= 4:
            pairs.add((col, short))
    return frozenset(pairs)


def _describe(values: dict) -> str:
    return "; ".join(f"{col}: {val}" for col, val in values.items() if pd.notna(val))


def _change(first: float, last: float) -> tuple[float, float | None]:
    delta = last - first
    pct = None if first == 0 else delta / abs(first) * 100
    return delta, pct


def _num(x: float) -> str:
    """Up to 2 decimals, no trailing zeros: 581336.12, 495.38, 100."""
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _change_text(first: float, last: float) -> str:
    delta, pct = _change(first, last)
    trend = "gained" if delta > 0 else "lost" if delta < 0 else "no change"
    pct_text = "" if pct is None else f", {pct:+.1f}%"
    return f"{_num(first)} to {_num(last)} ({trend} {_num(abs(delta))}{pct_text})"


def load_documents(path: str) -> list[Document]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")

    text_cols = [c for c in df.columns if df[c].dtype == object]
    docs = []
    for i, row in enumerate(df.to_dict("records"), start=1):
        docs.append(Document(i, _describe(row), "row", _labels({c: row[c] for c in text_cols})))

    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    if year_col and text_cols:
        value_cols = [c for c in df.select_dtypes("number").columns if c != year_col]
        docs += _summaries_and_rankings(df, text_cols, year_col, value_cols, next_id=len(docs) + 1)
    return docs


def _summaries_and_rankings(df, text_cols, year_col, value_cols, next_id) -> list[Document]:
    entity_col, measure_cols = text_cols[0], text_cols[1:]
    docs, series = [], []

    # One summary per series (all text columns together identify a series).
    for key, group in df.sort_values(year_col).groupby(text_cols, sort=False, dropna=False):
        key = dict(zip(text_cols, key if isinstance(key, tuple) else (key,)))
        first, last = group.iloc[0], group.iloc[-1]
        y0, y1 = int(first[year_col]), int(last[year_col])
        changes = [f"{c}: {_change_text(first[c], last[c])}" for c in value_cols]
        text = f"summary of change over time for {_describe(key)}, from {y0} to {y1}; " + "; ".join(changes)
        docs.append(Document(next_id, text, "summary", _labels(key)))
        next_id += 1
        series.append((key, y0, y1, first, last))

    # One ranking per measure (e.g. country + Forest area + 1000 HA), by absolute and % change.
    if not measure_cols:
        return docs
    by_measure: dict[tuple, list] = {}
    for key, y0, y1, first, last in series:
        by_measure.setdefault(tuple(key[c] for c in measure_cols), []).append((key, y0, y1, first, last))

    for measure, items in by_measure.items():
        measure_desc = _describe(dict(zip(measure_cols, measure)))
        span = f"{min(i[1] for i in items)} to {max(i[2] for i in items)}"
        for col in value_cols:
            changes = []
            for key, _, _, first, last in items:
                delta, pct = _change(first[col], last[col])
                changes.append((key[entity_col], delta, pct))
            for label, sort_key in (("absolute change", lambda c: c[1]), ("percentage change", lambda c: c[2])):
                ranked = sorted((c for c in changes if c[2] is not None), key=sort_key)
                losses = [c for c in ranked if c[1] < 0][:RANKING_SIZE]
                gains = [c for c in reversed(ranked) if c[1] > 0][:RANKING_SIZE]
                fmt = lambda c: f"{c[0]} {'+' if c[1] > 0 else '-'}{_num(abs(c[1]))} ({c[2]:+.1f}%)"
                text = (
                    f"ranking across {entity_col} by {label} in {col} for {measure_desc}, {span}; "
                    f"largest losses (lost the most): {', '.join(map(fmt, losses)) or 'none'}; "
                    f"largest gains (gained the most): {', '.join(map(fmt, gains)) or 'none'}"
                )
                docs.append(Document(next_id, text, "ranking", _labels(dict(zip(measure_cols, measure)))))
                next_id += 1
    return docs
