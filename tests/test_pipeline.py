"""Tests for data preparation, retrieval, token budgeting and answer validation,
using the real Forest and Carbon dataset (FAO 2022, 1992-2020)."""

from collections import Counter

import pandas as pd
import pytest

from app.data import Document, load_documents
from app.prompt import build_context, estimate_tokens
from app.retriever import Retriever
from app.validation import validate_answer
from scripts.prepare_data import prepare

DATA = "data/forest_carbon_long.csv"


@pytest.fixture(scope="module")
def docs():
    return load_documents(DATA)


@pytest.fixture(scope="module")
def retriever(docs):
    return Retriever(docs)


# --- Data preparation -------------------------------------------------------

def test_prepare_reshapes_wide_data_to_one_row_per_year(tmp_path):
    out = prepare("data/raw/forest_carbon_cleaned.csv", tmp_path / "long.csv")
    assert list(out.columns) == ["country", "area_type", "indicator", "unit", "year", "value"]
    assert len(out) == 1333 * 29  # 1,333 series x 29 years (1992-2020)
    assert (out.year.min(), out.year.max()) == (1992, 2020)


def test_prepare_labels_aggregates_as_regional_groups():
    df = pd.read_csv(DATA)
    groups = set(df[df.area_type == "regional group"].country)
    assert {"World", "G20", "G7", "Euro Area", "Africa", "Asia"} <= groups
    assert "Ireland" not in groups


# --- Sources ----------------------------------------------------------------

def test_builds_rows_summaries_and_rankings(docs):
    assert Counter(d.kind for d in docs) == {"row": 38657, "summary": 1333, "ranking": 20}


def test_summary_shows_change_with_percentage(docs):
    ireland = next(
        d for d in docs
        if d.kind == "summary" and "country: Ireland;" in d.text and "indicator: Forest area;" in d.text
    )
    assert "from 1992 to 2020" in ireland.text
    assert "495.38 to 782.02 (gained 286.64, +57.9%)" in ireland.text


def test_absolute_ranking_compares_countries_not_groups(docs):
    ranking = next(
        d for d in docs
        if d.kind == "ranking" and "absolute change" in d.text
        and "area_type: country; indicator: Forest area;" in d.text
    )
    losses = ranking.text.split("largest losses (lost the most): ")[1]
    assert losses.startswith("Brazil -84716.52 (-14.6%)")
    assert "World" not in ranking.text and "G20" not in ranking.text


# --- Retrieval --------------------------------------------------------------

def test_country_and_indicator_filter_finds_exact_row(retriever):
    top = retriever.search("Ireland forest area in 2010", top_k=5)[0]
    assert top.text == (
        "country: Ireland; area_type: country; indicator: Forest area; "
        "unit: 1000 HA; year: 2010; value: 720.38"
    )


def test_official_names_match_short_names(retriever):
    assert "china" in retriever.labels_in("Forest area in China")["country"]
    top = retriever.search("share of forest area in China in 2020", top_k=5)[0]
    assert "China, P.R.: Mainland" in top.text and "year: 2020" in top.text


def test_plural_matches_label(retriever):
    assert retriever.labels_in("Which countries gained forest?")["area_type"] == {"country"}


def test_change_question_puts_summary_first(retriever):
    top = retriever.search("How did forest area change in Ireland?", top_k=5)[0]
    assert top.kind == "summary" and "Ireland" in top.text


def test_most_question_puts_country_ranking_first(retriever):
    top = retriever.search("Which country lost the most forest area?", top_k=5)[0]
    assert top.kind == "ranking" and "area_type: country; indicator: Forest area;" in top.text


def test_unrelated_question_returns_nothing(retriever):
    assert retriever.search("zzzz qqqq", top_k=5) == []


# --- Token budget and validation -------------------------------------------

def test_context_stays_within_token_budget():
    docs = [Document(i, "x" * 400) for i in range(1, 50)]  # ~100 tokens each
    context, ids = build_context(docs, max_tokens=350)
    assert estimate_tokens(context) <= 350
    assert ids == [1, 2, 3]


def test_validation_accepts_correct_citations():
    check = validate_answer("Forest area was 495.38 [source 1] and 782.02 [source 2].", [1, 2, 3])
    assert check.grounded and check.cited_sources == [1, 2]


def test_validation_flags_invented_citation():
    check = validate_answer("It was 50% [source 99].", [1, 2, 3])
    assert not check.grounded and check.invalid_citations == [99]


def test_validation_flags_answer_without_citations():
    assert not validate_answer("Forest cover fell sharply.", [1, 2]).grounded


def test_validation_accepts_honest_refusal():
    assert validate_answer("I don't have that in the dataset.", []).grounded
