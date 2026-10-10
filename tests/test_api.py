"""End-to-end API tests using the offline FakeClient (no API key needed)."""

from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import LLMError, LLMResult
from app.main import create_app

SETTINGS = Settings(llm_provider="fake", data_path="data/forest_carbon_long.csv")


def make_client(llm=None):
    return TestClient(create_app(SETTINGS, client=llm))


def test_health_reports_sources_loaded():
    resp = make_client().get("/health")
    assert resp.status_code == 200
    assert resp.json()["sources_loaded"] == 40010


def test_ask_returns_grounded_answer_with_sources_and_usage():
    client = make_client()
    resp = client.post("/ask", json={"question": "Finland forest area in 2020"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["validation"]["grounded"] is True
    assert body["sources"] and "Finland" in body["sources"][0]["text"]
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["context_sources_sent"] > 0


def test_metrics_track_requests():
    client = make_client()
    client.post("/ask", json={"question": "Brazil carbon stock 1992"})
    client.post("/ask", json={"question": "India forest area 2005"})
    metrics = client.get("/metrics").json()
    assert metrics["requests"] == 2
    assert metrics["grounded_rate"] == 1.0
    assert metrics["input_tokens"] > 0


def test_rejects_empty_question():
    assert make_client().post("/ask", json={"question": ""}).status_code == 422


class HallucinatingLLM:
    def complete(self, system, user, max_tokens):
        return LLMResult("Forest cover doubled [source 9999].", 100, 10, 5.0, "test", "test")


def test_hallucinated_citation_is_flagged():
    body = make_client(HallucinatingLLM()).post(
        "/ask", json={"question": "Brazil forest area"}
    ).json()
    assert body["validation"]["grounded"] is False
    assert body["validation"]["invalid_citations"] == [9999]


class FailingLLM:
    def complete(self, system, user, max_tokens):
        raise LLMError("timeout")


def test_provider_failure_returns_502_and_counts_error():
    client = make_client(FailingLLM())
    assert client.post("/ask", json={"question": "Brazil forest"}).status_code == 502
    assert client.get("/metrics").json()["errors"] == 1
