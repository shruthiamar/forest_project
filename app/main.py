"""HTTP API: POST /ask runs the full pipeline; GET /metrics shows the KPIs."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import Settings
from app.data import load_documents
from app.llm import LLMClient, LLMError, make_client
from app.metrics import Metrics
from app.prompt import SYSTEM_PROMPT, build_context, build_user_message
from app.retriever import Retriever
from app.validation import validate_answer

STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def create_app(settings: Settings | None = None, client: LLMClient | None = None) -> FastAPI:
    settings = settings or Settings()
    documents = load_documents(settings.data_path)
    retriever = Retriever(documents)
    llm = client or make_client(settings)
    metrics = Metrics(settings.cost_per_mtok_input, settings.cost_per_mtok_output)
    by_id = {d.source_id: d.text for d in documents}

    app = FastAPI(title="Forest & Carbon Q&A")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "sources_loaded": len(documents), "provider": settings.llm_provider}

    @app.post("/ask")
    def ask(req: AskRequest):
        # 1. Retrieve the most relevant sources. 2. Fit them into the token budget.
        hits = retriever.search(req.question, settings.top_k)
        context, source_ids = build_context(hits, settings.max_context_tokens)

        # 3. Call the LLM through the provider-agnostic client.
        try:
            result = llm.complete(
                SYSTEM_PROMPT, build_user_message(req.question, context), settings.max_output_tokens
            )
        except LLMError as exc:
            metrics.record_error()
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        # 4. Validate the answer against the sources we actually sent, then record KPIs.
        check = validate_answer(result.text, source_ids)
        metrics.record(result.input_tokens, result.output_tokens, result.latency_ms, check.grounded)

        return {
            "answer": result.text,
            "validation": {
                "grounded": check.grounded,
                "reason": check.reason,
                "cited_sources": check.cited_sources,
                "invalid_citations": check.invalid_citations,
            },
            "sources": [{"id": s, "text": by_id[s]} for s in check.cited_sources if s in by_id],
            "usage": {
                "provider": result.provider,
                "model": result.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": round(result.latency_ms, 1),
                "est_cost_usd": round(metrics.cost_usd(result.input_tokens, result.output_tokens), 6),
                "context_sources_sent": len(source_ids),
            },
        }

    @app.get("/metrics")
    def get_metrics():
        return metrics.summary()

    return app


app = create_app()
