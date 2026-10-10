"""In-memory KPIs for the AI workflow: volume, latency, tokens, cost and grounding rate."""

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Metrics:
    cost_per_mtok_input: float = 0.0
    cost_per_mtok_output: float = 0.0
    requests: int = 0
    errors: int = 0
    grounded: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.cost_per_mtok_input + output_tokens * self.cost_per_mtok_output
        ) / 1_000_000

    def record(self, input_tokens: int, output_tokens: int, latency_ms: float, grounded: bool):
        with self._lock:
            self.requests += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.latencies_ms.append(latency_ms)
            self.grounded += int(grounded)

    def record_error(self):
        with self._lock:
            self.errors += 1

    def summary(self) -> dict:
        with self._lock:
            lat = sorted(self.latencies_ms)
            p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0
            return {
                "requests": self.requests,
                "errors": self.errors,
                "grounded_rate": round(self.grounded / self.requests, 3) if self.requests else 0.0,
                "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
                "p95_latency_ms": round(p95, 1),
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "est_cost_usd": round(self.cost_usd(self.input_tokens, self.output_tokens), 6),
            }
