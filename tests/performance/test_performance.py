# tests/performance/test_performance.py -- CMN-C1-053
# Hybrid retrieval latency benchmark
# Issue #30
"""
Performance benchmark for KBRetrieveHybridNode.

Validates that hybrid retrieval (merge + score + filter) completes within
the acceptable latency ceiling even for moderately sized chunk lists.

Approach: mocks heavy deps (no sentence_transformers / qdrant_client needed);
uses time.perf_counter for portable high-res timing.
"""

from __future__ import annotations

import time
from typing import Any


from src.nodes.kb_retrieve_hybrid import KBRetrieveHybridNode
from src.schemas.state import initial_state
from src.utils.serde import encode_list


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunks(n: int, base_score: float = 0.7, doc_type: str = "openapi_spec") -> list[dict[str, Any]]:
    """Generate n synthetic chunks with deterministic content."""
    return [
        {
            "chunk_text": f"GET /v1/endpoint-{i}\nSummary: Endpoint {i} description for performance test",
            "score": base_score - (i * 0.001),  # slightly decreasing scores
            "source_path": f"kb/api_doc_{i % 10}.json",
            "endpoint_ref": f"/v1/endpoint-{i} GET",
            "doc_type": doc_type,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Latency ceiling
# ---------------------------------------------------------------------------

# Hybrid merge is pure Python — ceiling is 500 ms for 200+200 chunks.
# In practice should be <5 ms.
_LATENCY_CEILING_S = 0.5


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestHybridRetrievalLatency:
    """Benchmark KBRetrieveHybridNode.execute() latency under load."""

    def _run_and_measure(
        self,
        dense_chunks: list[dict[str, Any]],
        sparse_chunks: list[dict[str, Any]],
        repetitions: int = 10,
    ) -> float:
        """Return average latency in seconds over `repetitions` calls."""
        node = KBRetrieveHybridNode()
        state = initial_state("what is the order endpoint?")
        state["dense_chunks"] = encode_list(dense_chunks)
        state["sparse_chunks"] = encode_list(sparse_chunks)

        latencies: list[float] = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            node.execute(state)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        return sum(latencies) / len(latencies)

    def test_small_list_latency(self) -> None:
        """5+5 chunks — typical production case."""
        dense = _make_chunks(5, base_score=0.80)
        sparse = _make_chunks(5, base_score=0.75)
        avg = self._run_and_measure(dense, sparse, repetitions=20)
        assert avg < _LATENCY_CEILING_S, f"Hybrid merge too slow: {avg:.4f}s average (ceiling {_LATENCY_CEILING_S}s)"

    def test_medium_list_latency(self) -> None:
        """50+50 chunks — stress test with moderate load."""
        dense = _make_chunks(50, base_score=0.80)
        sparse = _make_chunks(50, base_score=0.75)
        avg = self._run_and_measure(dense, sparse, repetitions=10)
        assert avg < _LATENCY_CEILING_S, (
            f"Hybrid merge too slow at 50+50: {avg:.4f}s average (ceiling {_LATENCY_CEILING_S}s)"
        )

    def test_large_list_latency(self) -> None:
        """200+200 chunks — upper bound scenario."""
        dense = _make_chunks(200, base_score=0.80)
        sparse = _make_chunks(200, base_score=0.75)
        avg = self._run_and_measure(dense, sparse, repetitions=5)
        assert avg < _LATENCY_CEILING_S, (
            f"Hybrid merge too slow at 200+200: {avg:.4f}s average (ceiling {_LATENCY_CEILING_S}s)"
        )

    def test_empty_input_latency(self) -> None:
        """Empty inputs — should be negligible."""
        avg = self._run_and_measure([], [], repetitions=50)
        assert avg < _LATENCY_CEILING_S, f"Hybrid merge on empty input too slow: {avg:.4f}s"

    def test_latency_with_degraded_path(self) -> None:
        """Chunks with low scores — all filtered, status=degraded path."""
        dense = _make_chunks(50, base_score=0.30)  # all below 0.65 threshold
        sparse = _make_chunks(50, base_score=0.20)
        avg = self._run_and_measure(dense, sparse, repetitions=10)
        assert avg < _LATENCY_CEILING_S, f"Hybrid merge (degraded path) too slow: {avg:.4f}s"

    def test_throughput_calls_per_second(self) -> None:
        """At typical load (5+5 chunks), node should handle >= 50 calls/second."""
        dense = _make_chunks(5, base_score=0.80)
        sparse = _make_chunks(5, base_score=0.75)
        node = KBRetrieveHybridNode()
        state = initial_state("throughput test")
        state["dense_chunks"] = encode_list(dense)
        state["sparse_chunks"] = encode_list(sparse)

        t0 = time.perf_counter()
        n_calls = 100
        for _ in range(n_calls):
            node.execute(state)
        elapsed = time.perf_counter() - t0

        calls_per_second = n_calls / elapsed
        assert calls_per_second >= 50, f"Throughput too low: {calls_per_second:.1f} calls/sec (need >= 50)"


# ---------------------------------------------------------------------------
# Additional performance tests (Issue #30 gap fill)
# ---------------------------------------------------------------------------


class TestHybridRetrievalThroughput:
    """Additional throughput and latency tests for CMN-C1-053."""

    def test_hybrid_retrieval_throughput(self) -> None:
        """50 calls in loop (mock nodes), assert total < 1.0s (>= 50 calls/sec)."""
        from src.schemas.state import initial_state as _init

        # Mock all heavy nodes; only KBRetrieveHybridNode runs real code
        dense = _make_chunks(5, base_score=0.80)
        sparse = _make_chunks(5, base_score=0.75)

        node = KBRetrieveHybridNode()
        state = _init("throughput test 50")
        state["dense_chunks"] = encode_list(dense)
        state["sparse_chunks"] = encode_list(sparse)

        n_calls = 50
        t0 = time.perf_counter()
        for _ in range(n_calls):
            node.execute(state)
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, f"test_hybrid_retrieval_throughput: {n_calls} calls took {elapsed:.3f}s (must be < 1.0s)"

    def test_single_call_p99_latency(self) -> None:
        """100 iterations, p99 latency < 0.5s per call."""
        dense = _make_chunks(10, base_score=0.80)
        sparse = _make_chunks(10, base_score=0.75)

        node = KBRetrieveHybridNode()
        state = initial_state("p99 latency test")
        state["dense_chunks"] = encode_list(dense)
        state["sparse_chunks"] = encode_list(sparse)

        latencies: list[float] = []
        for _ in range(100):
            t0 = time.perf_counter()
            node.execute(state)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        latencies.sort()
        p99 = latencies[int(len(latencies) * 0.99)]
        assert p99 < 0.5, f"test_single_call_p99_latency: p99={p99:.4f}s (must be < 0.5s)"

    def test_state_serialization_speed(self) -> None:
        """1000 msgpack round-trips of full state, assert total < 0.1s."""
        import msgpack

        state = initial_state("serialization speed test")
        state["answer_status"] = "ok"
        state["normalized_query"] = "list orders"
        state["detected_language"] = "en"
        state["dense_chunks"] = encode_list(_make_chunks(5, base_score=0.80))
        state["sparse_chunks"] = encode_list(_make_chunks(5, base_score=0.75))
        state["reranked_chunks"] = encode_list(_make_chunks(3, base_score=0.80))
        state["retrieval_count"] = 3
        state["answer_markdown"] = "Use GET /v1/orders to list orders."
        state["cited_endpoints"] = ["/v1/orders GET"]
        state["code_example"] = None
        state["s1_blocked"] = False
        state["s2_violation"] = False
        state["s3_violation"] = False
        state["credential_leak_detected"] = False
        state["duration_ms"] = 42

        n_iters = 1000
        t0 = time.perf_counter()
        for _ in range(n_iters):
            packed = msgpack.packb(dict(state), use_bin_type=True)
            msgpack.unpackb(packed, raw=False)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.1, (
            f"test_state_serialization_speed: {n_iters} round-trips took {elapsed:.4f}s (must be < 0.1s)"
        )
