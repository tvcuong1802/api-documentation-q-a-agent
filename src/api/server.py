"""src/api/server.py — CMN-C1-053. Standalone HTTP entry (adapter only)."""

from __future__ import annotations

import os
import secrets

from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext, TrustLevel
from src.graph.graph import Graph

app = FastAPI(title="CMN-C1-053 — API Documentation Q&A")
agent = Graph()
agent.compile()


class InvokeRequest(BaseModel):
    input: str
    invocation_id: str = ""


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Any:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL
    ctx = InvocationContext(
        session_id=req.invocation_id or str(uuid4()),
        caller_trust_level=trust,
        caller_id=getattr(request.state, "caller_id", ""),
    )
    return agent.invoke(req.input, ctx=ctx)


@app.get("/health")
def health() -> dict[str, str]:
    return {"answer_status": "ok"}
