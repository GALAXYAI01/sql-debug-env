"""FastAPI server for the SQL Debug & Optimize RL Environment."""

from __future__ import annotations
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .environment import SQLDebugEnvironment
from ..models import SQLDebugAction, SQLDebugObservation, SQLDebugState

app = FastAPI(
    title="SQL Debug & Optimize RL Environment",
    description=(
        "OpenEnv-compatible environment for training AI agents to debug and "
        "optimise SQL queries. Core grading is 100% deterministic via SQLite. "
        "Built for the Meta PyTorch × Scaler SST OpenEnv Hackathon 2026."
    ),
    version="1.0.0",
)

_env = SQLDebugEnvironment()


class ResetRequest(BaseModel):
    seed: Optional[int] = None


class StepRequest(BaseModel):
    fixed_query: str
    explanation: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "env": "sql_debug_env", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    obs: SQLDebugObservation = _env.reset(seed=req.seed)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    action = SQLDebugAction(fixed_query=req.fixed_query, explanation=req.explanation)
    obs: SQLDebugObservation = _env.step(action)
    return obs.model_dump()


@app.get("/state")
def state():
    return _env.state.model_dump()


@app.get("/schema")
def schema():
    return {
        "action": SQLDebugAction.model_json_schema(),
        "observation": SQLDebugObservation.model_json_schema(),
        "state": SQLDebugState.model_json_schema(),
    }


@app.get("/tasks")
def list_tasks():
    from .tasks import ALL_TASKS
    return [
        {
            "task_id": t["task_id"],
            "difficulty": t["difficulty"],
            "task_prompt": t["task_prompt"][:120] + "...",
            "schema_hint": t["schema_hint"],
        }
        for t in ALL_TASKS
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
