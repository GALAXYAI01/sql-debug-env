"""FastAPI server for the SQL Debug & Optimize RL Environment."""
from __future__ import annotations
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from sql_debug_env.server.environment import SQLDebugEnvironment
from sql_debug_env.models import SQLDebugAction, SQLDebugObservation, SQLDebugState

app = FastAPI(title="SQL Debug & Optimize RL Environment", version="1.0.0")

_env = SQLDebugEnvironment()

class ResetRequest(BaseModel):
    seed: Optional[int] = None

class StepRequest(BaseModel):
    fixed_query: str
    explanation: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok", "env": "sql_debug_env", "version": "1.0.0"}

@app.get("/reset")
@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    obs = _env.reset(seed=req.seed)
    return obs.model_dump()

@app.post("/step")
def step(req: StepRequest):
    action = SQLDebugAction(fixed_query=req.fixed_query, explanation=req.explanation)
    obs = _env.step(action)
    return obs.model_dump()

@app.get("/state")
def state():
    return _env.state.model_dump()

@app.get("/tasks")
def list_tasks():
    from sql_debug_env.server.tasks import ALL_TASKS
    return [
        {
            "task_id": t["task_id"],
            "difficulty": t["difficulty"],
            "task_prompt": t["task_prompt"],
            "schema_hint": t["schema_hint"],
            "grader": {
                "type": "programmatic" if t["difficulty"] in ("easy", "medium") else "hybrid",
                "score_range": [0.0, 1.0],
                "deterministic": t["difficulty"] in ("easy", "medium")
            }
        }
        for t in ALL_TASKS
    ]

@app.get("/graders")
def list_graders():
    from sql_debug_env.server.tasks import ALL_TASKS
    return [
        {
            "task_id": t["task_id"],
            "grader_type": "programmatic" if t["difficulty"] in ("easy", "medium") else "hybrid",
            "score_min": 0.0,
            "score_max": 1.0,
        }
        for t in ALL_TASKS
    ]

def main():
    import uvicorn
    uvicorn.run("sql_debug_env.server.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))

if __name__ == "__main__":
    main()
