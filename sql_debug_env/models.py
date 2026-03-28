"""
Typed Pydantic models for the SQL Debug & Optimize RL Environment.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SQLDebugAction(BaseModel):
    """
    Agent's response to a SQL task.

    For EASY   tasks: `fixed_query` is the corrected SQL string.
    For MEDIUM tasks: `fixed_query` is the semantically correct SQL string.
    For HARD   tasks: `fixed_query` is the corrected + optimised SQL,
                      `explanation` optionally describes the optimisation.
    """
    fixed_query: str = Field(
        ...,
        description="The agent's corrected (and optionally optimised) SQL query.",
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Optional explanation of changes / optimisations made.",
    )


class ColumnInfo(BaseModel):
    name: str
    type: str


class TableSchema(BaseModel):
    table_name: str
    columns: List[ColumnInfo]
    create_sql: str


class SQLDebugObservation(BaseModel):
    """Everything the agent sees at each step."""

    task_id: str
    difficulty: str                     # "easy" | "medium" | "hard"
    task_prompt: str
    buggy_query: str
    db_schema: List[TableSchema] = Field(default_factory=list)
    sample_data_description: str = ""  # human-readable hint about the data

    # filled after step()
    reward: float = 0.0
    done: bool = False
    feedback: str = ""
    success: bool = False


class SQLDebugState(BaseModel):
    """Internal episode bookkeeping state."""
    episode_id: str = ""
    step_count: int = 0
    total_reward: float = 0.0
    current_task_index: int = 0
    tasks_completed: int = 0
    task_scores: List[float] = Field(default_factory=list)
    task_ids: List[str] = Field(default_factory=list)


class StepResult(BaseModel):
    """
    Standard OpenEnv StepResult wrapper returned by client reset() and step().
    Callers use result.observation, result.reward, result.done — matching
    the official OpenEnv client interface pattern.
    """
    observation: SQLDebugObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
