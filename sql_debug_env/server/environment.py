"""
Core environment logic for the SQL Debug & Optimize RL Environment.
"""

from __future__ import annotations

import uuid
from typing import Optional

try:
    from openenv.core.env_server import Environment
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False
    class Environment:  # type: ignore[no-redef]
        pass

from ..models import SQLDebugAction, SQLDebugObservation, SQLDebugState, TableSchema, ColumnInfo
from .tasks import ALL_TASKS
from .graders import grade_easy_medium, grade_hard


# ---------------------------------------------------------------------------
# Schema introspection helpers
# ---------------------------------------------------------------------------

def _get_schema_for_task(task: dict) -> list[TableSchema]:
    """Return the table schemas for the DB associated with this task."""
    from .db_fixtures import get_db
    conn = get_db(task["task_id"])
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    table_names = [r[0] for r in cursor.fetchall()]

    schemas = []
    for tname in table_names:
        cursor.execute(f"PRAGMA table_info({tname});")
        cols = [
            ColumnInfo(name=row[1], type=row[2])
            for row in cursor.fetchall()
        ]
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;",
            (tname,),
        )
        row = cursor.fetchone()
        create_sql = row[0] if row else ""
        schemas.append(TableSchema(table_name=tname, columns=cols, create_sql=create_sql))
    conn.close()
    return schemas


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SQLDebugEnvironment(Environment):
    """
    RL environment for SQL debugging and optimisation.

    Episode: 9 tasks in sequence (easy_001 → hard_003).
    Agent receives the buggy SQL + full schema at each step.
    """

    def __init__(self) -> None:
        if _OPENENV_AVAILABLE:
            super().__init__()
        self._state = SQLDebugState()
        self._tasks = ALL_TASKS

    def reset(self, seed: Optional[int] = None) -> SQLDebugObservation:
        self._state = SQLDebugState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            total_reward=0.0,
            current_task_index=0,
            tasks_completed=0,
            task_scores=[],
            task_ids=[],
        )
        return self._build_observation(0)

    def step(self, action: SQLDebugAction) -> SQLDebugObservation:
        task = self._tasks[self._state.current_task_index]
        difficulty = task["difficulty"]

        if difficulty in ("easy", "medium"):
            result = grade_easy_medium(
                task_id=task["task_id"],
                agent_sql=action.fixed_query,
                correct_sql=task["correct_query"],
                order_sensitive=task["order_sensitive"],
            )
        else:  # hard
            result = grade_hard(
                task_id=task["task_id"],
                agent_sql=action.fixed_query,
                agent_explanation=action.explanation,
                correct_sql=task["correct_query"],
                order_sensitive=task["order_sensitive"],
                optimised_approach=task["optimised_approach"],
                buggy_query=task["buggy_query"],
            )

        self._state.step_count += 1
        self._state.total_reward += result.score
        self._state.tasks_completed += 1
        self._state.task_scores.append(result.score)
        self._state.task_ids.append(task["task_id"])

        next_index = self._state.current_task_index + 1
        done = next_index >= len(self._tasks)
        self._state.current_task_index = next_index

        if done:
            return SQLDebugObservation(
                task_id=task["task_id"],
                difficulty=difficulty,
                task_prompt="Episode complete.",
                buggy_query="",
                reward=result.score,
                done=True,
                feedback=(
                    f"{result.feedback}\n\n"
                    f"=== EPISODE SUMMARY ===\n"
                    f"Tasks completed: {self._state.tasks_completed}/{len(self._tasks)}\n"
                    f"Total reward:    {self._state.total_reward:.4f}\n"
                    f"Mean reward:     {self._state.total_reward / self._state.tasks_completed:.4f}\n"
                    f"Per-task:        {dict(zip(self._state.task_ids, self._state.task_scores))}"
                ),
                success=result.score >= 0.5,
            )

        next_obs = self._build_observation(next_index)
        return SQLDebugObservation(
            task_id=next_obs.task_id,
            difficulty=next_obs.difficulty,
            task_prompt=next_obs.task_prompt,
            buggy_query=next_obs.buggy_query,
            db_schema=next_obs.db_schema,
            sample_data_description=next_obs.sample_data_description,
            reward=result.score,
            done=False,
            feedback=result.feedback,
            success=result.score >= 0.5,
        )

    @property
    def state(self) -> SQLDebugState:
        return self._state.model_copy()

    def _build_observation(self, task_index: int) -> SQLDebugObservation:
        task = self._tasks[task_index]
        schemas = _get_schema_for_task(task)
        return SQLDebugObservation(
            task_id=task["task_id"],
            difficulty=task["difficulty"],
            task_prompt=task["task_prompt"],
            buggy_query=task["buggy_query"],
            db_schema=schemas,
            sample_data_description=task.get("sample_data_description", ""),
            reward=0.0,
            done=False,
            feedback="",
        )

    def close(self) -> None:
        """Release any resources held by the environment. Safe to call multiple times."""
        self._state = SQLDebugState()
