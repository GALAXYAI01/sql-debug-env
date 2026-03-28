"""
Graders for the SQL Debug & Optimize RL Environment.

grade_easy_medium() — 100% deterministic: execute agent's query against
                      a fresh SQLite DB and compare result sets to the
                      reference query's output. No LLM required.

grade_hard()        — Hybrid:
                        Correctness (0–0.6)  via result set comparison
                        Optimisation (0–0.4) via LLM judge examining
                        whether the agent eliminated the inefficiency.

All graders return GradeResult(score, feedback, sub_scores).
"""

from __future__ import annotations

import json
import re
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .db_fixtures import get_db, run_query, results_match


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GradeResult:
    score: float
    feedback: str
    sub_scores: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM helpers  (only used for HARD optimisation scoring)
# ---------------------------------------------------------------------------

def _make_openai_client():
    import os
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ.get("HF_TOKEN", "EMPTY"),
        base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
    )


def _get_model_name() -> str:
    import os
    return os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


def _strip_sql_fence(text: str) -> str:
    """Remove ```sql ... ``` or ``` ... ``` fences if present."""
    m = re.search(r"```(?:sql)?\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# EASY + MEDIUM grader — deterministic result-set comparison
# ---------------------------------------------------------------------------

def grade_easy_medium(
    task_id: str,
    agent_sql: str,
    correct_sql: str,
    order_sensitive: bool = False,
) -> GradeResult:
    """
    1. Spin up a fresh in-memory SQLite DB for this task.
    2. Run the reference query → expected result set.
    3. Run the agent's query → got result set.
    4. Compare result sets.
    Score: 1.0 if match, 0.0 if mismatch or error.
    Partial credit: 0.5 if row count matches but values differ slightly.
    """
    agent_sql = _strip_sql_fence(agent_sql)

    # --- run reference ---
    try:
        ref_conn = get_db(task_id)
        exp_cols, exp_rows = run_query(ref_conn, correct_sql)
        ref_conn.close()
    except Exception as exc:
        return GradeResult(
            score=0.0,
            feedback=f"[Internal] Reference query failed: {exc}",
            sub_scores={"deterministic": 0.0},
        )

    # --- run agent ---
    try:
        agent_conn = get_db(task_id)
        got_cols, got_rows = run_query(agent_conn, agent_sql)
        agent_conn.close()
    except Exception as exc:
        return GradeResult(
            score=0.0,
            feedback=f"Agent query failed to execute: {exc}",
            sub_scores={"deterministic": 0.0},
        )

    # --- compare ---
    match, reason = results_match(got_cols, got_rows, exp_cols, exp_rows, order_sensitive)

    if match:
        return GradeResult(
            score=1.0,
            feedback=(
                f"✅ Query correct. Result set matches reference.\n"
                f"   {len(exp_rows)} rows × {len(exp_cols)} columns returned."
            ),
            sub_scores={"deterministic": 1.0},
        )

    # Partial credit: correct row count but wrong values
    partial = 0.3 if len(got_rows) == len(exp_rows) else 0.0
    return GradeResult(
        score=partial,
        feedback=(
            f"❌ Result mismatch.\n{reason}\n"
            f"Expected {len(exp_rows)} rows, got {len(got_rows)} rows.\n"
            + (f"Partial credit ({partial}) for correct row count." if partial else "")
        ),
        sub_scores={"deterministic": partial},
    )


# ---------------------------------------------------------------------------
# HARD grader — correctness (0–0.6) + LLM optimisation (0–0.4)
# ---------------------------------------------------------------------------

def grade_hard(
    task_id: str,
    agent_sql: str,
    agent_explanation: Optional[str],
    correct_sql: str,
    order_sensitive: bool,
    optimised_approach: str,
    buggy_query: str,
) -> GradeResult:
    """
    Part 1: Correctness (0–0.6) — same deterministic comparison as EASY/MEDIUM.
    Part 2: Optimisation (0–0.4) — LLM judges whether the agent eliminated the
            performance problem (correlated subquery, N+1, repeated scan, etc.)
    """
    agent_sql = _strip_sql_fence(agent_sql)

    # ---- Part 1: correctness ----
    try:
        ref_conn = get_db(task_id)
        exp_cols, exp_rows = run_query(ref_conn, correct_sql)
        ref_conn.close()
    except Exception as exc:
        return GradeResult(
            score=0.0,
            feedback=f"[Internal] Reference query failed: {exc}",
        )

    correctness_score = 0.0
    correctness_feedback = ""
    try:
        agent_conn = get_db(task_id)
        got_cols, got_rows = run_query(agent_conn, agent_sql)
        agent_conn.close()
        match, reason = results_match(got_cols, got_rows, exp_cols, exp_rows, order_sensitive)
        if match:
            correctness_score = 0.6
            correctness_feedback = f"✅ Correct results ({len(exp_rows)} rows match)."
        else:
            partial_prog = 0.2 if len(got_rows) == len(exp_rows) else 0.0
            correctness_score = partial_prog
            correctness_feedback = f"❌ {reason}"
    except Exception as exc:
        correctness_feedback = f"Agent query failed: {exc}"

    # ---- Part 2: LLM optimisation judge ----
    client = _make_openai_client()
    model = _get_model_name()

    system_prompt = textwrap.dedent("""
        You are a senior database engineer acting as an RL grader.
        Evaluate whether the submitted SQL query makes a meaningful performance
        optimisation over the original buggy query.
        Respond ONLY with valid JSON (no markdown, no prose).
        JSON format: {"opt_score": <float 0-10>, "feedback": "<one sentence>"}
    """).strip()

    user_prompt = textwrap.dedent(f"""
        ORIGINAL BUGGY QUERY:
        {buggy_query}

        EXPECTED OPTIMISATION:
        {optimised_approach}

        SUBMITTED QUERY:
        {agent_sql}

        AGENT'S EXPLANATION (may be empty):
        {agent_explanation or ""}

        Score the optimisation from 0 to 10:
        - 10: Correctly implements the expected or equivalent optimisation (CTE, window fn, JOIN).
        - 7–9: Meaningful improvement, largely eliminates the inefficiency.
        - 4–6: Some improvement but correlated/repeated subquery partially remains.
        - 1–3: Minimal change, original inefficiency is still present.
        - 0: No optimisation attempt, or query is worse.
    """).strip()

    opt_score_raw = 0.0
    opt_feedback = "LLM judge unavailable."
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`")
        data = json.loads(raw)
        opt_score_raw = float(data.get("opt_score", 0))
        opt_feedback = str(data.get("feedback", "No feedback."))
    except Exception as exc:
        opt_feedback = f"LLM judge failed: {exc}"

    opt_score = round(max(0.0, min(0.4, (opt_score_raw / 10.0) * 0.4)), 4)
    total = round(correctness_score + opt_score, 4)

    feedback = (
        f"Correctness: {correctness_feedback} → {correctness_score:.2f}/0.60\n"
        f"Optimisation (LLM judge): {opt_score_raw:.1f}/10 → {opt_score:.2f}/0.40\n"
        f"Total reward: {total:.4f}\n"
        f"Optimisation feedback: {opt_feedback}"
    )

    return GradeResult(
        score=total,
        feedback=feedback,
        sub_scores={
            "correctness": correctness_score,
            "llm_optimisation": opt_score,
        },
    )
