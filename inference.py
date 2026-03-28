#!/usr/bin/env python3
"""
inference.py — Mandatory root-level inference script.

Runs an LLM agent through all 9 SQL Debug tasks and emits the
exact stdout format required by the OpenEnv hackathon evaluator:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Required environment variables:
    API_BASE_URL         OpenAI-compatible LLM endpoint
    MODEL_NAME           Model identifier
    HF_TOKEN             Hugging Face / API key
    LOCAL_IMAGE_NAME     (optional) Docker image name for from_docker_image() mode

Usage:
    python inference.py                              # in-process (default, no server needed)
    python inference.py --server-url http://...      # via running HTTP server
    python inference.py --docker                     # spin up Docker container automatically
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import textwrap
from typing import List, Optional

# ---------------------------------------------------------------------------
# Load .env file (if present) so secrets don't need manual export each run
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — fall back to real env vars

# ---------------------------------------------------------------------------
# Validate required env vars early — fail fast before any imports
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN     = os.getenv("HF_TOKEN")     or os.getenv("API_KEY") or ""
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "sql-debug-env:latest")

TASK_NAME  = "sql_debug"
BENCHMARK  = "sql_debug_env"
MAX_STEPS  = 9          # one step per task (9 tasks total in the episode)
SUCCESS_SCORE_THRESHOLD = 0.5

if not HF_TOKEN:
    print("[WARN] HF_TOKEN not set — LLM calls for HARD tasks will fail", flush=True)

from openai import OpenAI  # noqa: E402
llm_client = OpenAI(api_key=HF_TOKEN or "EMPTY", base_url=API_BASE_URL)


# ---------------------------------------------------------------------------
# Mandatory stdout logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    # Truncate action to 120 chars and strip newlines for single-line output
    action_str = action.replace("\n", " ").replace("\r", "").strip()
    if len(action_str) > 120:
        action_str = action_str[:117] + "..."
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Agent — prompts the LLM to fix the SQL
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior SQL developer and database engineer.
    You will be given a broken SQL query and the database schema.
    Your job is to fix the query exactly as instructed.

    For EASY and MEDIUM tasks: return ONLY the corrected SQL inside ```sql``` fences.
    For HARD tasks: return the corrected + optimised SQL inside ```sql``` fences,
    then add a one-sentence comment after the closing fence explaining the optimisation.

    Always return syntactically valid SQL. Never add explanatory prose before the SQL fence.
""").strip()


def _extract_sql(text: str) -> str:
    """Extract SQL from ```sql...``` fences, or return text as-is."""
    m = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def agent_respond(
    task_prompt: str,
    buggy_query: str,
    schema_str: str,
    difficulty: str,
) -> dict:
    """Call the LLM and return {'fixed_query': str, 'explanation': str|None}."""
    if difficulty == "hard":
        user_content = (
            f"DATABASE SCHEMA:\n{schema_str}\n\n"
            f"TASK:\n{task_prompt}\n\n"
            f"BUGGY QUERY:\n```sql\n{buggy_query}\n```\n\n"
            "Fix the bug AND optimise the query. "
            "Return fixed+optimised SQL in ```sql``` fences, "
            "then one sentence explaining your optimisation."
        )
    else:
        user_content = (
            f"DATABASE SCHEMA:\n{schema_str}\n\n"
            f"TASK:\n{task_prompt}\n\n"
            f"BUGGY QUERY:\n```sql\n{buggy_query}\n```\n\n"
            "Return ONLY the corrected SQL in ```sql``` fences."
        )

    try:
        resp = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=800,
            temperature=0.1,
            stream=False,
        )
        full = (resp.choices[0].message.content or "").strip()
        sql_part = _extract_sql(full)
        explanation = full[full.rfind("```") + 3:].strip() if "```" in full else None
        return {"fixed_query": sql_part or buggy_query, "explanation": explanation}
    except Exception as exc:
        return {"fixed_query": buggy_query, "explanation": None, "_error": str(exc)}


def schema_to_str(schema_list) -> str:
    lines = []
    for t in schema_list:
        d = t if isinstance(t, dict) else t.model_dump()
        cols = ", ".join(f"{c['name']} {c['type']}" for c in d.get("columns", []))
        lines.append(f"  {d['table_name']}({cols})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# In-process episode runner (no server/Docker needed)
# ---------------------------------------------------------------------------

async def run_in_process() -> tuple[List[float], int]:
    """Run all 9 tasks directly in-process. Returns (rewards, steps_taken)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from sql_debug_env.server.environment import SQLDebugEnvironment
    from sql_debug_env.models import SQLDebugAction

    env = SQLDebugEnvironment()
    obs = env.reset()

    rewards: List[float] = []
    steps_taken = 0
    task_index = 0

    for step in range(1, MAX_STEPS + 1):
        if obs.done:
            break

        # Get current task metadata for the agent
        schema_str = schema_to_str(obs.db_schema)
        out = agent_respond(
            obs.task_prompt,
            obs.buggy_query,
            schema_str,
            obs.difficulty,
        )
        error_msg = out.pop("_error", None)

        action = SQLDebugAction(**out)
        result_obs = env.step(action)

        reward = result_obs.reward
        done   = result_obs.done
        rewards.append(reward)
        steps_taken = step
        task_index += 1

        log_step(
            step=step,
            action=out["fixed_query"],
            reward=reward,
            done=done,
            error=error_msg,
        )

        obs = result_obs
        if done:
            break

    return rewards, steps_taken


# ---------------------------------------------------------------------------
# HTTP server episode runner
# ---------------------------------------------------------------------------

async def run_via_server(server_url: str) -> tuple[List[float], int]:
    """Run all tasks via an already-running HTTP server."""
    from sql_debug_env.client import SQLDebugEnv
    from sql_debug_env.models import SQLDebugAction

    async with SQLDebugEnv(base_url=server_url) as env:
        result = await env.reset()
        obs = result.observation

        rewards: List[float] = []
        steps_taken = 0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            schema_str = schema_to_str(obs.db_schema)
            out = agent_respond(
                obs.task_prompt,
                obs.buggy_query,
                schema_str,
                obs.difficulty,
            )
            error_msg = out.pop("_error", None)

            action = SQLDebugAction(**out)
            result = await env.step(action)
            obs = result.observation

            reward = result.reward
            done   = result.done
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=out["fixed_query"],
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

    return rewards, steps_taken


# ---------------------------------------------------------------------------
# Docker runner
# ---------------------------------------------------------------------------

async def run_via_docker(image_name: str) -> tuple[List[float], int]:
    """Spin up image_name in Docker, run episode, stop container."""
    from sql_debug_env.client import SQLDebugEnv
    from sql_debug_env.models import SQLDebugAction

    env = await SQLDebugEnv.from_docker_image(image_name)
    rewards: List[float] = []
    steps_taken = 0

    try:
        result = await env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            schema_str = schema_to_str(obs.db_schema)
            out = agent_respond(
                obs.task_prompt, obs.buggy_query, schema_str, obs.difficulty
            )
            error_msg = out.pop("_error", None)

            result = await env.step(SQLDebugAction(**out))
            obs = result.observation

            rewards.append(result.reward)
            steps_taken = step

            log_step(
                step=step,
                action=out["fixed_query"],
                reward=result.reward,
                done=result.done,
                error=error_msg,
            )

            if result.done:
                break

    finally:
        await env.close()

    return rewards, steps_taken


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="SQL Debug & Optimize RL Environment — inference script"
    )
    parser.add_argument("--server-url", type=str, default=None,
                        help="URL of running server, e.g. http://localhost:7860")
    parser.add_argument("--docker", action="store_true",
                        help="Spin up LOCAL_IMAGE_NAME in Docker automatically")
    parser.add_argument("--image", type=str, default=LOCAL_IMAGE_NAME,
                        help="Docker image to use with --docker (default: LOCAL_IMAGE_NAME env var)")
    args = parser.parse_args()

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        if args.docker:
            rewards, steps_taken = await run_via_docker(args.image)
        elif args.server_url:
            rewards, steps_taken = await run_via_server(args.server_url)
        else:
            rewards, steps_taken = await run_in_process()

        # Normalise score: max possible reward = 9 tasks × 1.0 each = 9.0
        max_total = MAX_STEPS * 1.0
        score = sum(rewards) / max_total if max_total > 0 else 0.0
        score = min(max(score, 0.0), 1.0)   # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode failed: {exc}", flush=True)
        # Still emit [END] even on failure
        success = False
        score = 0.0

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


if __name__ == "__main__":
    asyncio.run(main())
