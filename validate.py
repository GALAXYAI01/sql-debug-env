#!/usr/bin/env python3
"""validate.py — Pre-submission validator for SQL Debug Env."""

from __future__ import annotations
import sys, os, traceback
from pathlib import Path

PASS="✅"; FAIL="❌"; WARN="⚠️ "
errors=[]; warnings=[]

def check(label, condition, error_msg="", warn=False):
    if condition:
        print(f"  {PASS} {label}"); return True
    sym = WARN if warn else FAIL
    msg = f"  {sym} {label}" + (f"\n       → {error_msg}" if error_msg else "")
    print(msg)
    (warnings if warn else errors).append(label)
    return False

def validate_structure():
    print("\n[1] File structure")
    root = Path(__file__).parent
    for f in ["inference.py","pyproject.toml","sql_debug_env/openenv.yaml",
              "sql_debug_env/__init__.py","sql_debug_env/models.py",
              "sql_debug_env/client.py","sql_debug_env/server/app.py",
              "sql_debug_env/server/environment.py","sql_debug_env/server/tasks.py",
              "sql_debug_env/server/graders.py","sql_debug_env/server/db_fixtures.py",
              "sql_debug_env/server/requirements.txt","sql_debug_env/server/Dockerfile","README.md"]:
        check(f, (root/f).exists())

def validate_yaml():
    print("\n[2] openenv.yaml")
    try:
        import yaml
        with open(Path(__file__).parent/"sql_debug_env"/"openenv.yaml") as fh:
            d = yaml.safe_load(fh)
        for k in ["name","version","client","action","observation","spec_version"]:
            check(f"{k} field present", k in d)
    except ImportError:
        check("PyYAML available", False, "pip install pyyaml")
    except Exception as exc:
        check("openenv.yaml parseable", False, str(exc))

def validate_models():
    print("\n[3] Pydantic models")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from sql_debug_env.models import SQLDebugAction, SQLDebugObservation, SQLDebugState
        a = SQLDebugAction(fixed_query="SELECT 1")
        o = SQLDebugObservation(task_id="t", difficulty="easy", task_prompt="p", buggy_query="q")
        s = SQLDebugState()
        check("SQLDebugAction instantiates", True)
        check("SQLDebugObservation instantiates", True)
        check("SQLDebugState instantiates", True)
        check("action.fixed_query", hasattr(a, "fixed_query"))
        check("obs.reward", hasattr(o, "reward"))
        check("obs.done", hasattr(o, "done"))
    except Exception:
        check("Models import", False, traceback.format_exc(limit=3))

def validate_db_fixtures():
    print("\n[4] DB fixtures + SQL execution")
    try:
        from sql_debug_env.server.db_fixtures import make_ecommerce_db, make_hr_db, make_analytics_db, run_query, results_match
        for name, factory in [("ecommerce", make_ecommerce_db), ("hr", make_hr_db), ("analytics", make_analytics_db)]:
            conn = factory()
            cols, rows = run_query(conn, "SELECT name FROM sqlite_master WHERE type='table'")
            check(f"{name} DB creates OK ({len(rows)} tables)", len(rows) >= 4)
            conn.close()
        # result comparison
        conn = make_ecommerce_db()
        c1, r1 = run_query(conn, "SELECT customer_id FROM customers ORDER BY customer_id")
        c2, r2 = run_query(conn, "SELECT customer_id FROM customers ORDER BY customer_id")
        match, _ = results_match(c1, r1, c2, r2, order_sensitive=True)
        check("results_match identical queries = True", match)
        conn.close()
    except Exception:
        check("DB fixtures work", False, traceback.format_exc(limit=4))

def validate_graders():
    print("\n[5] Graders")
    try:
        from sql_debug_env.server.graders import grade_easy_medium
        from sql_debug_env.server.tasks import EASY_TASKS, MEDIUM_TASKS
        # perfect fix for easy_001
        r = grade_easy_medium("easy_001", EASY_TASKS[0]["correct_query"], EASY_TASKS[0]["correct_query"], False)
        check("grade_easy_medium correct → 1.0", r.score == 1.0, f"got {r.score}")
        # intentionally wrong
        r2 = grade_easy_medium("easy_001", "SELECT 1", EASY_TASKS[0]["correct_query"], False)
        check("grade_easy_medium wrong → < 1.0", r2.score < 1.0)
        # all medium perfect fixes
        for t in MEDIUM_TASKS:
            r3 = grade_easy_medium(t["task_id"], t["correct_query"], t["correct_query"], t["order_sensitive"])
            check(f"{t['task_id']} perfect fix → 1.0", r3.score == 1.0, f"got {r3.score}")
    except Exception:
        check("Graders work", False, traceback.format_exc(limit=4))

def validate_environment():
    print("\n[6] Environment logic")
    try:
        from sql_debug_env.server.environment import SQLDebugEnvironment
        from sql_debug_env.models import SQLDebugAction
        env = SQLDebugEnvironment()
        obs = env.reset()
        check("reset() works", obs is not None)
        check("reset() difficulty valid", obs.difficulty in ("easy","medium","hard"))
        check("reset() done=False", obs.done is False)
        check("reset() has schema", len(obs.db_schema) >= 3)
        s = env.state
        check("state.episode_id set", bool(s.episode_id))
        # step with correct answer
        action = SQLDebugAction(fixed_query="SELECT 1 WHERE 1=0")  # wrong on purpose
        obs2 = env.step(action)
        check("step() returns obs", obs2 is not None)
        check("step() reward in [0,1]", 0.0 <= obs2.reward <= 1.0, f"got {obs2.reward}")
        # full episode
        obs = env.reset()
        steps = 0
        while not obs.done and steps < 20:
            obs = env.step(SQLDebugAction(fixed_query="SELECT 1"))
            steps += 1
        check("Full episode completes", obs.done, f"done=False after {steps} steps")
        check("Exactly 9 tasks", steps == 9, f"steps={steps}")
    except Exception:
        check("Environment logic", False, traceback.format_exc(limit=5))

def validate_inference():
    print("\n[7] inference.py")
    root = Path(__file__).parent
    p = root/"inference.py"
    check("inference.py at root", p.exists())
    if p.exists():
        t = p.read_text()
        check("Uses OpenAI client", "OpenAI" in t)
        check("Reads API_BASE_URL", "API_BASE_URL" in t)
        check("Reads MODEL_NAME",   "MODEL_NAME" in t)
        check("Reads HF_TOKEN",     "HF_TOKEN" in t)
        check("Has main()",         "main()" in t)

def validate_live_server(url):
    print(f"\n[8] Live server at {url}")
    try:
        import httpx
        r = httpx.get(f"{url}/health", timeout=10)
        check("GET /health 200", r.status_code == 200)
        r2 = httpx.post(f"{url}/reset", json={}, timeout=15)
        check("POST /reset 200", r2.status_code == 200)
        obs = r2.json()
        check("reset returns task_id", "task_id" in obs)
        r3 = httpx.post(f"{url}/step", json={"fixed_query": "SELECT 1"}, timeout=60)
        check("POST /step 200", r3.status_code == 200)
        reward = float(r3.json().get("reward", -1))
        check("reward in [0,1]", 0.0 <= reward <= 1.0, f"got {reward}")
    except Exception as exc:
        check("Live server accessible", False, str(exc))

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--server-url", type=str, default=None)
    args = p.parse_args()

    print("="*60)
    print("SQL Debug & Optimize Env — Pre-Submission Validator")
    print("="*60)

    validate_structure()
    validate_yaml()
    validate_models()
    validate_db_fixtures()
    validate_graders()
    validate_environment()
    validate_inference()
    if args.server_url:
        validate_live_server(args.server_url)

    print("\n" + "="*60)
    if errors:
        print(f"❌ FAILED — {len(errors)} error(s):")
        for e in errors: print(f"   • {e}")
        sys.exit(1)
    elif warnings:
        print(f"⚠️  WARNINGS: {warnings}")
    else:
        print("✅ ALL CHECKS PASSED — Ready to submit!")
        sys.exit(0)

if __name__ == "__main__":
    main()
