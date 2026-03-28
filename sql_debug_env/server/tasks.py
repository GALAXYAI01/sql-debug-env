"""
Task bank for the SQL Debug & Optimize RL Environment.

9 tasks across 3 tiers:
  EASY   (3) — syntax errors only. Agent returns corrected SQL.
                Graded by: execute both queries, compare result sets.
  MEDIUM (3) — semantic / logic bugs (wrong JOIN, bad GROUP BY, wrong filter).
                Graded by: execute both queries, compare result sets.
  HARD   (3) — logic bug + performance problem (correlated subquery, N+1, full scan).
                Graded by: correctness (0–0.6) via result comparison
                           + LLM judge on query plan / optimisation (0–0.4).

Each task carries:
  buggy_query    — the broken SQL the agent must fix
  correct_query  — the reference answer (used to produce expected result set)
  schema_hint    — which DB to spin up
  order_sensitive — whether result order matters for grading
  optimised_hint  — (HARD only) the expected optimisation approach, for LLM judge
"""

from __future__ import annotations
from typing import List, Dict, Any

Task = Dict[str, Any]

# ---------------------------------------------------------------------------
# EASY — syntax errors
# ---------------------------------------------------------------------------

EASY_TASKS: List[Task] = [
    {
        "task_id": "easy_001",
        "difficulty": "easy",
        "schema_hint": "ecommerce",
        "task_prompt": (
            "The query below should return the name and total revenue "
            "(quantity × unit_price) for every product that has been ordered. "
            "It has a syntax error. Fix it and return only the corrected SQL."
        ),
        "buggy_query": (
            "SELECT p.name, SUM(oi.quantity * oi.unit_price) AS total_revenue\n"
            "FROM products p\n"
            "JOINN order_items oi ON p.product_id = oi.product_id\n"  # JOINN typo
            "GROUP BY p.product_id, p.name\n"
            "ORDER BY total_revenue DESC;"
        ),
        "correct_query": (
            "SELECT p.name, SUM(oi.quantity * oi.unit_price) AS total_revenue\n"
            "FROM products p\n"
            "JOIN order_items oi ON p.product_id = oi.product_id\n"
            "GROUP BY p.product_id, p.name\n"
            "ORDER BY total_revenue DESC;"
        ),
        "order_sensitive": True,
        "bug_description": "JOINN is a typo — should be JOIN.",
        "sample_data_description": (
            "Tables: products(product_id, name, category, price), "
            "order_items(item_id, order_id, product_id, quantity, unit_price). "
            "5 products, 10 order items."
        ),
    },
    {
        "task_id": "easy_002",
        "difficulty": "easy",
        "schema_hint": "hr",
        "task_prompt": (
            "The query below should return each department's name and the number "
            "of employees in it. It has a syntax error. Fix it and return only "
            "the corrected SQL."
        ),
        "buggy_query": (
            "SELECT d.dept_name, COUNT(e.emp_id) AS headcount\n"
            "FROM departments d\n"
            "LEFT JOIN employees e ON d.dept_id = e.dept_id\n"
            "GRUOP BY d.dept_id, d.dept_name;"  # GRUOP typo
        ),
        "correct_query": (
            "SELECT d.dept_name, COUNT(e.emp_id) AS headcount\n"
            "FROM departments d\n"
            "LEFT JOIN employees e ON d.dept_id = e.dept_id\n"
            "GROUP BY d.dept_id, d.dept_name;"
        ),
        "order_sensitive": False,
        "bug_description": "GRUOP is a typo — should be GROUP.",
        "sample_data_description": (
            "Tables: departments(dept_id, dept_name, location), "
            "employees(emp_id, name, dept_id, hire_date, job_title). "
            "4 departments, 7 employees."
        ),
    },
    {
        "task_id": "easy_003",
        "difficulty": "easy",
        "schema_hint": "analytics",
        "task_prompt": (
            "The query below should return the total number of page_views per "
            "user plan (e.g. free, pro, enterprise). It has a syntax error. "
            "Fix it and return only the corrected SQL."
        ),
        "buggy_query": (
            "SELECT u.plan, COUNT(pv.view_id) AS total_views\n"
            "FROM users u\n"
            "JOIN sessions s ON u.user_id = s.user_id\n"
            "LEFFT JOIN page_views pv ON s.session_id = pv.session_id\n"  # LEFFT typo
            "GROUP BY u.plan;"
        ),
        "correct_query": (
            "SELECT u.plan, COUNT(pv.view_id) AS total_views\n"
            "FROM users u\n"
            "JOIN sessions s ON u.user_id = s.user_id\n"
            "JOIN page_views pv ON s.session_id = pv.session_id\n"
            "GROUP BY u.plan;"
        ),
        "order_sensitive": False,
        "bug_description": "ON clause uses == (Python style) instead of = (SQL).",
        "sample_data_description": (
            "Tables: users(user_id, username, plan, created_at), "
            "sessions(session_id, user_id, ...), page_views(view_id, session_id, page, duration_sec, viewed_at). "
            "5 users with plans: free, pro, enterprise."
        ),
    },
]

# ---------------------------------------------------------------------------
# MEDIUM — semantic / logic bugs
# ---------------------------------------------------------------------------

MEDIUM_TASKS: List[Task] = [
    {
        "task_id": "medium_001",
        "difficulty": "medium",
        "schema_hint": "ecommerce",
        "task_prompt": (
            "The query below should return customers who have placed at least "
            "one DELIVERED order, along with their total amount spent across "
            "all delivered orders. It has a logic bug — it returns wrong results. "
            "Fix it and return only the corrected SQL."
        ),
        "buggy_query": (
            "SELECT c.name, SUM(oi.unit_price) AS total_spent\n"  # BUG: missing quantity multiplier\n            "FROM customers c\n"
            "JOIN orders o ON c.customer_id = o.customer_id\n"
            "JOIN order_items oi ON oi.order_id = o.order_id\n"
            "WHERE o.status = 'delivered'\n"
            "GROUP BY c.customer_id, c.name\n"
            "HAVING total_spent > 0\n"
            "ORDER BY total_spent DESC;"
        ),
        "correct_query": (
            "SELECT c.name, SUM(oi.quantity * oi.unit_price) AS total_spent\n"
            "FROM customers c\n"
            "JOIN orders o ON c.customer_id = o.customer_id\n"
            "JOIN order_items oi ON oi.order_id = o.order_id\n"
            "WHERE o.status = 'delivered'\n"
            "GROUP BY c.customer_id, c.name\n"
            "HAVING total_spent > 0\n"
            "ORDER BY total_spent DESC;"
        ),
        "order_sensitive": True,
        "bug_description": "SUM(oi.unit_price) ignores quantity — should be SUM(oi.quantity * oi.unit_price) to get true revenue.",
        "sample_data_description": (
            "customers, orders (status: delivered/pending/cancelled), order_items. "
            "Only delivered orders count."
        ),
    },
    {
        "task_id": "medium_002",
        "difficulty": "medium",
        "schema_hint": "hr",
        "task_prompt": (
            "The query should return, for each department, the average salary "
            "of employees in that department. It currently returns incorrect "
            "averages due to a logic bug. Fix it."
        ),
        "buggy_query": (
            "SELECT d.dept_name, AVG(s.amount) AS avg_salary\n"
            "FROM departments d\n"
            "JOIN employees e ON d.dept_id = e.dept_id\n"
            "JOIN salaries s ON s.emp_id = d.dept_id\n"  # BUG: joining on wrong column (dept_id vs emp_id)
            "GROUP BY d.dept_id, d.dept_name;"
        ),
        "correct_query": (
            "SELECT d.dept_name, AVG(s.amount) AS avg_salary\n"
            "FROM departments d\n"
            "JOIN employees e ON d.dept_id = e.dept_id\n"
            "JOIN salaries s ON s.emp_id = e.emp_id\n"
            "GROUP BY d.dept_id, d.dept_name;"
        ),
        "order_sensitive": False,
        "bug_description": "JOIN condition uses d.dept_id instead of e.emp_id — joining salaries to departments rather than employees.",
        "sample_data_description": (
            "departments, employees, salaries (emp_id, amount, from_date, to_date). "
            "Each employee has one current salary record."
        ),
    },
    {
        "task_id": "medium_003",
        "difficulty": "medium",
        "schema_hint": "analytics",
        "task_prompt": (
            "The query should return the number of unique users who triggered a "
            "'purchase' event, grouped by their plan. The current query returns "
            "wrong counts. Fix the logic bug."
        ),
        "buggy_query": (
            "SELECT u.plan, COUNT(u.user_id) AS purchasers\n"  # BUG: should be COUNT(DISTINCT u.user_id)
            "FROM users u\n"
            "JOIN sessions s ON u.user_id = s.user_id\n"
            "JOIN events e ON e.session_id = s.session_id\n"
            "WHERE e.event_type = 'purchase'\n"
            "GROUP BY u.plan;"
        ),
        "correct_query": (
            "SELECT u.plan, COUNT(DISTINCT u.user_id) AS purchasers\n"
            "FROM users u\n"
            "JOIN sessions s ON u.user_id = s.user_id\n"
            "JOIN events e ON e.session_id = s.session_id\n"
            "WHERE e.event_type = 'purchase'\n"
            "GROUP BY u.plan;"
        ),
        "order_sensitive": False,
        "bug_description": "COUNT(user_id) counts all rows including duplicates; should be COUNT(DISTINCT user_id) to count unique purchasers.",
        "sample_data_description": (
            "users, sessions, events (event_type: signup, purchase, pageview). "
            "Users can have multiple sessions and multiple purchase events."
        ),
    },
]

# ---------------------------------------------------------------------------
# HARD — logic bug + performance optimisation
# ---------------------------------------------------------------------------

HARD_TASKS: List[Task] = [
    {
        "task_id": "hard_001",
        "difficulty": "hard",
        "schema_hint": "ecommerce",
        "task_prompt": (
            "The query below finds the top-spending customer per product category. "
            "It has a logic bug AND uses a correlated subquery that is O(n²). "
            "1) Fix the bug so results are correct. "
            "2) Rewrite to eliminate the correlated subquery — use a JOIN or CTE instead. "
            "Return only the corrected + optimised SQL. "
            "Optionally add a SQL comment explaining the optimisation."
        ),
        "buggy_query": (
            "SELECT p.category, c.name AS top_customer,\n"
            "       (SELECT SUM(oi2.quantity * oi2.unit_price)\n"
            "        FROM order_items oi2\n"
            "        JOIN orders o2 ON oi2.order_id = o2.order_id\n"
            "        WHERE o2.customer_id = o.customer_id\n"
            "          AND EXISTS (SELECT 1 FROM products p2\n"
            "                      WHERE p2.product_id = oi2.product_id\n"
            "                        AND p2.category = p.category)\n"
            "       ) AS spend\n"
            "FROM customers c\n"
            "JOIN orders o ON c.customer_id = o.customer_id\n"
            "JOIN order_items oi ON oi.order_id = o.order_id\n"
            "JOIN products p ON p.product_id = oi.product_id\n"
            "WHERE o.status = 'delivered'\n"
            "GROUP BY p.category, c.customer_id, c.name\n"
            "HAVING spend = (\n"
            "  SELECT MAX(sub.total)\n"
            "  FROM (\n"
            "    SELECT SUM(oi3.quantity * oi3.unit_price) AS total\n"
            "    FROM order_items oi3\n"
            "    JOIN orders o3 ON oi3.order_id = o3.order_id\n"
            "    JOIN products p3 ON p3.product_id = oi3.product_id\n"
            "    WHERE o3.status = 'deliverd'\n"  # BUG: typo 'deliverd'
            "      AND p3.category = p.category\n"
            "    GROUP BY o3.customer_id\n"
            "  ) sub\n"
            ");"
        ),
        "correct_query": (
            "WITH category_spend AS (\n"
            "  SELECT p.category, o.customer_id,\n"
            "         SUM(oi.quantity * oi.unit_price) AS total_spend\n"
            "  FROM order_items oi\n"
            "  JOIN orders o  ON oi.order_id  = o.order_id\n"
            "  JOIN products p ON oi.product_id = p.product_id\n"
            "  WHERE o.status = 'delivered'\n"
            "  GROUP BY p.category, o.customer_id\n"
            "),\n"
            "max_per_category AS (\n"
            "  SELECT category, MAX(total_spend) AS max_spend\n"
            "  FROM category_spend\n"
            "  GROUP BY category\n"
            ")\n"
            "SELECT cs.category, c.name AS top_customer, cs.total_spend AS spend\n"
            "FROM category_spend cs\n"
            "JOIN max_per_category mc ON cs.category = mc.category\n"
            "                        AND cs.total_spend = mc.max_spend\n"
            "JOIN customers c ON c.customer_id = cs.customer_id\n"
            "ORDER BY cs.category;"
        ),
        "order_sensitive": True,
        "bug_description": "Typo 'deliverd' in HAVING subquery means the HAVING clause matches nothing, so the correlated subquery always returns NULL.",
        "optimised_approach": "Replace nested correlated subqueries with two CTEs: one for category-level spend aggregation, one for max per category. O(n) scan instead of O(n²) correlated execution.",
        "sample_data_description": (
            "E-commerce: customers, orders (delivered/pending/cancelled), order_items, products with categories."
        ),
    },
    {
        "task_id": "hard_002",
        "difficulty": "hard",
        "schema_hint": "hr",
        "task_prompt": (
            "The query below should return engineers (job_title LIKE '%Engineer%') "
            "who earn above the average salary of their own department. "
            "It has a logic bug AND uses a correlated subquery in SELECT that "
            "recomputes the average for every row. "
            "1) Fix the bug. "
            "2) Move the per-department average into a CTE or subquery so it is "
            "computed only once per department. "
            "Return only the corrected + optimised SQL."
        ),
        "buggy_query": (
            "SELECT e.name, e.job_title, s.amount AS salary,\n"
            "       (SELECT AVG(s2.amount)\n"
            "        FROM salaries s2\n"
            "        JOIN employees e2 ON s2.emp_id = e2.emp_id\n"
            "        WHERE e2.dept_id = e.dept_id) AS dept_avg\n"
            "FROM employees e\n"
            "JOIN salaries s ON s.emp_id = e.emp_id\n"
            "WHERE e.job_title LIKE '%Engineer%'\n"
            "  AND s.amount > (SELECT AVG(s3.amount)\n"
            "                  FROM salaries s3\n"
            "                  JOIN employees e3 ON s3.emp_id = e3.emp_id\n"
            "                  WHERE e3.dept_id = e.dept_id\n"
            "                    AND s3.to_date IS NOT NULL);\n"  # BUG: filters to inactive salaries
        ),
        "correct_query": (
            "WITH dept_avg AS (\n"
            "  SELECT e.dept_id, AVG(s.amount) AS avg_salary\n"
            "  FROM employees e\n"
            "  JOIN salaries s ON s.emp_id = e.emp_id\n"
            "  WHERE s.to_date IS NULL\n"
            "  GROUP BY e.dept_id\n"
            ")\n"
            "SELECT e.name, e.job_title, s.amount AS salary, da.avg_salary AS dept_avg\n"
            "FROM employees e\n"
            "JOIN salaries s  ON s.emp_id  = e.emp_id\n"
            "JOIN dept_avg da ON da.dept_id = e.dept_id\n"
            "WHERE e.job_title LIKE '%Engineer%'\n"
            "  AND s.to_date IS NULL\n"
            "  AND s.amount > da.avg_salary\n"
            "ORDER BY s.amount DESC;"
        ),
        "order_sensitive": True,
        "bug_description": "Correlated subquery in WHERE filters s3.to_date IS NOT NULL — selecting inactive (historical) salaries instead of current ones (to_date IS NULL). The current salary records all have to_date IS NULL.",
        "optimised_approach": "Compute per-department average once in a CTE (not once per row). Eliminates O(n) correlated subquery executions.",
        "sample_data_description": (
            "HR: employees, departments, salaries (to_date IS NULL = current salary). "
            "Engineers are in Engineering dept."
        ),
    },
    {
        "task_id": "hard_003",
        "difficulty": "hard",
        "schema_hint": "analytics",
        "task_prompt": (
            "The query below should return each user's username, their total session "
            "count, total page_view count, and their most-visited page — all in one row. "
            "It has a logic bug AND is inefficient (scans page_views three times). "
            "1) Fix the bug. "
            "2) Rewrite to scan page_views only once using CTEs or window functions. "
            "Return only the corrected + optimised SQL."
        ),
        "buggy_query": (
            "SELECT u.username,\n"
            "       COUNT(DISTINCT s.session_id) AS session_count,\n"
            "       (SELECT COUNT(*) FROM page_views pv\n"
            "        JOIN sessions s2 ON pv.session_id = s2.session_id\n"
            "        WHERE s2.user_id = u.user_id) AS total_views,\n"
            "       (SELECT pv2.page\n"
            "        FROM page_views pv2\n"
            "        JOIN sessions s3 ON pv2.session_id = s3.session_id\n"
            "        WHERE s3.user_id = u.user_id\n"
            "        GROUP BY pv2.page\n"
            "        ORDER BY COUNT(*) DESC\n"
            "        LIMIT 1) AS top_page\n"
            "FROM users u\n"
            "LEFT JOIN sessions s ON u.user_id = s.user_id\n"
            "GROUP BY u.user_id, u.username\n"
            "HAVING session_count > 0;"  # BUG: HAVING excludes users with 0 sessions
                                          # but task says "each user" — should use no HAVING
        ),
        "correct_query": (
            "WITH pv_stats AS (\n"
            "  SELECT s.user_id,\n"
            "         COUNT(*)  AS total_views,\n"
            "         COUNT(DISTINCT pv.session_id) AS view_sessions\n"
            "  FROM page_views pv\n"
            "  JOIN sessions s ON pv.session_id = s.session_id\n"
            "  GROUP BY s.user_id\n"
            "),\n"
            "top_pages AS (\n"
            "  SELECT s.user_id, pv.page,\n"
            "         ROW_NUMBER() OVER (PARTITION BY s.user_id ORDER BY COUNT(*) DESC) AS rn\n"
            "  FROM page_views pv\n"
            "  JOIN sessions s ON pv.session_id = s.session_id\n"
            "  GROUP BY s.user_id, pv.page\n"
            ")\n"
            "SELECT u.username,\n"
            "       COUNT(DISTINCT s.session_id) AS session_count,\n"
            "       COALESCE(pvs.total_views, 0) AS total_views,\n"
            "       tp.page AS top_page\n"
            "FROM users u\n"
            "LEFT JOIN sessions s   ON u.user_id = s.user_id\n"
            "LEFT JOIN pv_stats pvs ON u.user_id = pvs.user_id\n"
            "LEFT JOIN top_pages tp ON u.user_id = tp.user_id AND tp.rn = 1\n"
            "GROUP BY u.user_id, u.username, pvs.total_views, tp.page\n"
            "ORDER BY u.user_id;"
        ),
        "order_sensitive": True,
        "bug_description": "HAVING session_count > 0 excludes users with no sessions — but the task requires ALL users. Also page_views table is scanned twice in correlated subqueries.",
        "optimised_approach": "Pre-aggregate page_views once in CTEs (pv_stats and top_pages with ROW_NUMBER window function), then LEFT JOIN the results. Single pass over page_views instead of O(n) correlated scans.",
        "sample_data_description": (
            "Analytics: users, sessions, page_views(view_id, session_id, page, duration_sec, viewed_at). "
            "Users may have 0 or more sessions."
        ),
    },
]

ALL_TASKS: List[Task] = EASY_TASKS + MEDIUM_TASKS + HARD_TASKS


def get_task_by_id(task_id: str) -> Task | None:
    for t in ALL_TASKS:
        if t["task_id"] == task_id:
            return t
    return None
