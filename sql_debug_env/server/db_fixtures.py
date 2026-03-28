"""
Database fixtures for the SQL Debug RL Environment.

Each fixture is a self-contained SQLite database with schema + seed data.
All fixtures are created fresh per-task using in-memory SQLite connections
so there are zero side-effects between episodes.
"""

from __future__ import annotations
import sqlite3
from typing import Any, Dict, List, Tuple


def make_ecommerce_db() -> sqlite3.Connection:
    """
    E-commerce schema: customers, orders, order_items, products.
    Used by: easy_001, medium_001, hard_001
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE customers (
            customer_id   INTEGER PRIMARY KEY,
            name          TEXT    NOT NULL,
            email         TEXT    NOT NULL,
            country       TEXT    NOT NULL,
            signup_date   TEXT    NOT NULL
        );
        CREATE TABLE products (
            product_id    INTEGER PRIMARY KEY,
            name          TEXT    NOT NULL,
            category      TEXT    NOT NULL,
            price         REAL    NOT NULL
        );
        CREATE TABLE orders (
            order_id      INTEGER PRIMARY KEY,
            customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
            order_date    TEXT    NOT NULL,
            status        TEXT    NOT NULL
        );
        CREATE TABLE order_items (
            item_id       INTEGER PRIMARY KEY,
            order_id      INTEGER NOT NULL REFERENCES orders(order_id),
            product_id    INTEGER NOT NULL REFERENCES products(product_id),
            quantity      INTEGER NOT NULL,
            unit_price    REAL    NOT NULL
        );

        INSERT INTO customers VALUES
          (1,'Alice','alice@ex.com','IN','2022-01-10'),
          (2,'Bob','bob@ex.com','US','2022-03-15'),
          (3,'Carol','carol@ex.com','IN','2023-06-01'),
          (4,'Dave','dave@ex.com','UK','2021-11-20'),
          (5,'Eve','eve@ex.com','IN','2023-09-05');

        INSERT INTO products VALUES
          (1,'Laptop','Electronics',75000),
          (2,'Phone','Electronics',30000),
          (3,'Desk','Furniture',12000),
          (4,'Chair','Furniture',8000),
          (5,'Headphones','Electronics',4500);

        INSERT INTO orders VALUES
          (101,1,'2024-01-05','delivered'),
          (102,2,'2024-01-10','delivered'),
          (103,1,'2024-02-14','delivered'),
          (104,3,'2024-03-01','pending'),
          (105,4,'2024-03-10','delivered'),
          (106,5,'2024-04-01','cancelled'),
          (107,2,'2024-04-15','delivered');

        INSERT INTO order_items VALUES
          (1,101,1,1,75000),
          (2,101,5,2,4500),
          (3,102,2,1,30000),
          (4,103,3,1,12000),
          (5,103,4,2,8000),
          (6,104,2,1,30000),
          (7,105,1,1,75000),
          (8,106,5,1,4500),
          (9,107,2,2,30000),
          (10,107,5,1,4500);
    """)
    conn.commit()
    return conn


def make_hr_db() -> sqlite3.Connection:
    """
    HR schema: employees, departments, salaries, projects.
    Used by: easy_002, medium_002, hard_002
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE departments (
            dept_id    INTEGER PRIMARY KEY,
            dept_name  TEXT    NOT NULL,
            location   TEXT    NOT NULL
        );
        CREATE TABLE employees (
            emp_id     INTEGER PRIMARY KEY,
            name       TEXT    NOT NULL,
            dept_id    INTEGER NOT NULL REFERENCES departments(dept_id),
            hire_date  TEXT    NOT NULL,
            job_title  TEXT    NOT NULL
        );
        CREATE TABLE salaries (
            emp_id     INTEGER NOT NULL REFERENCES employees(emp_id),
            amount     REAL    NOT NULL,
            from_date  TEXT    NOT NULL,
            to_date    TEXT
        );
        CREATE TABLE projects (
            project_id INTEGER PRIMARY KEY,
            name       TEXT    NOT NULL,
            dept_id    INTEGER NOT NULL REFERENCES departments(dept_id),
            budget     REAL    NOT NULL,
            start_date TEXT    NOT NULL
        );
        CREATE TABLE assignments (
            emp_id     INTEGER NOT NULL REFERENCES employees(emp_id),
            project_id INTEGER NOT NULL REFERENCES projects(project_id),
            role       TEXT    NOT NULL,
            hours      INTEGER NOT NULL
        );

        INSERT INTO departments VALUES
          (1,'Engineering','Bangalore'),
          (2,'Marketing','Mumbai'),
          (3,'Finance','Delhi'),
          (4,'HR','Bangalore');

        INSERT INTO employees VALUES
          (1,'Arjun',1,'2020-06-01','Senior Engineer'),
          (2,'Priya',1,'2021-03-15','Engineer'),
          (3,'Rahul',2,'2019-11-01','Marketing Lead'),
          (4,'Sneha',3,'2022-01-10','Financial Analyst'),
          (5,'Vikram',1,'2018-07-20','Staff Engineer'),
          (6,'Anita',4,'2023-02-01','HR Manager'),
          (7,'Kiran',2,'2021-08-10','Marketing Analyst');

        INSERT INTO salaries VALUES
          (1,120000,'2020-06-01',NULL),
          (2,80000,'2021-03-15',NULL),
          (3,95000,'2019-11-01',NULL),
          (4,70000,'2022-01-10',NULL),
          (5,150000,'2018-07-20',NULL),
          (6,85000,'2023-02-01',NULL),
          (7,72000,'2021-08-10',NULL);

        INSERT INTO projects VALUES
          (1,'Apollo',1,500000,'2023-01-01'),
          (2,'Hermes',2,200000,'2023-06-01'),
          (3,'Zeus',1,800000,'2024-01-01'),
          (4,'Athena',3,150000,'2023-09-01');

        INSERT INTO assignments VALUES
          (1,1,'Lead',200),(2,1,'Developer',300),(5,1,'Architect',150),
          (1,3,'Lead',100),(2,3,'Developer',200),(5,3,'Reviewer',80),
          (3,2,'Lead',250),(7,2,'Analyst',180),
          (4,4,'Analyst',120);
    """)
    conn.commit()
    return conn


def make_analytics_db() -> sqlite3.Connection:
    """
    Analytics schema: events, sessions, users, page_views.
    Used by: easy_003, medium_003, hard_003
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE users (
            user_id        INTEGER PRIMARY KEY,
            username       TEXT    NOT NULL,
            plan           TEXT    NOT NULL,
            created_at     TEXT    NOT NULL
        );
        CREATE TABLE sessions (
            session_id     INTEGER PRIMARY KEY,
            user_id        INTEGER NOT NULL REFERENCES users(user_id),
            started_at     TEXT    NOT NULL,
            ended_at       TEXT,
            device         TEXT    NOT NULL
        );
        CREATE TABLE events (
            event_id       INTEGER PRIMARY KEY,
            session_id     INTEGER NOT NULL REFERENCES sessions(session_id),
            event_type     TEXT    NOT NULL,
            occurred_at    TEXT    NOT NULL,
            properties     TEXT
        );
        CREATE TABLE page_views (
            view_id        INTEGER PRIMARY KEY,
            session_id     INTEGER NOT NULL REFERENCES sessions(session_id),
            page           TEXT    NOT NULL,
            duration_sec   INTEGER NOT NULL,
            viewed_at      TEXT    NOT NULL
        );

        INSERT INTO users VALUES
          (1,'alice_p','pro','2023-01-01'),
          (2,'bob_q','free','2023-03-10'),
          (3,'carol_r','pro','2022-11-05'),
          (4,'dave_s','free','2024-01-20'),
          (5,'eve_t','enterprise','2023-07-15');

        INSERT INTO sessions VALUES
          (1,1,'2024-03-01 09:00','2024-03-01 09:45','desktop'),
          (2,2,'2024-03-01 10:00','2024-03-01 10:20','mobile'),
          (3,1,'2024-03-02 14:00','2024-03-02 14:30','desktop'),
          (4,3,'2024-03-02 11:00','2024-03-02 12:00','desktop'),
          (5,4,'2024-03-03 08:00','2024-03-03 08:10','mobile'),
          (6,5,'2024-03-03 15:00','2024-03-03 16:30','desktop'),
          (7,2,'2024-03-04 09:30','2024-03-04 09:50','tablet'),
          (8,3,'2024-03-04 13:00','2024-03-04 14:00','desktop');

        INSERT INTO events VALUES
          (1,1,'signup','2024-03-01 09:05',NULL),
          (2,1,'purchase','2024-03-01 09:30','{"amount":999}'),
          (3,2,'pageview','2024-03-01 10:05',NULL),
          (4,3,'purchase','2024-03-02 14:15','{"amount":499}'),
          (5,4,'signup','2024-03-02 11:10',NULL),
          (6,6,'purchase','2024-03-03 15:30','{"amount":1999}'),
          (7,7,'pageview','2024-03-04 09:35',NULL),
          (8,8,'purchase','2024-03-04 13:30','{"amount":499}');

        INSERT INTO page_views VALUES
          (1,1,'/dashboard',120,'2024-03-01 09:10'),
          (2,1,'/settings',45,'2024-03-01 09:25'),
          (3,2,'/home',30,'2024-03-01 10:02'),
          (4,3,'/dashboard',200,'2024-03-02 14:05'),
          (5,4,'/pricing',60,'2024-03-02 11:15'),
          (6,5,'/home',15,'2024-03-03 08:02'),
          (7,6,'/dashboard',300,'2024-03-03 15:10'),
          (8,6,'/analytics',250,'2024-03-03 15:45'),
          (9,7,'/home',20,'2024-03-04 09:32'),
          (10,8,'/dashboard',180,'2024-03-04 13:10');
    """)
    conn.commit()
    return conn


# Registry: task_id → factory function
DB_FACTORIES = {
    "easy_001":   make_ecommerce_db,
    "easy_002":   make_hr_db,
    "easy_003":   make_analytics_db,
    "medium_001": make_ecommerce_db,
    "medium_002": make_hr_db,
    "medium_003": make_analytics_db,
    "hard_001":   make_ecommerce_db,
    "hard_002":   make_hr_db,
    "hard_003":   make_analytics_db,
}


def get_db(task_id: str) -> sqlite3.Connection:
    factory = DB_FACTORIES.get(task_id)
    if factory is None:
        raise ValueError(f"Unknown task_id: {task_id}")
    return factory()


def run_query(conn: sqlite3.Connection, sql: str) -> Tuple[List[str], List[Tuple]]:
    """
    Execute a SQL query and return (column_names, rows).
    Raises on syntax/execution error.
    """
    cursor = conn.cursor()
    cursor.execute(sql)
    cols = [d[0] for d in cursor.description] if cursor.description else []
    rows = cursor.fetchall()
    return cols, [tuple(r) for r in rows]


def results_match(
    got_cols: List[str],
    got_rows: List[Tuple],
    expected_cols: List[str],
    expected_rows: List[Tuple],
    order_sensitive: bool = False,
) -> Tuple[bool, str]:
    """
    Compare two result sets.
    Returns (match: bool, reason: str).
    """
    # Normalise column names to lowercase for comparison
    gc = [c.lower() for c in got_cols]
    ec = [c.lower() for c in expected_cols]

    if set(gc) != set(ec):
        return False, f"Column mismatch: got {gc}, expected {ec}"

    # Re-order got_rows columns to match expected column order if needed
    if gc != ec:
        idx = [gc.index(c) for c in ec]
        got_rows = [tuple(r[i] for i in idx) for r in got_rows]

    # Normalise numeric types for comparison (int vs float)
    def norm_row(row):
        return tuple(
            int(v) if isinstance(v, float) and v == int(v) else v
            for v in row
        )

    got_norm = [norm_row(r) for r in got_rows]
    exp_norm = [norm_row(r) for r in expected_rows]

    if not order_sensitive:
        got_set = sorted(str(r) for r in got_norm)
        exp_set = sorted(str(r) for r in exp_norm)
        if got_set == exp_set:
            return True, "Results match."
        return False, (
            f"Row mismatch (order-insensitive).\n"
            f"  Expected {len(exp_norm)} rows: {exp_norm[:3]}{'...' if len(exp_norm)>3 else ''}\n"
            f"  Got      {len(got_norm)} rows: {got_norm[:3]}{'...' if len(got_norm)>3 else ''}"
        )
    else:
        if got_norm == exp_norm:
            return True, "Results match (order-sensitive)."
        return False, (
            f"Row mismatch (order-sensitive).\n"
            f"  Expected: {exp_norm[:3]}\n"
            f"  Got:      {got_norm[:3]}"
        )
