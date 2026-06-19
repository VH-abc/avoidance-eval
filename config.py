from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
PAIRS_PATH = DATA_DIR / "pairs.json"


def resolve_pairs_path(pairs_file: str | Path | None = None) -> Path:
    if pairs_file is None:
        return PAIRS_PATH
    path = Path(pairs_file).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def pairs_file_for_manifest(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1"))
GRADER_PROVIDER = os.getenv("GRADER_PROVIDER", DEFAULT_PROVIDER)
GRADER_MODEL = os.getenv("GRADER_MODEL", DEFAULT_MODEL)
GRADER_TEMPERATURE = float(os.getenv("GRADER_TEMPERATURE", "0"))

# Strong model used to generate and verify candidate (X, Y) pairs.
GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "anthropic")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "claude-opus-4-8")
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "1"))
# Solver/target model whose Y accuracy defines "hard"; reuses the cheap default.
SOLVER_PROVIDER = os.getenv("SOLVER_PROVIDER", DEFAULT_PROVIDER)
SOLVER_MODEL = os.getenv("SOLVER_MODEL", DEFAULT_MODEL)
GENERATION_MAX_SPEND = float(os.getenv("GENERATION_MAX_SPEND", "30"))

# Per-request HTTP timeout (seconds) and retry count for LLM clients. Prevents
# individual requests from hanging for many minutes on a stalled connection.
REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))

TOKEN_BUDGETS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
FAST_TOKEN_BUDGETS = [32, 64, 128, 256, 512, 1024]
DEFAULT_TRIALS = 5
DEFAULT_Y_TRIALS = 10
DEFAULT_WORKERS = 0
DEFAULT_JOB_WORKERS = 0
FAST_SCAFFOLDS = ["control_answer_only", "control", "baseline_avoid", "cheating_avoid"]


def resolve_pool_workers(requested: int, job_count: int) -> int:
    if job_count <= 0:
        return 1
    if requested <= 0:
        return job_count
    return min(requested, job_count)
Y_BUDGET_WARNING_FRACTIONS = (0.5, 0.9)
Y_BUDGET_MIN_FOR_WARNINGS = 16
Y_BUDGET_MIN_FINAL_SEGMENT = 16
Y_BUDGET_FINAL_FRACTION = 0.2
THREAD_MONITOR_THRESHOLD = 6
MAX_THREAD_ITERATIONS = 12
MAX_TWO_AGENT_RETRIES = 4
