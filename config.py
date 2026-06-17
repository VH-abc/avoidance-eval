from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
PAIRS_PATH = DATA_DIR / "pairs.json"

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1"))
GRADER_PROVIDER = os.getenv("GRADER_PROVIDER", DEFAULT_PROVIDER)
GRADER_MODEL = os.getenv("GRADER_MODEL", DEFAULT_MODEL)
GRADER_TEMPERATURE = float(os.getenv("GRADER_TEMPERATURE", "0"))

TOKEN_BUDGETS = [16, 32, 64, 128, 256, 512, 1024]
FAST_TOKEN_BUDGETS = [32, 64, 128, 256]
DEFAULT_TRIALS = 5
DEFAULT_Y_TRIALS = 10
DEFAULT_WORKERS = 16
THREAD_MONITOR_THRESHOLD = 6
MAX_THREAD_ITERATIONS = 12
MAX_TWO_AGENT_RETRIES = 4
