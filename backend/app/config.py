"""Central place for settings. Values come from the .env file (never hard-code secrets)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- LLM settings -----------------------------------------------------------
# LM Studio exposes an OpenAI-compatible server at http://localhost:1234/v1
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-7b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")  # LM Studio ignores it
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
# Set USE_LLM=false to run with the built-in rule-based planner (no LLM needed)
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"

# --- Upload settings --------------------------------------------------------
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "20"))
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Agent settings ---------------------------------------------------------
MAX_AGENT_STEPS = 3          # how many tools the agent may call per question
MAX_TABLE_ROWS = 20          # rows sent back to the frontend per table
