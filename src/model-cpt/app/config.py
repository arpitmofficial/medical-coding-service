import os
import logging
import logging.handlers
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HEADERS = {
    "api-key": QDRANT_API_KEY,
    "Content-Type": "application/json",
}

# Keep backward-compatible alias
HEADERS = QDRANT_HEADERS

# --- LLM (OpenAI-compatible) ---
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Pipeline thresholds (adjustable without touching logic) ---
QDRANT_TOP_K: int = int(os.getenv("QDRANT_TOP_K", "50"))  # candidates per entity
FINAL_TOP_N: int = int(
    os.getenv("FINAL_TOP_N", "5")
)  # codes returned to caller (strict top 5)
MIN_SCORE: float = float(os.getenv("MIN_SCORE", "0.0"))  # no score cutoff (was 0.80)


# --- Logging Configuration ---
def setup_logging():
    """Configure clean logging: detailed file logs + minimal console output."""

    # Create a shared logs directory inside src/ for all model pipelines.
    src_dir = Path(__file__).resolve().parents[2]
    log_dir = src_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # === FILE HANDLER: Detailed logs ===
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "medical_coding.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # === CONSOLE HANDLER: Minimal output ===
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings/errors
    console_formatter = logging.Formatter("%(levelname)s | %(message)s")
    console_handler.setFormatter(console_formatter)

    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # === SILENCE NOISY LIBRARIES ===
    # Reduce httpx verbosity (only errors to console, debug to file)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google.genai").setLevel(logging.WARNING)

    # === CUSTOM CONSOLE LOGGER: Important status messages ===
    console_logger = logging.getLogger("console")
    console_logger.setLevel(logging.INFO)
    console_logger.propagate = False  # Don't send to root logger

    status_handler = logging.StreamHandler()
    status_handler.setLevel(logging.INFO)
    status_formatter = logging.Formatter("%(message)s")
    status_handler.setFormatter(status_formatter)
    console_logger.addHandler(status_handler)

    # Also log status to file
    console_logger.addHandler(file_handler)

    print("Logging configured: detailed logs -> logs/medical_coding.log")
    return console_logger


# Initialize logging
console_logger = setup_logging()
