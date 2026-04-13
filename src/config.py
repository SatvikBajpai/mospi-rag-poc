from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DB_DIR = ROOT / "db"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
COLLECTION = "mospi_press_releases"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
MAX_NEW_TOKENS = 220
