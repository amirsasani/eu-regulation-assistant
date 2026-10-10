import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

DATA_PATH = PROJECT_ROOT / "data"

DATA_EVAL_PATH = DATA_PATH / "evaluation"
DATA_INDEXES_PATH = DATA_PATH / "indexes"
DATA_PROCESSED_PATH = DATA_PATH / "processed"
DATA_RAW_PATH = DATA_PATH / "raw"

SCRIPTS_PATH = PROJECT_ROOT / "scripts"

SRC_PATH = PROJECT_ROOT / "src"

SRC_INJEST_PATH = SRC_PATH / "injestion"
SRC_RETRIEVAL_PATH = SRC_PATH / "retrieval"


DEFAULT_SEMANTIC_MODEL_NAME = "intfloat/multilingual-e5-base"

DEFAULT_OPENROUTER_MODEL = "openrouter/free"

def load_json_file(file_path: Path, permission: str = "r", encoding: str = "utf-8"):
    with open(file_path, permission, encoding=encoding) as f:
        return json.load(f)