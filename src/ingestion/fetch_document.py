import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
import json
from pathlib import Path
from hashlib import sha256
import utils

SOURCE_URL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
    "?uri=OJ:L_202401689"
)

SOURCE_PATH = utils.DATA_RAW_PATH / "ai-act-raw.html"
METADATA_PATH = utils.DATA_RAW_PATH / "ai-act-metadata.json"

def download_source_file():
    SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(SOURCE_URL, timeout=(10, 60))
    response.raise_for_status()

    content = response.content

    if not content:
        raise RuntimeError("Downloaded file is empty.")

    SOURCE_PATH.write_bytes(content)

    print(f"Downloaded {len(content):,} bytes to {SOURCE_PATH}")

    return SOURCE_PATH, response

def calculate_sha256(file_path: Path) -> str:
    hash_sha256 = sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def generate_metadata(source_path: Path, response):
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_url": SOURCE_URL,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "path": str(source_path.relative_to(utils.PROJECT_ROOT)),
        "size_bytes": source_path.stat().st_size,
        "sha256": calculate_sha256(source_path),
        "encoding": "utf-8",
        "content_type": response.headers.get("Content-Type"),
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Metadata written to {METADATA_PATH}")


if __name__ == "__main__":
    source_path, response = download_source_file()
    generate_metadata(source_path, response)