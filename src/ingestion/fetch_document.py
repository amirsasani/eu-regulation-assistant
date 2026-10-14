import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import requests

import utils


DOCUMENT_ID = "32024R1689"
DOWNLOAD_URL = f"https://publications.europa.eu/resource/celex/{DOCUMENT_ID}"
DEFAULT_LANGUAGE = "en"

SOURCES = {
    "en": {
        "canonical_url": (
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
            f"?uri=CELEX:{DOCUMENT_ID}"
        ),
        "accept_language": "en-GB,en;q=0.9",
        "source_path": utils.DATA_RAW_PATH / "ai-act-raw.html",
        "metadata_path": utils.DATA_RAW_PATH / "ai-act-metadata.json",
    },
    "it": {
        "canonical_url": (
            "https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/"
            f"?uri=CELEX:{DOCUMENT_ID}"
        ),
        "accept_language": "it-IT,it;q=0.9",
        "source_path": utils.DATA_RAW_PATH / "ai-act-it-raw.html",
        "metadata_path": utils.DATA_RAW_PATH / "ai-act-it-metadata.json",
    },
}


def calculate_sha256(file_path: Path) -> str:
    digest = sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_source_file(
    language: str = DEFAULT_LANGUAGE,
    source_path: Path | None = None,
):
    if language not in SOURCES:
        raise ValueError(f"Unsupported language: {language}")

    source = SOURCES[language]
    destination = source_path or source["source_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        DOWNLOAD_URL,
        headers={
            "Accept": "application/xhtml+xml,text/html;q=0.9,*/*;q=0.1",
            "Accept-Language": source["accept_language"],
            "User-Agent": "eu-regulation-assistant/0.1",
        },
        timeout=(10, 60),
    )
    response.raise_for_status()

    content = response.content

    if not content:
        raise RuntimeError("Downloaded file is empty.")

    if b"oj-ti-art" not in content:
        preview = response.text[:200].replace("\n", " ")
        raise RuntimeError(
            "Downloaded file does not contain AI Act article markup. "
            f"Response preview: {preview!r}"
        )

    destination.write_bytes(content)

    print(f"Downloaded {len(content):,} bytes to {destination}")

    return destination, response


def generate_metadata(
    source_path: Path,
    response,
    language: str = DEFAULT_LANGUAGE,
    metadata_path: Path | None = None,
):
    if language not in SOURCES:
        raise ValueError(f"Unsupported language: {language}")

    source = SOURCES[language]
    destination = metadata_path or source["metadata_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "document_id": DOCUMENT_ID,
        "language": language,
        "source_url": source["canonical_url"],
        "download_url": DOWNLOAD_URL,
        "resolved_download_url": response.url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "path": str(source_path.relative_to(utils.PROJECT_ROOT)),
        "size_bytes": source_path.stat().st_size,
        "sha256": calculate_sha256(source_path),
        "encoding": response.encoding or "utf-8",
        "content_type": response.headers.get("Content-Type"),
    }

    destination.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Metadata written to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an official AI Act source.")
    parser.add_argument("--language", choices=sorted(SOURCES), default=DEFAULT_LANGUAGE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    args = parser.parse_args()

    source_path, response = download_source_file(
        language=args.language,
        source_path=args.output,
    )
    generate_metadata(
        source_path=source_path,
        response=response,
        language=args.language,
        metadata_path=args.metadata_output,
    )


if __name__ == "__main__":
    main()
