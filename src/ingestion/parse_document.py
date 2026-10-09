import json
import re
from bs4 import BeautifulSoup
from pathlib import Path
import utils


DOCUMENT_ID = "32024R1689"
DOCUMENT_TITLE = "EU Artificial Intelligence Act"
LANGUAGE = "en"

SOURCE_URL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
    "?uri=OJ:L_202401689"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = utils.DATA_RAW_PATH / "ai-act-raw.html"
OUTPUT_PATH = utils.DATA_PROCESSED_PATH / "ai-act-articles.json"

def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()

def extract_content_blocks(article_container):
    """
    Extract paragraphs and table rows in document order.

    Paragraphs inside tables are skipped because their containing
    table row is processed separately.
    """
    blocks = []

    for element in article_container.find_all(["p", "table"]):
        if element.name == "p":
            # Table text is processed through its rows.
            if element.find_parent("table"):
                continue

            classes = element.get("class", [])

            # Skip article number and article title.
            if ("oj-ti-art" in classes or "oj-sti-art" in classes):
                continue

            text = clean_text(element.get_text(" ", strip=True))

            if text:
                blocks.append(text)

        elif element.name == "table":
            # Process only outermost tables.
            if element.find_parent("table"):
                continue

            for row in element.find_all("tr"):
                cells = [
                    clean_text(cell.get_text(" ", strip=True))
                    for cell in row.find_all(["td", "th"], recursive=False)
                ]

                text = clean_text(" ".join(cell for cell in cells if cell))

                if text:
                    blocks.append(text)

    return blocks

def split_paragraphs(blocks):
    """
    Groups content by legal paragraph number:
        1. ...
        2. ...
        3. ...

    Subpoints such as (a), (b), etc. remain part of
    the numbered paragraph.
    """

    paragraphs = []

    current_number = None
    current_text = []

    paragraph_pattern = re.compile(
        r"^(\d+)\.\s*(.*)$",
        re.DOTALL
    )

    for block in blocks:
        match = paragraph_pattern.match(block)

        if match:
            if current_text:
                paragraphs.append({
                    "number": current_number,
                    "text": clean_text(" ".join(current_text)),
                })

            current_number = match.group(1)
            current_text = [match.group(2)]

        else:
            current_text.append(block)

    if current_text:
        paragraphs.append({
            "number": current_number,
            "text": clean_text(" ".join(current_text)),
        })

    # Some articles have no explicit numbered paragraphs.
    # For dataset consistency, treat the whole article as paragraph 1.
    if (
        len(paragraphs) == 1
        and paragraphs[0]["number"] is None
    ):
        paragraphs[0]["number"] = "1"

    return paragraphs


def validate_records(records: list[dict]) -> None:
    if not records:
        raise ValueError("Parser produced no article records.")

    chunk_ids = set()
    article_numbers = set()

    for position, record in enumerate(records):
        text = record["text"].strip()

        if not text:
            raise ValueError(f"Record {position} has empty text.")

        chunk_id = record["chunk_id"]

        if chunk_id in chunk_ids:
            raise ValueError(f"Duplicate chunk ID: {chunk_id}")

        chunk_ids.add(chunk_id)
        article_numbers.add(record["article_number"])

    expected_articles = {str(number) for number in range(1, 114)}

    missing_articles = (expected_articles - article_numbers)

    if missing_articles:
        raise ValueError(f"Missing articles: {sorted(missing_articles, key=int)}")

    print(f"Validated {len(records)} unique chunks across {len(article_numbers)} articles.")

def parse_articles(source_path: Path):
    html = source_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    records = []

    article_pattern = re.compile(r"Article\s+(\d+[A-Za-z]?)", re.IGNORECASE)

    article_headers = soup.find_all("p", class_="oj-ti-art")

    print(f"Found {len(article_headers)} article headers")

    for article_header in article_headers:
        header_text = clean_text(article_header.get_text(" ", strip=True))

        match = article_pattern.fullmatch(header_text)

        if not match:
            continue

        article_number = match.group(1)

        # Locate the container belonging only to this article.
        article_container = article_header.find_parent(
            "div",
            id=re.compile(rf"^art_{re.escape(article_number)}$", re.IGNORECASE),
        )

        if article_container is None:
            raise ValueError(f"Could not locate container for Article {article_number}.")

        title_element = article_container.find("p", class_="oj-sti-art")

        if title_element is None:
            raise ValueError(f"Article {article_number} has no title.")

        article_title = clean_text(title_element.get_text(" ", strip=True))

        blocks = extract_content_blocks(article_container)

        paragraphs = split_paragraphs(blocks)

        for paragraph in paragraphs:
            paragraph_number = paragraph["number"]
            text = clean_text(paragraph["text"])

            if not text:
                continue

            records.append({
                "document_id": DOCUMENT_ID,
                "document_title": DOCUMENT_TITLE,
                "language": LANGUAGE,
                "section_type": "article",
                "article_number": article_number,
                "article_title": article_title,
                "paragraph_number": paragraph_number,
                "text": text,
                "source_url": SOURCE_URL,
                "chunk_id": (
                    f"ai-act-{LANGUAGE}"
                    f"-article-{article_number}"
                    f"-paragraph-{paragraph_number}"
                ),
            })

    validate_records(records)

    return records


def save_articles(records):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(
            records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Generated {len(records)} chunks")
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    records = parse_articles(SOURCE_PATH)
    save_articles(records)