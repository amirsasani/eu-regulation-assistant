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

def extract_content_blocks__p(article):
    blocks = []
    
    for element in article.find_all("p"):

        # Avoid duplicate text from elements inside tables
        if element.name == "p":
            if element.find_parent("table"):
                continue

            classes = element.get("class", [])

            # Skip article number/title
            if "oj-ti-art" in classes or "oj-sti-art" in classes:
                continue

            text = clean_text(element.get_text(" ", strip=True))

            if text:
                blocks.append(text)
    
        return blocks

def extract_content_blocks__table(article):
    blocks = []

    for element in article.find_all("table"):
        # Avoid nested table duplication
        if element.find_parent("table"):
            continue

        rows = []

        for row in element.find_all("tr"):
            cells = [
                clean_text(cell.get_text(" ", strip=True))
                for cell in row.find_all(["td", "th"], recursive=False)
            ]

            cells = [cell for cell in cells if cell]

            if cells:
                rows.append(" ".join(cells))

        if rows:
            blocks.append(" ".join(rows))

    return blocks

def extract_content_blocks(article):
    blocks = []
    blocks.extend(extract_content_blocks__p(article))
    blocks.extend(extract_content_blocks__table(article))
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


def parse_articles(source_path: Path):
    html = source_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    records = []

    article_pattern = re.compile(
        r"Article\s+(\d+[A-Za-z]?)",
        re.IGNORECASE
    )

    article_headers = soup.find_all("p", class_="oj-ti-art")

    print(f"Found {len(article_headers)} article headers")

    for article_header in article_headers:
        header_text = clean_text(
            article_header.get_text(" ", strip=True)
        )

        match = article_pattern.search(header_text)

        if not match:
            continue

        article_number = match.group(1)

        # More tolerant than find_next_sibling()
        title_element = article_header.find_next(
            "p",
            class_="oj-sti-art"
        )

        if title_element is None:
            continue

        article_title = clean_text(
            title_element.get_text(" ", strip=True)
        )

        blocks = []

        # Walk forward until the next article header
        for element in title_element.find_all_next(["p", "tr"]):

            # Stop when next Article starts
            if (
                element.name == "p"
                and "oj-ti-art" in element.get("class", [])
            ):
                break

            if element.name == "p":
                classes = element.get("class", [])

                if "oj-sti-art" in classes:
                    continue

                text = clean_text(
                    element.get_text(" ", strip=True)
                )

                if text:
                    blocks.append(text)

            elif element.name == "tr":
                cells = [
                    clean_text(td.get_text(" ", strip=True))
                    for td in element.find_all(["td", "th"])
                ]

                text = clean_text(" ".join(
                    cell for cell in cells if cell
                ))

                if text:
                    blocks.append(text)

        paragraphs = split_paragraphs(blocks)

        for paragraph in paragraphs:
            paragraph_number = paragraph["number"]

            records.append({
                "document_id": DOCUMENT_ID,
                "document_title": DOCUMENT_TITLE,
                "language": LANGUAGE,
                "section_type": "article",
                "article_number": article_number,
                "article_title": article_title,
                "paragraph_number": paragraph_number,
                "text": paragraph["text"],
                "source_url": SOURCE_URL,
                "chunk_id": f"ai-act-en-article-{article_number}-paragraph-{paragraph_number}",
            })

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