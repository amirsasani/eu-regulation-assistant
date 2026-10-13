import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_FILE = PROJECT_ROOT / "data/processed/ai-act-articles.json"
RETRIEVAL_QUESTIONS_FILE = PROJECT_ROOT / "data/evaluation/questions.json"
ITALIAN_RETRIEVAL_QUESTIONS_FILE = (
    PROJECT_ROOT / "data/evaluation/questions-it.json"
)
GENERATION_QUESTIONS_FILE = (PROJECT_ROOT / "data/evaluation/generation_questions.json")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_retrieval_labels_reference_indexed_articles():
    documents = load_json(DOCUMENTS_FILE)
    questions = load_json(RETRIEVAL_QUESTIONS_FILE)
    indexed_articles = {str(document["article_number"]) for document in documents}
    question_ids = [question["id"] for question in questions]

    assert len(questions) >= 40
    assert len(question_ids) == len(set(question_ids))
    assert sum(not question["relevant_articles"] for question in questions) >= 4

    for question in questions:
        assert question["question"].strip()
        assert set(question["relevant_articles"]) <= indexed_articles


def test_italian_retrieval_questions_match_english_labels():
    english_questions = load_json(RETRIEVAL_QUESTIONS_FILE)
    italian_questions = load_json(ITALIAN_RETRIEVAL_QUESTIONS_FILE)

    assert len(italian_questions) == len(english_questions)

    for english, italian in zip(english_questions, italian_questions, strict=True):
        assert italian["id"] == english["id"]
        assert italian["question"].strip()
        assert italian["question"] != english["question"]
        assert italian["relevant_articles"] == english["relevant_articles"]
        assert italian["category"] == english["category"]
        assert italian["difficulty"] == english["difficulty"]


def test_generation_labels_reference_existing_chunks():
    documents = load_json(DOCUMENTS_FILE)
    questions = load_json(GENERATION_QUESTIONS_FILE)
    chunks = {document["chunk_id"]: document for document in documents}
    question_ids = [question["id"] for question in questions]

    assert len(questions) >= 20
    assert len(question_ids) == len(set(question_ids))
    assert sum(not question["answerable"] for question in questions) >= 4

    for question in questions:
        expected_points = question["expected_points"]

        assert question["question"].strip()
        assert question["answerable"] == bool(expected_points)

        if not question["answerable"]:
            assert question["expected_articles"] == []
            continue

        expected_articles = {
            str(chunks[point["chunk_id"]]["article_number"])
            for point in expected_points
        }

        assert all(point["point"].strip() for point in expected_points)
        assert expected_articles == set(question["expected_articles"])
