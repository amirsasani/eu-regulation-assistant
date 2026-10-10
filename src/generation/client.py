import os

import requests


OPENROUTER_URL = ("https://openrouter.ai/api/v1/chat/completions")


def generate_answer(messages: list[dict], model: str | None = None) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY")
    selected_model = (model or os.getenv("OPENROUTER_MODEL"))

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured.")

    if not selected_model:
        raise ValueError("OPENROUTER_MODEL is not configured.")

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-OpenRouter-Title": ("EU Regulation Assistant"),
        },
        json={"model": selected_model, "messages": messages, "max_tokens": 600},
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(f"OpenRouter request failed: {response.status_code} {response.text}")

    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    if not answer:
        raise RuntimeError("The language model returned an empty answer.")

    return {
        "answer": answer.strip(),
        "model": data.get("model", selected_model),
        "response_id": data.get("id"),
    }