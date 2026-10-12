import os

import requests
from dotenv import load_dotenv

import utils

load_dotenv(utils.PROJECT_ROOT / ".env")


def nanoseconds_to_ms(value: int | None) -> float:
    return round((value or 0) / 1_000_000, 3)


def generate_answer(messages: list[dict], model: str | None = None, json_mode: bool = False) -> dict:
    selected_model = model or os.getenv("OLLAMA_MODEL")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    if not selected_model:
        raise ValueError("OLLAMA_MODEL is not configured.")

    payload = {
        "model": selected_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "num_predict": 400,
        },
    }

    if json_mode:
        payload["format"] = "json"

    response = requests.post(f"{base_url}/api/chat", json=payload, timeout=300)

    if not response.ok:
        detail = response.text.strip()
        raise RuntimeError(f"Ollama request failed ({response.status_code}): {detail or response.reason}")

    response_data = response.json()
    answer = response_data.get("message", {}).get("content", "").strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")

    return {
        "answer": answer,
        "model": response_data.get("model", selected_model),
        "response_id": None,
        "total_ms": nanoseconds_to_ms(response_data.get("total_duration")),
        "model_load_ms": nanoseconds_to_ms(response_data.get("load_duration")),
        "prompt_ms": nanoseconds_to_ms(response_data.get("prompt_eval_duration")),
        "inference_ms": nanoseconds_to_ms(response_data.get("eval_duration")),
        "input_tokens": response_data.get("prompt_eval_count", 0),
        "output_tokens": response_data.get("eval_count", 0),
    }
