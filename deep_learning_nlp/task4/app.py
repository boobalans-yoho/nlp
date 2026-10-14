"""FastAPI service for the fine-tuned customer feedback classifier."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_raw_model_dir = os.getenv(
    "MODEL_DIR",
    str(PROJECT_ROOT / "models" / "distilbert-feedback"),
)
_model_dir = Path(_raw_model_dir).expanduser()
MODEL_DIR = (
    (_model_dir if _model_dir.is_absolute() else PROJECT_ROOT / _model_dir)
    .resolve()
)
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

tokenizer: Any | None = None
model: Any | None = None
labels: list[str] = []
model_error: str | None = None


class FeedbackRequest(BaseModel):
    """Validated request body for a sentiment prediction."""

    text: str = Field(min_length=1, max_length=5000)


def _load_label_mapping(labels_path: Path) -> list[str]:
    """Read and validate the saved class names used by the model."""
    if not labels_path.is_file():
        raise RuntimeError(
            f"Label mapping not found: {labels_path}. "
            "Retrain the model so labels.txt is saved with it."
        )

    loaded_labels = [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(loaded_labels) < 2 or len(set(loaded_labels)) != len(loaded_labels):
        raise RuntimeError(f"Invalid class mapping in {labels_path}.")
    return loaded_labels


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the fine-tuned model once when the API starts."""
    global tokenizer, model, labels, model_error

    tokenizer = None
    model = None
    labels = []
    model_error = None

    try:
        labels_path = MODEL_DIR / "labels.txt"
        if not MODEL_DIR.is_dir():
            raise FileNotFoundError(
                f"Fine-tuned model directory not found: {MODEL_DIR}. "
                "Run task4/train_transformer.py first."
            )

        loaded_labels = _load_label_mapping(labels_path)
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR), local_files_only=True
        )
        loaded_model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_DIR), local_files_only=True
        ).to(DEVICE)
        if loaded_model.config.num_labels != len(loaded_labels):
            raise RuntimeError(
                "Model class count does not match labels.txt "
                f"({loaded_model.config.num_labels} model outputs, "
                f"{len(loaded_labels)} labels). Retrain the model."
            )

        loaded_model.eval()
        tokenizer = loaded_tokenizer
        model = loaded_model
        labels = loaded_labels
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        model_error = str(error)
        logger.error("Classifier model is unavailable: %s", model_error)

    try:
        yield
    finally:
        tokenizer = None
        model = None
        labels = []
        model_error = None


app = FastAPI(
    title="Customer Feedback Classifier",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home() -> dict[str, str]:
    """Return a simple service status message."""
    return {"message": "Customer feedback classifier is running"}


@app.get("/health")
def health() -> dict[str, str]:
    """Report whether the model is ready to serve predictions."""
    if model is None or tokenizer is None:
        return {
            "status": "not_ready",
            "detail": model_error or "Classifier model is not loaded.",
        }
    return {"status": "ok"}


@app.post("/predict")
def predict(request: FeedbackRequest) -> dict[str, str | float]:
    """Predict a sentiment label and its confidence for one text."""
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="Text must not be empty or whitespace.",
        )
    if model is None or tokenizer is None or not labels:
        raise HTTPException(
            status_code=503,
            detail=(
                model_error
                or "Classifier is not ready; ensure the model is loaded."
            ),
        )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    if predicted_id >= len(labels):
        raise RuntimeError(
            "Model predicted a class outside the saved label mapping."
        )

    return {
        "text": text,
        "label": labels[predicted_id],
        "confidence": round(float(probabilities[predicted_id].item()), 4),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
