"""Fine-tune a DistilBERT classifier using a labelled feedback CSV.

Expected CSV columns: ``text`` and ``label``.

Run from any directory with:
    python task4/train_transformer.py
or provide explicit paths with ``--csv`` and ``--output-dir``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "results" / "sentiment_dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "distilbert-feedback"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128


class FeedbackDataset(Dataset):
    """Tokenized feedback examples with integer class labels."""

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer: Any,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        self.labels = labels.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def load_feedback(csv_path: Path) -> tuple[pd.DataFrame, LabelEncoder]:
    """Read and validate the labelled CSV before training begins."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Feedback CSV not found: {csv_path}\n"
            "Create a CSV with 'text' and 'label' columns or pass --csv PATH."
        )

    try:
        frame = pd.read_csv(csv_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read feedback CSV {csv_path}: {error}") from error

    missing_columns = {"text", "label"} - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Feedback CSV is missing required column(s): {missing}")

    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip()
    frame = frame.loc[(frame["text"] != "") & (frame["label"] != "")].copy()

    if len(frame) < 10:
        raise ValueError(
            f"Only {len(frame)} valid labelled examples were found; "
            "provide at least 10."
        )

    encoder = LabelEncoder()
    frame["label_id"] = encoder.fit_transform(frame["label"])
    class_counts = frame["label_id"].value_counts()
    if len(class_counts) < 2:
        raise ValueError("At least two distinct labels are required for training.")
    if int(class_counts.min()) < 2:
        rare_label_id = int(class_counts.idxmin())
        rare_label = encoder.inverse_transform([rare_label_id])[0]
        raise ValueError(
            f"Label {rare_label!r} has fewer than two examples; "
            "each class needs at least two for a stratified train/validation split."
        )

    return frame.reset_index(drop=True), encoder


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Calculate validation accuracy and macro F1 for Hugging Face Trainer."""
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=1)
    true_labels = eval_pred.label_ids
    return {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "f1": float(
            f1_score(
                true_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }


def train(csv_path: Path, output_dir: Path) -> dict[str, float]:
    """Fine-tune the model, evaluate it, and save the model and label mapping."""
    frame, encoder = load_feedback(csv_path)

    try:
        train_frame, val_frame = train_test_split(
            frame,
            test_size=0.2,
            random_state=SEED,
            stratify=frame["label_id"],
        )
    except ValueError as error:
        raise ValueError(
            "Could not create a stratified training/validation split. "
            "Add more examples for each label."
        ) from error

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(encoder.classes_),
            id2label={
                index: label for index, label in enumerate(encoder.classes_)
            },
            label2id={
                label: index for index, label in enumerate(encoder.classes_)
            },
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Could not load pretrained model {MODEL_NAME!r}. "
            "Check internet access or whether the model is cached locally."
        ) from error

    train_dataset = FeedbackDataset(
        train_frame["text"].tolist(),
        train_frame["label_id"].to_numpy(),
        tokenizer,
    )
    val_dataset = FeedbackDataset(
        val_frame["text"].tolist(),
        val_frame["label_id"].to_numpy(),
        tokenizer,
    )

    checkpoint_dir = output_dir / "checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    start = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - start

    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)
    print(f"Training time in seconds: {training_seconds:.2f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "labels.txt").write_text(
        "\n".join(encoder.classes_) + "\n",
        encoding="utf-8",
    )

    print(f"Saved fine-tuned model and label mapping to: {output_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on a labelled feedback CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"CSV with text,label columns (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for the fine-tuned model (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    csv_path = (
        args.csv if args.csv.is_absolute() else (Path.cwd() / args.csv).resolve()
    )
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else (Path.cwd() / args.output_dir).resolve()
    )
    train(csv_path, output_dir)


if __name__ == "__main__":
    main()
