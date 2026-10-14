"""Classical sentiment classification with TF-IDF and pretrained word vectors."""

from __future__ import annotations

import argparse
import importlib
import re
import time
from pathlib import Path
from urllib.error import URLError

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

TOPICS = (
    "application",
    "delivery",
    "experience",
    "interface",
    "order",
    "product",
    "purchase",
    "quality",
    "service",
    "support",
)

POSITIVE_WORDS = (
    "amazing",
    "excellent",
    "fantastic",
    "helpful",
    "impressive",
    "pleasant",
    "reliable",
    "satisfying",
    "smooth",
    "wonderful",
)

NEGATIVE_WORDS = (
    "awful",
    "confusing",
    "disappointing",
    "frustrating",
    "horrible",
    "inconvenient",
    "unhelpful",
    "unreliable",
    "unsatisfactory",
    "terrible",
)

SENTENCE_TEMPLATES = (
    "The {topic} was {sentiment}.",
    "I found the {topic} {sentiment}.",
    "Overall, the {topic} feels {sentiment}.",
)


def clean_text(text: str) -> str:
    """Lowercase text and replace punctuation and digits with spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def create_dataset() -> pd.DataFrame:
    """Create 600 balanced, reproducible illustrative sentiment examples."""
    records: list[dict[str, str]] = []
    for label, words in (
        ("positive", POSITIVE_WORDS),
        ("negative", NEGATIVE_WORDS),
    ):
        for sentiment in words:
            for topic in TOPICS:
                for template in SENTENCE_TEMPLATES:
                    records.append(
                        {
                            "text": template.format(
                                topic=topic,
                                sentiment=sentiment,
                            ),
                            "label": label,
                        }
                    )

    dataset = pd.DataFrame.from_records(records)
    dataset["clean_text"] = dataset["text"].map(clean_text)
    if len(dataset) != 600 or dataset["label"].value_counts().to_dict() != {
        "positive": 300,
        "negative": 300,
    }:
        raise RuntimeError("Synthetic dataset must contain 300 examples per class.")
    return dataset


def build_pipeline() -> Pipeline:
    """Build a TF-IDF and logistic-regression text classifier."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
            ),
            ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )


def print_top_features(pipeline: Pipeline, count: int = 10) -> None:
    """Print the ten features most indicative of each class."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()

    # For binary classification, coef_[0] points toward classes_[1].
    weights_by_class = (
        (-classifier.coef_[0], classifier.coef_[0])
        if len(classifier.classes_) == 2
        else classifier.coef_
    )
    for class_name, weights in zip(classifier.classes_, weights_by_class):
        top_indices = np.argsort(weights)[-count:][::-1]
        print(f"\nTop words for {class_name}:")
        print(", ".join(feature_names[top_indices]))


def evaluate_glove(
    train_texts: pd.Series,
    test_texts: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate sentence vectors formed by averaging pretrained GloVe vectors."""
    try:
        api = importlib.import_module("gensim.downloader")
    except ImportError as error:
        raise RuntimeError(
            "GloVe evaluation requires a Python-compatible gensim install. "
            "Install gensim or rerun with --skip-glove."
        ) from error

    print("\nLoading GloVe vectors (the first run may download the model)...")
    glove = api.load("glove-wiki-gigaword-50")

    for word in ("king", "computer"):
        if word in glove:
            print(f"\nNearest words to {word}:")
            print(glove.most_similar(word, topn=5))

    if all(word in glove for word in ("king", "man", "woman")):
        print("\nking - man + woman:")
        print(
            glove.most_similar(
                positive=["king", "woman"],
                negative=["man"],
                topn=1,
            )
        )

    def sentence_vector(sentence: str) -> np.ndarray:
        vectors = [glove[word] for word in sentence.split() if word in glove]
        if not vectors:
            return np.zeros(50, dtype=np.float32)
        return np.mean(vectors, axis=0)

    train_vectors = np.vstack(train_texts.map(sentence_vector).to_numpy())
    test_vectors = np.vstack(test_texts.map(sentence_vector).to_numpy())
    classifier = LogisticRegression(max_iter=1000, random_state=SEED)
    classifier.fit(train_vectors, y_train)
    predictions = classifier.predict(test_vectors)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
    }


def run(include_glove: bool = True) -> None:
    """Train models, report metrics, and save all task outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = create_dataset()
    print(f"Dataset shape: {dataset.shape}")
    print("Class counts:")
    print(dataset["label"].value_counts().sort_index().to_string())
    print(dataset[["text", "label"]].head().to_string(index=False))

    X_train, X_test, y_train, y_test = train_test_split(
        dataset["clean_text"],
        dataset["label"],
        test_size=0.2,
        random_state=SEED,
        stratify=dataset["label"],
    )

    pipeline = build_pipeline()
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = pipeline.predict(X_test)
    inference_time = (time.perf_counter() - start) / len(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    macro_f1 = float(f1_score(y_test, predictions, average="macro"))

    print("\nTF-IDF + Logistic Regression")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Training time: {training_time:.6f} seconds")
    print(f"Mean inference time: {inference_time:.6f} seconds/sample")
    print(classification_report(y_test, predictions, zero_division=0))

    class_labels = ["negative", "positive"]
    matrix = confusion_matrix(y_test, predictions, labels=class_labels)
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_labels,
    ).plot(cmap="Blues", values_format="d")
    plt.title("TF-IDF Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "task3_confusion_matrix.png")
    plt.close()

    print_top_features(pipeline)

    comparison = [
        {
            "Model": "TF-IDF + Logistic Regression",
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Status": "completed",
        }
    ]
    if include_glove:
        try:
            glove_metrics = evaluate_glove(X_train, X_test, y_train, y_test)
        except (ImportError, OSError, RuntimeError, ValueError, URLError) as error:
            message = str(error)
            print(f"\nGloVe evaluation unavailable: {message}")
            comparison.append(
                {
                    "Model": "Average GloVe + Logistic Regression",
                    "Accuracy": np.nan,
                    "Macro F1": np.nan,
                    "Status": f"unavailable: {message}",
                }
            )
        else:
            print("\nAverage GloVe + Logistic Regression")
            print(f"Accuracy: {glove_metrics['accuracy']:.4f}")
            print(f"Macro F1: {glove_metrics['macro_f1']:.4f}")
            comparison.append(
                {
                    "Model": "Average GloVe + Logistic Regression",
                    "Accuracy": glove_metrics["accuracy"],
                    "Macro F1": glove_metrics["macro_f1"],
                    "Status": "completed",
                }
            )

    dataset.to_csv(RESULTS_DIR / "sentiment_dataset.csv", index=False)
    pd.DataFrame(comparison).to_csv(
        RESULTS_DIR / "task3_embedding_comparison.csv",
        index=False,
    )
    print(f"\nSaved dataset and comparison CSV under {RESULTS_DIR}.")
    print(f"Saved confusion matrix under {RESULTS_DIR}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a classical NLP sentiment baseline and compare embeddings."
    )
    parser.add_argument(
        "--skip-glove",
        action="store_true",
        help="skip pretrained GloVe evaluation (useful offline)",
    )
    args = parser.parse_args()
    run(include_glove=not args.skip_glove)


if __name__ == "__main__":
    main()
