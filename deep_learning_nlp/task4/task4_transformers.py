"""Demonstrate Hugging Face tokenization, sentiment analysis, and NER."""

from __future__ import annotations

from transformers import AutoTokenizer, pipeline

TOKENIZER_MODEL = "distilbert-base-uncased"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
NER_MODEL = "dslim/bert-base-NER"

TOKENIZER_SENTENCES = [
    "I love the service and the helpful support team.",
    "My payment was deducted twice. Please refund me.",
]

SENTIMENT_TEXTS = [
    "The product is excellent.",
    "I am unhappy with this service.",
    "The delivery arrived on time.",
]

NER_TEXTS = [
    "Ravi works at Google in Chennai.",
    "Microsoft opened an office in London.",
    "Priya joined Amazon in Bengaluru.",
]


def show_tokenization(tokenizer) -> None:
    """Print tokens, token IDs, and attention masks for the sample text."""
    print("\nTokenizer demonstration:")
    for sentence in TOKENIZER_SENTENCES:
        encoded = tokenizer(
            sentence,
            truncation=True,
            padding=False,
        )
        print(f"\nSentence: {sentence}")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")
        print(f"Input IDs: {encoded['input_ids']}")
        print(f"Attention mask: {encoded['attention_mask']}")


def run_sentiment_analysis() -> None:
    """Run and display pretrained sentiment classification results."""
    classifier = pipeline(
        task="sentiment-analysis",
        model=SENTIMENT_MODEL,
    )
    results = classifier(SENTIMENT_TEXTS)

    print("\nSentiment results:")
    for text, result in zip(SENTIMENT_TEXTS, results):
        print(f"{text}\n  {result}")


def run_named_entity_recognition() -> None:
    """Run and display aggregated named entities for the sample text."""
    ner = pipeline(
        task="ner",
        model=NER_MODEL,
        aggregation_strategy="simple",
    )
    results_by_text = ner(NER_TEXTS)

    print("\nNamed entity recognition results:")
    for text, results in zip(NER_TEXTS, results_by_text):
        print(f"\nText: {text}")
        if results:
            for entity in results:
                print(
                    "  "
                    f"{entity['word']} "
                    f"({entity['entity_group']}, "
                    f"score={entity['score']:.4f})"
                )
        else:
            print("  No entities detected.")


def main() -> None:
    """Load the tokenizer and run the three demonstrations."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Could not load tokenizer {TOKENIZER_MODEL!r}. "
            "Check your internet connection and Hugging Face access."
        ) from error

    show_tokenization(tokenizer)
    run_sentiment_analysis()
    run_named_entity_recognition()


if __name__ == "__main__":
    main()
