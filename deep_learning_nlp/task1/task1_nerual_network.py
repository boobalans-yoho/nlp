"""Task 1: simple neural network implementation.

This module contains a compact, self-contained MLP implementation that can be
used for small classification tasks. The code is intentionally dependency-light
so it works in a clean Python environment without needing external datasets.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


class NeuralNetwork:
    """A small fully connected neural network trained with gradient descent."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.5,
        seed: int = 42,
    ) -> None:
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError("All network dimensions must be positive integers.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(seed)

        self.weights1 = self.rng.normal(
            0.0,
            np.sqrt(2.0 / self.input_size),
            size=(self.input_size, self.hidden_size),
        )
        self.bias1 = np.zeros((1, self.hidden_size), dtype=float)

        self.weights2 = self.rng.normal(
            0.0,
            np.sqrt(2.0 / self.hidden_size),
            size=(self.hidden_size, self.output_size),
        )
        self.bias2 = np.zeros((1, self.output_size), dtype=float)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            shifted = x - np.max(x)
            exps = np.exp(shifted)
            return exps / np.sum(exps)
        if x.ndim != 2:
            raise ValueError("Softmax expects a 1D or 2D array-like input.")
        shifted = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _prepare_inputs(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim == 0:
            raise ValueError("X must contain at least one sample and one feature.")
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != self.input_size:
            raise ValueError(
                f"Expected input dimension {self.input_size}, got {matrix.shape[1]}."
            )
        return matrix

    def _prepare_targets(self, y: Iterable[int]) -> np.ndarray:
        targets = np.asarray(y, dtype=int)
        if targets.ndim == 0:
            targets = targets.reshape(1)
        if self.output_size == 1:
            return targets.reshape(-1, 1)
        return targets

    def forward(self, X: Iterable[Iterable[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        inputs = self._prepare_inputs(X)
        hidden_linear = inputs @ self.weights1 + self.bias1
        hidden_activation = self._sigmoid(hidden_linear)
        output_linear = hidden_activation @ self.weights2 + self.bias2
        if self.output_size == 1:
            output_activation = self._sigmoid(output_linear)
        else:
            output_activation = self._softmax(output_linear)
        return output_activation, hidden_activation, hidden_linear, output_linear

    def predict_proba(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        probabilities, _, _, _ = self.forward(X)
        return probabilities

    def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        probabilities = self.predict_proba(X)
        if self.output_size == 1:
            return (probabilities >= 0.5).astype(int).reshape(-1)
        return np.argmax(probabilities, axis=1)

    def fit(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[int],
        epochs: int = 5000,
        verbose: bool = False,
    ) -> "NeuralNetwork":
        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        inputs = self._prepare_inputs(X)
        targets = self._prepare_targets(y)

        if len(inputs) == 0 or len(targets) == 0:
            raise ValueError("X and y must not be empty.")
        if len(inputs) != len(targets):
            raise ValueError("X and y must contain the same number of samples.")

        if self.output_size == 1:
            target_values = targets.astype(float).reshape(-1, 1)
            target_matrix = target_values
        else:
            if np.any(targets < 0) or np.any(targets >= self.output_size):
                raise ValueError("Class labels must be in [0, output_size).")
            target_matrix = np.zeros((len(targets), self.output_size), dtype=float)
            target_matrix[np.arange(len(targets)), targets] = 1.0

        for epoch in range(1, epochs + 1):
            probabilities, hidden_activation, _, _ = self.forward(inputs)

            output_error = (probabilities - target_matrix) / len(inputs)
            grad_weights2 = hidden_activation.T @ output_error
            grad_bias2 = np.sum(output_error, axis=0, keepdims=True)
            hidden_error = (output_error @ self.weights2.T) * (
                hidden_activation * (1.0 - hidden_activation)
            )
            grad_weights1 = inputs.T @ hidden_error
            grad_bias1 = np.sum(hidden_error, axis=0, keepdims=True)

            self.weights2 -= self.learning_rate * grad_weights2
            self.bias2 -= self.learning_rate * grad_bias2
            self.weights1 -= self.learning_rate * grad_weights1
            self.bias1 -= self.learning_rate * grad_bias1

            if verbose and epoch % 10 == 0:
                loss = self.loss(inputs, target_matrix)
                print(f"Epoch {epoch}: loss={loss:.6f}")

        return self

    def loss(self, X: Iterable[Iterable[float]], y: Optional[Iterable[int]] = None) -> float:
        probabilities, _, _, _ = self.forward(X)
        if y is None:
            if self.output_size == 1:
                target = np.asarray(self.predict(X), dtype=float).reshape(-1, 1)
                return -np.mean(
                    target * np.log(np.clip(probabilities, 1e-9, 1.0))
                    + (1.0 - target) * np.log(np.clip(1.0 - probabilities, 1e-9, 1.0))
                )
            return -np.mean(np.log(np.clip(probabilities[np.arange(len(probabilities)), np.argmax(probabilities, axis=1)], 1e-9, 1.0)))

        y_array = np.asarray(y)
        if self.output_size == 1:
            y_array = y_array.astype(float).reshape(-1, 1)
            if len(y_array) != len(probabilities):
                raise ValueError("X and y must contain the same number of samples.")
            return -np.mean(
                y_array * np.log(np.clip(probabilities, 1e-9, 1.0))
                + (1.0 - y_array) * np.log(np.clip(1.0 - probabilities, 1e-9, 1.0))
            )

        if y_array.ndim == 1:
            labels = y_array.astype(int)
            if len(labels) != len(probabilities):
                raise ValueError("X and y must contain the same number of samples.")
            if np.any(labels < 0) or np.any(labels >= self.output_size):
                raise ValueError("Class labels must be in [0, output_size).")
            y_array = np.eye(self.output_size, dtype=float)[labels]
        else:
            y_array = y_array.astype(float)
            if y_array.shape != probabilities.shape:
                raise ValueError(
                    "One-hot targets must have the same shape as the predictions."
                )
        return -np.mean(np.sum(y_array * np.log(np.clip(probabilities, 1e-9, 1.0)), axis=1))

    def accuracy(self, X: Iterable[Iterable[float]], y: Iterable[int]) -> float:
        predictions = self.predict(X)
        labels = np.asarray(y, dtype=int).reshape(-1)
        return float(np.mean(predictions == labels))


def train_neural_network(
    X: Iterable[Iterable[float]],
    y: Iterable[int],
    hidden_size: int = 8,
    learning_rate: float = 0.5,
    epochs: int = 5000,
    seed: int = 42,
) -> NeuralNetwork:
    """Create and train a network using the provided data."""
    features = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=int)
    if features.ndim == 0:
        raise ValueError("X must contain at least one sample and one feature.")
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if labels.ndim == 0:
        labels = labels.reshape(1)
    if features.size == 0 or labels.size == 0:
        raise ValueError("X and y must not be empty.")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    input_size = features.shape[1]
    output_size = int(np.max(labels)) + 1 if len(labels) > 0 else 1
    model = NeuralNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    model.fit(features, labels, epochs=epochs, verbose=False)
    return model


def evaluate_model(model: NeuralNetwork, X: Iterable[Iterable[float]], y: Iterable[int]) -> float:
    """Return the classification accuracy of the given model."""
    return model.accuracy(X, y)


def make_toy_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Create a small, deterministic dataset for quick smoke tests."""
    features = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.9, 0.9],
            [0.1, 0.1],
        ],
        dtype=float,
    )
    labels = np.array([0, 1, 1, 0, 1, 1, 0, 0], dtype=int)
    return features, labels


def main() -> None:
    """Train on a small toy dataset and print the resulting accuracy."""
    X, y = make_toy_dataset()
    model = train_neural_network(
        X,
        y,
        hidden_size=6,
        learning_rate=0.2,
        epochs=2000,
        seed=7,
    )
    accuracy = evaluate_model(model, X, y)
    print(f"Toy dataset accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
