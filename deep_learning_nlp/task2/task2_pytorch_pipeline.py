import copy
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# 1. Load and split dataset
X, y = load_breast_cancer(return_X_y=True)
X = X.astype(np.float32)
y = y.astype(np.int64)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=SEED,
    stratify=y,
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=SEED,
    stratify=y_temp,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)


# 2. PyTorch dataset and loaders
class CsvDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


train_ds = CsvDataset(X_train, y_train)
val_ds = CsvDataset(X_val, y_val)
test_ds = CsvDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


# 3. MLP model
class MLP(nn.Module):
    def __init__(self, input_size=30, hidden_size=64, num_classes=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


model = MLP().to(device)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 4. Evaluation helper
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


# 5. Training loop with early stopping
best_accuracy = 0.0
best_state = None
best_epoch = 0
patience = 6
wait = 0
train_losses = []
val_accuracies = []

for epoch in range(40):
    model.train()
    total_loss = 0.0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / len(train_ds)
    val_accuracy = evaluate(model, val_loader)

    train_losses.append(avg_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1:02d}/40 | "
        f"Loss: {avg_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.4f}"
    )

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
        wait = 0
    else:
        wait += 1

    if wait >= patience:
        print("Early stopping triggered.")
        break

if best_state is None:
    raise RuntimeError("Training failed to produce any valid model state.")

model.load_state_dict(best_state)
model_path = MODELS_DIR / "breast_cancer_mlp.pth"
torch.save(best_state, model_path)

print(f"\nBest validation accuracy: {best_accuracy:.4f}")
print(f"Best epoch: {best_epoch}")

# 6. Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PyTorch MLP Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "task2_training_loss.png")

# 7. Reload into a new model and evaluate
reloaded_model = MLP().to(device)
reloaded_model.load_state_dict(torch.load(model_path, map_location=device))
reloaded_model.eval()

reloaded_val_accuracy = evaluate(reloaded_model, val_loader)
test_accuracy = evaluate(reloaded_model, test_loader)

print(f"Reloaded validation accuracy: {reloaded_val_accuracy:.4f}")
print(f"Final test accuracy: {test_accuracy:.4f}")

assert abs(reloaded_val_accuracy - best_accuracy) < 1e-6
print("Reloaded validation accuracy matches the saved best model.")

# 8. Compare with scikit-learn Logistic Regression
lr_model = LogisticRegression(max_iter=2000, random_state=SEED)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print(f"MLP test accuracy: {test_accuracy:.4f}")
print(f"Logistic Regression test accuracy: {lr_accuracy:.4f}")

# Optional: print improvement summary
if test_accuracy > lr_accuracy:
    print("MLP outperformed logistic regression on the test set.")
elif test_accuracy < lr_accuracy:
    print("Logistic regression outperformed the MLP on the test set.")
else:
    print("MLP and logistic regression performed similarly on the test set.")
