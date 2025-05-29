import json
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset


"""
This script trains a simple regression model on your dataset and outputs a `.pt` model and normalization stats.

Usage:
    python train.py --dataset dataset.json --tolerance 5

– https://github.com/ihunna
"""


class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)


def main():
    parser = argparse.ArgumentParser(description="Train a regression model from JSON dataset")
    parser.add_argument("--dataset", type=str, default="dataset.json", help="Path to dataset JSON file")
    parser.add_argument("--tolerance", type=int, default=5, help="Tolerance for rounded accuracy check")
    args = parser.parse_args()

    dataset_path = args.dataset
    tolerance = args.tolerance

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ {dataset_path} not found. Please provide the correct dataset file.")

    with open(dataset_path, "r") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list) or not all("input" in d and "output" in d for d in raw_data):
        raise ValueError("❌ Dataset must be a list of {'input': [...], 'output': [...]} objects.")

    X = np.array([entry["input"] for entry in raw_data], dtype=np.float32)
    y = np.array([entry["output"] for entry in raw_data], dtype=np.float32)

    # Normalize input and output
    input_mean = X.mean(axis=0, keepdims=True)
    input_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - input_mean) / input_std

    output_mean = y.mean()
    output_std = y.std() + 1e-8
    y = (y - output_mean) / output_std

    # Ensure y is 2D (N, 1) if it's 1D
    if y.ndim == 1:
        y = y[:, np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X.shape[1]
    output_size = y.shape[1]
    model = RegressionModel(input_size, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs_batch, labels_batch in train_loader:
            inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    correct = total = 0

    with torch.no_grad():
        for inputs_batch, labels_batch in test_loader:
            inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
            outputs = model(inputs_batch)

            preds_denorm = outputs * output_std + output_mean
            targets_denorm = labels_batch * output_std + output_mean

            preds_np = preds_denorm.cpu().numpy()
            targets_np = targets_denorm.cpu().numpy()

            all_preds.extend(preds_np)
            all_targets.extend(targets_np)

            # Accuracy check with tolerance
            preds_rounded = np.round(preds_np).astype(int)
            targets_rounded = np.round(targets_np).astype(int)
            correct += (np.abs(preds_rounded - targets_rounded) <= tolerance).all(axis=1).sum()
            total += targets_rounded.shape[0]

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    accuracy = correct / total if total else 0

    print(f"\n📉 MSE: {mse:.2f}")
    print(f"📉 MAE: {mae:.2f}")
    print(f"✅ Accuracy within ±{tolerance}: {accuracy:.6f}")

    os.makedirs("models", exist_ok=True)
    model_path = "models/trained_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"📦 Model saved as: {model_path}")

    norm_stats = {
        "input_mean": input_mean.flatten().tolist(),
        "input_std": input_std.flatten().tolist(),
        "output_mean": output_mean.item() if np.isscalar(output_mean) else output_mean.tolist(),
        "output_std": output_std.item() if np.isscalar(output_std) else output_std.tolist()
    }
    norm_stats_path = "models/norm_stats.json"
    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats, f)
    print(f"📦 Normalization stats saved as: {norm_stats_path}")


if __name__ == "__main__":
    main()
