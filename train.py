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
Improved training script with proper normalization and model architecture.

Usage:
    python train.py --dataset dataset.json --tolerance 5

– https://github.com/ihunna
"""

class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
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
        raise ValueError("-- Dataset must be a list of {'input': [...], 'output': [...]} objects.")

    X = np.array([entry["input"] for entry in raw_data], dtype=np.float32)
    y = np.array([entry["output"] for entry in raw_data], dtype=np.float32)

    print(f"-- Dataset loaded: {len(X)} samples")
    print(f"-- Input shape: {X.shape}")
    print(f"-- Output range: {y.min():.1f} - {y.max():.1f}")

    input_mean = X.mean(axis=0, keepdims=True)
    input_std = X.std(axis=0, keepdims=True) + 1e-8
    X_normalized = (X - input_mean) / input_std

    output_mean = y.mean()
    output_std = y.std() + 1e-8
    y_normalized = (y - output_mean) / output_std

    print(f"-- Normalized output range: {y_normalized.min():.3f} - {y_normalized.max():.3f}")

    print(f"-- y shape: {y.shape}")
    print(f"-- y_normalized shape: {y_normalized.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_normalized, test_size=0.2, random_state=42
    )

    print(f"-- Training samples: {len(X_train)}")
    print(f"-- Test samples: {len(X_test)}")

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X.shape[1]  # 40 features
    output_size = y_normalized.shape[1]  # 12 outputs
    model = RegressionModel(input_size, output_size).to(device)
    
    print(f"-- Device: {device}")
    print(f"-- Model input size: {input_size}")
    print(f"-- Model output size: {output_size}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    epochs = 200
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 25

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
        
        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs_batch, labels_batch in test_loader:
                inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
                outputs = model(inputs_batch)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item() * inputs_batch.size(0)
        
        avg_val_loss = val_loss / len(test_loader.dataset)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/trained_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"-- Early stopping at epoch {epoch + 1}")
                break

    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs_batch, labels_batch in test_loader:
            inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
            outputs = model(inputs_batch)
            
            preds_denorm = outputs.cpu().numpy() * output_std + output_mean
            targets_denorm = labels_batch.cpu().numpy() * output_std + output_mean
            
            all_preds.extend(preds_denorm)
            all_targets.extend(targets_denorm)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mse = mean_squared_error(all_targets.flatten(), all_preds.flatten())
    mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
    
    correct_samples = 0
    total_samples = len(all_targets)
    
    for i in range(total_samples):
        errors = np.abs(all_preds[i] - all_targets[i])
        if np.all(errors <= tolerance):
            correct_samples += 1
    
    accuracy = correct_samples / total_samples if total_samples else 0
    
    per_output_correct = np.sum(np.abs(all_preds - all_targets) <= tolerance, axis=0)
    per_output_accuracy = per_output_correct / total_samples

    print(f"\n-- FINAL RESULTS:")
    print(f"-- MSE: {mse:.2f}")
    print(f"-- MAE: {mae:.2f}")
    print(f"-- Full sample accuracy within ±{tolerance}: {accuracy:.6f}")
    print(f"-- Correct full samples: {correct_samples}/{total_samples}")
    print(f"-- Per-output accuracy: {per_output_accuracy}")
    print(f"-- Total correct individual outputs: {np.sum(per_output_correct)}/{total_samples * 12}")

    print(f"\n-- Sample predictions (first 3 samples, first 6 outputs):")
    for i in range(min(3, len(all_preds))):
        print(f"   Sample {i+1}:")
        for j in range(min(6, output_size)):
            pred = all_preds[i][j]
            actual = all_targets[i][j] 
            error = abs(pred - actual)
            status = "✅" if error <= tolerance else "❌"
            print(f"     Output {j+1}: {status} Pred={pred:.1f}, Actual={actual:.1f}, Error={error:.1f}")

    norm_stats = {
        "input_mean": input_mean.flatten().tolist(),
        "input_std": input_std.flatten().tolist(),
        "output_mean": float(output_mean),
        "output_std": float(output_std)
    }
    
    norm_stats_path = "models/norm_stats.json"
    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    
    print(f"-- Model saved as: models/trained_model.pt")
    print(f"-- Normalization stats saved as: {norm_stats_path}")

    individual_correct = np.sum(per_output_correct)
    if individual_correct >= 5:
        print(f"-- SUCCESS! Model got {individual_correct} individual outputs right (≥5 target)")
    elif correct_samples >= 1:
        print(f"-- GOOD! Model got {correct_samples} complete samples right")
    else:
        print(f"⚠️  Model needs improvement. Only {individual_correct} individual outputs correct.")

if __name__ == "__main__":
    main()
