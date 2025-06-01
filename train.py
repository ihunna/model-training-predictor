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
import time
import psutil
import gc

try:
    from tqdm import tqdm
except ImportError:
    from tqdm.notebook import tqdm


class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims=[128, 64, 32], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_size
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def get_optimal_config(num_samples, device):
    available_ram = psutil.virtual_memory().available / (1024**3)
    config = {
        'batch_size': 32,
        'num_workers': 0,
        'pin_memory': False,
        'use_progress_bar': False,
        'memory_efficient': False,
        'max_epochs': 200
    }

    if num_samples < 1000:
        config.update({'batch_size': min(32, max(8, num_samples // 10)), 'max_epochs': 200})
    elif num_samples < 50000:
        config.update({
            'batch_size': 64 if available_ram > 4 else 32,
            'num_workers': min(4, os.cpu_count() or 1),
            'pin_memory': device.type == 'cuda',
            'use_progress_bar': True,
            'max_epochs': 150
        })
    else:
        config.update({
            'batch_size': 128 if device.type == 'cuda' and available_ram > 8 else 64,
            'num_workers': min(6, os.cpu_count() or 1),
            'pin_memory': device.type == 'cuda',
            'use_progress_bar': True,
            'memory_efficient': True,
            'max_epochs': 100
        })
    return config


def load_dataset_efficiently(dataset_path, max_samples=None):
    try:
        with open(dataset_path, 'r') as f:
            raw_data = json.load(f)
    except json.JSONDecodeError:
        print("Warning: JSON too large, switching to line-by-line mode.")
        raw_data = []
        with open(dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    raw_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    if max_samples and len(raw_data) > max_samples:
        indices = np.linspace(0, len(raw_data) - 1, max_samples, dtype=int)
        raw_data = [raw_data[i] for i in indices]
    return raw_data


def main():
    parser = argparse.ArgumentParser(description="Train a regression model from JSON dataset")
    parser.add_argument("--dataset", type=str, default="dataset.json", help="Path to dataset JSON file")
    parser.add_argument("--tolerance", type=int, default=5, help="Tolerance for rounded accuracy check")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to load from dataset")
    args = parser.parse_args()

    dataset_path = args.dataset
    tolerance = args.tolerance
    max_samples = args.max_samples

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ {dataset_path} not found. Please provide the correct dataset file.")

    raw_data = load_dataset_efficiently(dataset_path, max_samples=max_samples)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_optimal_config(len(X), device)

    print(f"-- Device: {device}")
    print(f"-- Using batch size: {config['batch_size']}, workers: {config['num_workers']}, pin_memory: {config['pin_memory']}")
    print(f"-- Max epochs: {config['max_epochs']}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_normalized, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=config['pin_memory'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                             num_workers=config['num_workers'], pin_memory=config['pin_memory'])

    input_size = X.shape[1]
    output_size = y_normalized.shape[1]
    model = RegressionModel(input_size, output_size, dropout=0.1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 30

    print(f"\n-- Starting training for up to {config['max_epochs']} epochs...")
    print("-- Key improvements: 10x higher learning rate, gradient clipping, better monitoring")

    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0
        if config['use_progress_bar']:
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        else:
            train_iter = train_loader
        for inputs_batch, labels_batch in train_iter:
            inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print("Stopping early due to no improvement...")
            break

        gc.collect()

    print("\n-- Training finished.")
    model.load_state_dict(torch.load("best_model.pth"))

    # Evaluate final model on test set
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs_batch, labels_batch in test_loader:
            inputs_batch = inputs_batch.to(device)
            outputs = model(inputs_batch).cpu().numpy()
            y_pred.append(outputs)
            y_true.append(labels_batch.numpy())

    y_pred = np.vstack(y_pred) * output_std + output_mean
    y_true = np.vstack(y_true) * output_std + output_mean

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"Final Test MSE: {mse:.6f}")
    print(f"Final Test MAE: {mae:.6f}")

    # Additional rounded accuracy check
    rounded_preds = np.round(y_pred)
    rounded_truth = np.round(y_true)
    rounded_correct = np.sum(np.abs(rounded_preds - rounded_truth) <= tolerance)
    total_values = np.prod(rounded_truth.shape)
    accuracy = rounded_correct / total_values * 100

    print(f"Rounded Accuracy (within ±{tolerance}): {accuracy:.2f}%")

if __name__ == "__main__":
    main()
