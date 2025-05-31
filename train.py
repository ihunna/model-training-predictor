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
from tqdm import tqdm

"""
Enhanced training script with improved learning (for 0–2047 targets) and automatic optimization for any dataset size.

Major fixes to improve regression accuracy:
- Removed output normalization (train on raw 0–2047 targets)
- Switched from MSELoss → SmoothL1Loss (more robust to outliers)
- Lowered LR from 0.01 → 0.001 to avoid divergence
- Increased model capacity with an extra hidden layer
- Retained gradient clipping, LR scheduling, early stopping, memory optimizations

Usage:
    python train.py --dataset dataset.json --tolerance 5
    python train.py --dataset large_dataset.json --max_samples 100000
"""

class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Increased capacity: 128→64→32→16 hidden dims
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        return self.model(x)

def get_optimal_config(num_samples, device):
    """Automatically determine optimal training configuration based on dataset size and hardware."""
    available_ram = psutil.virtual_memory().available / (1024**3)  # GB
    config = {
        'batch_size': 32,
        'num_workers': 0,
        'pin_memory': False,
        'use_progress_bar': False,
        'memory_efficient': False,
        'max_epochs': 200
    }
    if num_samples < 1000:
        config.update({
            'batch_size': min(32, max(8, num_samples // 10)),
            'use_progress_bar': False,
            'memory_efficient': False,
            'max_epochs': 200
        })
        print(f"📊 Small dataset detected ({num_samples} samples) - Standard training")
    elif num_samples < 50000:
        config.update({
            'batch_size': 64 if available_ram > 4 else 32,
            'num_workers': min(4, os.cpu_count() or 1),
            'pin_memory': device.type == 'cuda',
            'use_progress_bar': True,
            'memory_efficient': False,
            'max_epochs': 150
        })
        print(f"📊 Medium dataset detected ({num_samples} samples) - Optimized training")
    else:
        optimal_batch_size = 128 if device.type == 'cuda' and available_ram > 8 else 64
        config.update({
            'batch_size': optimal_batch_size,
            'num_workers': min(6, os.cpu_count() or 1),
            'pin_memory': device.type == 'cuda',
            'use_progress_bar': True,
            'memory_efficient': True,
            'max_epochs': 100
        })
        print(f"📊 Large dataset detected ({num_samples} samples) - Memory-efficient training")
    print(f"🔧 Config: batch_size={config['batch_size']}, workers={config['num_workers']}, RAM={available_ram:.1f}GB")
    return config

def load_dataset_efficiently(dataset_path, max_samples=None):
    """Load dataset with memory management for large files."""
    file_size_mb = os.path.getsize(dataset_path) / (1024*1024)
    print(f"📁 Dataset file size: {file_size_mb:.1f}MB")
    if file_size_mb > 100:
        print("⚡ Loading large dataset with memory optimization...")
        with open(dataset_path, 'r') as f:
            raw_data = json.load(f)
        if max_samples and len(raw_data) > max_samples:
            print(f"✂️ Limiting to {max_samples} samples (from {len(raw_data)} total)")
            indices = np.linspace(0, len(raw_data)-1, max_samples, dtype=int)
            raw_data = [raw_data[i] for i in indices]
    else:
        print("📖 Loading dataset...")
        with open(dataset_path, 'r') as f:
            raw_data = json.load(f)
    return raw_data

def create_data_loaders(X, y, config, test_size=0.2):
    """Create optimized DataLoader objects."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=config['pin_memory'],
                              drop_last=len(train_dataset) > config['batch_size'])
    test_loader = DataLoader(test_dataset,
                             batch_size=config['batch_size'],
                             num_workers=config['num_workers'],
                             pin_memory=config['pin_memory'])
    return train_loader, test_loader, X_train, X_test, y_train, y_test

def train_with_progress(model, train_loader, test_loader, config, device):
    """Train model with progress tracking and memory management."""
    # Use SmoothL1Loss (robust L1-based regression)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    epochs = config['max_epochs']
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 25
    start_time = time.time()
    print(f"\n🎯 Starting training (up to {epochs} epochs)...")
    epoch_iterator = tqdm(range(epochs), desc="Training") if config['use_progress_bar'] else range(epochs)

    for epoch in epoch_iterator:
        model.train()
        total_loss = 0.0
        num_batches = 0
        batch_iter = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}") if config['use_progress_bar'] else train_loader
        for inputs_batch, labels_batch in batch_iter:
            inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            if config['memory_efficient'] and num_batches % 100 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

        avg_train_loss = total_loss / num_batches
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs_batch, labels_batch in test_loader:
                inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
                outputs = model(inputs_batch)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_loss / (val_batches if val_batches else 1)
        scheduler.step(avg_val_loss)

        # Reporting
        if config['use_progress_bar']:
            epoch_iterator.set_postfix({
                'train_loss': f'{avg_train_loss:.6f}',
                'val_loss': f'{avg_val_loss:.6f}',
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        elif epoch % 10 == 0 or epoch < 5:
            elapsed = time.time() - start_time
            eta = elapsed * (epochs / (epoch + 1)) - elapsed
            print(f"Epoch {epoch + 1:3d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | ETA: {eta/60:.1f}m")

        # Early stopping & save best
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/trained_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\n⏹️ Early stopping at epoch {epoch + 1}")
                break

        if np.isnan(avg_train_loss):
            print("❌ Training failed: NaN loss detected")
            break

        if config['memory_efficient'] and epoch % 10 == 0:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\n✅ Training complete. Best val loss: {best_loss:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Train regression model with memory-efficient config")
    parser.add_argument('--dataset', required=True, help="Path to JSON dataset file")
    parser.add_argument('--max_samples', type=int, default=None, help="Max number of samples to load (for large datasets)")
    parser.add_argument('--tolerance', type=float, default=5.0, help="Tolerance for accuracy metric")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    raw_data = load_dataset_efficiently(args.dataset, args.max_samples)

    # Helper to get a key that exists in the first item for targets
    possible_target_keys = ['target', 'output', 'label', 'y']

    target_key = None
    for key in possible_target_keys:
        if key in raw_data[0]:
            target_key = key
            break
    if target_key is None:
        raise KeyError(f"None of the keys {possible_target_keys} found in data items. Please check your dataset format.")

    # Extract inputs and targets safely
    inputs = np.array([item['input'] for item in raw_data], dtype=np.float32)
    targets = np.array([item[target_key] for item in raw_data], dtype=np.float32)

    # No normalization on outputs (0-2047 targets)
    # inputs normalization optional but recommended
    input_mean = inputs.mean(axis=0)
    input_std = inputs.std(axis=0) + 1e-8
    inputs = (inputs - input_mean) / input_std

    print(f"🔢 Dataset shape: inputs {inputs.shape}, targets {targets.shape}")

    config = get_optimal_config(len(inputs), device)

    train_loader, test_loader, X_train, X_test, y_train, y_test = create_data_loaders(inputs, targets, config)

    model = RegressionModel(input_size=inputs.shape[1], output_size=targets.shape[1]).to(device)

    train_with_progress(model, train_loader, test_loader, config, device)

    # Load best model for evaluation
    model.load_state_dict(torch.load("models/trained_model.pt"))
    model.eval()

    all_preds = []
    all_targets = []

    tolerance = args.tolerance

    with torch.no_grad():
        eval_iter = tqdm(test_loader, desc="Evaluating") if config['use_progress_bar'] else test_loader
        for Xb, yb in eval_iter:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)

    within_tolerance = np.abs(all_preds - all_targets) <= tolerance
    tolerance_accuracy = np.mean(np.all(within_tolerance, axis=1)) * 100

    print(f"\n📈 MSE: {mse:.4f}")
    print(f"📈 MAE: {mae:.4f}")
    print(f"📈 % predictions within ±{tolerance}: {tolerance_accuracy:.2f}%\n")


if __name__ == "__main__":
    main()
