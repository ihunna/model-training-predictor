import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
import time
import psutil
import gc

# Use tqdm.notebook when available
try:
    from tqdm import tqdm
except ImportError:
    from tqdm.notebook import tqdm


class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims=[128, 64, 32], dropout=0.0):
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


def create_data_loaders(X, y, config, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=len(train_dataset) > config['batch_size']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    return train_loader, test_loader, X_train, X_test, y_train, y_test


def detect_target_key(sample):
    """Detect target key in sample dict."""
    for key in ['target', 'output', 'label', 'y']:
        if key in sample:
            return key
    raise RuntimeError("Target key not found. Expected one of: 'target', 'output', 'label', 'y'")


def train_model(
    dataset_path,
    device,
    max_samples=None,
    tolerance=5.0,
    hidden_dims=[128, 64, 32],
    dropout=0.0,
    lr=0.001,
    weight_decay=1e-6,
    max_epochs=None,
    batch_size=None,
    use_progress_bar=True,
):
    raw_data = load_dataset_efficiently(dataset_path, max_samples)
    target_key = detect_target_key(raw_data[0])

    X = np.asarray([item['input'] for item in raw_data], dtype=np.float32)
    y = np.asarray([item[target_key] for item in raw_data], dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    config = get_optimal_config(len(raw_data), device)
    if batch_size:
        config['batch_size'] = batch_size
    if max_epochs:
        config['max_epochs'] = max_epochs
    config['use_progress_bar'] = use_progress_bar

    train_loader, test_loader, _, X_test, _, y_test = create_data_loaders(X, y, config)

    model = RegressionModel(input_size=X.shape[1], output_size=y.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 25
    start_time = time.time()
    epochs = config['max_epochs']
    epoch_iterator = tqdm(range(epochs), desc="Training") if config['use_progress_bar'] else range(epochs)

    for epoch in epoch_iterator:
        model.train()
        total_loss = 0.0
        batch_iter = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}") if config['use_progress_bar'] else train_loader

        for inputs_batch, labels_batch in batch_iter:
            inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            if config['memory_efficient']:
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                gc.collect()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs_batch, labels_batch in test_loader:
                inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
                outputs = model(inputs_batch)
                val_loss += criterion(outputs, labels_batch).item()
        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)

        if config['use_progress_bar']:
            epoch_iterator.set_postfix({
                'train_loss': f'{avg_train_loss:.6f}',
                'val_loss': f'{avg_val_loss:.6f}',
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        elif epoch % 10 == 0 or epoch < 5:
            elapsed = time.time() - start_time
            eta = elapsed * (epochs / (epoch + 1)) - elapsed
            print(f"Epoch {epoch+1:3d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | ETA: {eta/60:.1f}m")

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/trained_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if np.isnan(avg_train_loss):
            print("Training failed: NaN loss detected")
            break

    # Final evaluation
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    within_tol = np.mean(np.abs(preds - y_test) <= tolerance)

    return {'mse': mse, 'mae': mae, 'accuracy': within_tol}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train regression model with memory-efficient config")
    parser.add_argument('--dataset', required=True, help="Path to JSON dataset file")
    parser.add_argument('--max_samples', type=int, default=None, help="Max number of samples to load")
    parser.add_argument('--tolerance', type=float, default=5.0, help="Tolerance for accuracy metric")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results = train_model(
        dataset_path=args.dataset,
        device=device,
        max_samples=args.max_samples,
        tolerance=args.tolerance,
        use_progress_bar=True,
        lr=0.001,
        batch_size=16,
        dropout=0.0,
        hidden_dims=[128, 64, 32],
        weight_decay=1e-6
    )

    print("\nTraining complete.")
    print(f"MSE: {results['mse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"Accuracy (within tolerance): {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
