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
from tqdm import tqdm

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
    """Automatic config based on dataset size and hardware."""
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
    elif num_samples < 50000:
        config.update({
            'batch_size': 64 if available_ram > 4 else 32,
            'num_workers': min(4, os.cpu_count() or 1),
            'pin_memory': device.type == 'cuda',
            'use_progress_bar': True,
            'memory_efficient': False,
            'max_epochs': 150
        })
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
    return config

def load_dataset_efficiently(dataset_path, max_samples=None):
    file_size_mb = os.path.getsize(dataset_path) / (1024*1024)
    if file_size_mb > 100:
        with open(dataset_path, 'r') as f:
            raw_data = json.load(f)
        if max_samples and len(raw_data) > max_samples:
            indices = np.linspace(0, len(raw_data)-1, max_samples, dtype=int)
            raw_data = [raw_data[i] for i in indices]
    else:
        with open(dataset_path, 'r') as f:
            raw_data = json.load(f)
    return raw_data

def create_data_loaders(X, y, config, test_size=0.2):
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

    # Determine target key
    possible_target_keys = ['target', 'output', 'label', 'y']
    target_key = None
    for key in possible_target_keys:
        if key in raw_data[0]:
            target_key = key
            break
    if target_key is None:
        raise RuntimeError("Could not determine target key in dataset")

    X = np.array([item['input'] for item in raw_data])
    y = np.array([item[target_key] for item in raw_data])
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # Config defaults
    config = get_optimal_config(len(raw_data), device)
    if batch_size:
        config['batch_size'] = batch_size
    if max_epochs:
        config['max_epochs'] = max_epochs
    config['use_progress_bar'] = use_progress_bar

    train_loader, test_loader, X_train, X_test, y_train, y_test = create_data_loaders(X, y, config)

    model = RegressionModel(input_size=X.shape[1], output_size=y.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    epochs = config['max_epochs']

    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 25
    start_time = time.time()

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            if config['memory_efficient'] and num_batches % 100 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

        avg_train_loss = total_loss / num_batches

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

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/trained_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if np.isnan(avg_train_loss):
            print("Training failed: NaN loss detected")
            break

        if config['memory_efficient'] and epoch % 10 == 0:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    # Final evaluation on test
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    within_tol = np.mean(np.abs(y_test - preds) <= tolerance)

    return {'mse': mse, 'mae': mae, 'accuracy': within_tol}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train regression model with memory-efficient config")
    parser.add_argument('--dataset', required=True, help="Path to JSON dataset file")
    parser.add_argument('--max_samples', type=int, default=None, help="Max number of samples to load (for large datasets)")
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

        # Best hyperparameters applied here
        lr=0.001,
        batch_size=16,
        dropout=0.0,
        hidden_dims=[128, 64, 32],
        weight_decay=1e-6,
    )

    print("Training complete.")
    print(f"MSE: {results['mse']:.4f}, MAE: {results['mae']:.4f}, Accuracy (within tolerance): {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
