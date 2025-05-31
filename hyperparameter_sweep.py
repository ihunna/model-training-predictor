import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import time

"""
Comprehensive hyperparameter sweep to find optimal configuration.
This will test multiple combinations and rank them by performance.
"""

class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

class MediumModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)

class WideModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.model(x)

def get_model(architecture, input_size, output_size):
    models = {
        'simple': SimpleModel,
        'medium': MediumModel,
        'wide': WideModel
    }
    return models[architecture](input_size, output_size)

def normalize_data(X, y, method):
    if method == "zscore":
        input_mean = X.mean(axis=0, keepdims=True)
        input_std = X.std(axis=0, keepdims=True) + 1e-8
        X_norm = (X - input_mean) / input_std
        
        output_mean = y.mean()
        output_std = y.std() + 1e-8
        y_norm = (y - output_mean) / output_std
        return X_norm, y_norm, (input_mean, input_std, output_mean, output_std)
    
    elif method == "minmax":
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        y_min, y_max = y.min(), y.max()
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        y_norm = (y - y_min) / (y_max - y_min + 1e-8)
        return X_norm, y_norm, (X_min, X_max, y_min, y_max)
    
    else:  # no normalization
        return X, y, None

def denormalize_outputs(y_norm, y_true_norm, norm_params, method):
    if method == "zscore":
        _, _, output_mean, output_std = norm_params
        y_pred = y_norm * output_std + output_mean
        y_true = y_true_norm * output_std + output_mean
    elif method == "minmax":
        _, _, y_min, y_max = norm_params
        y_pred = y_norm * (y_max - y_min) + y_min
        y_true = y_true_norm * (y_max - y_min) + y_min
    else:
        y_pred = y_norm
        y_true = y_true_norm
    
    return y_pred, y_true

def train_and_evaluate(config, X, y, device):
    """Train model with given configuration and return performance metrics"""
    
    X_norm, y_norm, norm_params = normalize_data(X, y, config['normalization'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_norm, test_size=0.2, random_state=42
    )
    

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                 torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                                torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    input_size = X.shape[1]
    output_size = y.shape[1]
    model = get_model(config['architecture'], input_size, output_size).to(device)
    
    
    if config['optimizer'] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    elif config['optimizer'] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=1e-5)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        
        for inputs_batch, labels_batch in train_loader:
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
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        
        if np.isnan(avg_train_loss):
            return None
        
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs_batch, labels_batch in test_loader:
            inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
            outputs = model(inputs_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(labels_batch.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    all_preds_denorm, all_targets_denorm = denormalize_outputs(
        all_preds, all_targets, norm_params, config['normalization']
    )
    
    mse = mean_squared_error(all_targets_denorm.flatten(), all_preds_denorm.flatten())
    mae = mean_absolute_error(all_targets_denorm.flatten(), all_preds_denorm.flatten())
    
    tolerance = 5
    correct_outputs = np.sum(np.abs(all_preds_denorm - all_targets_denorm) <= tolerance)
    total_outputs = all_preds_denorm.size
    accuracy = correct_outputs / total_outputs
    
    loss_improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] if train_losses[0] > 0 else 0
    
    return {
        'mse': mse,
        'mae': mae,
        'accuracy': accuracy,
        'correct_outputs': int(correct_outputs),
        'total_outputs': int(total_outputs),
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'loss_improvement': loss_improvement,
        'converged': not np.isnan(train_losses[-1])
    }

def main():
    print("🔍 Starting comprehensive hyperparameter sweep...")
    
    with open("dataset.json", "r") as f:
        raw_data = json.load(f)
    
    X = np.array([entry["input"] for entry in raw_data], dtype=np.float32)
    y = np.array([entry["output"] for entry in raw_data], dtype=np.float32)
    
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} inputs, {y.shape[1]} outputs")
    
    hyperparameters = {
        'architecture': ['simple', 'medium', 'wide'],
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'optimizer': ['adam', 'sgd'],
        'batch_size': [16, 32, 64],
        'normalization': ['zscore', 'minmax', 'none'],
        'epochs': [30] 
    }
    
    param_names = list(hyperparameters.keys())
    param_values = list(hyperparameters.values())
    all_combinations = list(product(*param_values))
    
    print(f"🎯 Testing {len(all_combinations)} configurations...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    
    for i, combination in enumerate(all_combinations):
        config = dict(zip(param_names, combination))
        
        print(f"\n[{i+1}/{len(all_combinations)}] Testing: {config}")
        
        start_time = time.time()
        try:
            result = train_and_evaluate(config, X, y, device)
            if result is not None:
                result['config'] = config
                result['training_time'] = time.time() - start_time
                results.append(result)
                
                print(f"   ✅ MSE: {result['mse']:.1f}, MAE: {result['mae']:.1f}, "
                      f"Accuracy: {result['accuracy']:.1%}, Correct: {result['correct_outputs']}")
            else:
                print(f"   ❌ Failed (NaN loss)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    results.sort(key=lambda x: (-x['accuracy'], x['mse']))
    
    print(f"\n🏆 Top 10 configurations:")
    print("=" * 100)
    
    for i, result in enumerate(results[:10]):
        config = result['config']
        print(f"{i+1:2d}. Accuracy: {result['accuracy']:6.1%} | MSE: {result['mse']:8.1f} | "
              f"Correct: {result['correct_outputs']:3d} | "
              f"Arch: {config['architecture']:6s} | LR: {config['learning_rate']:5.3f} | "
              f"Opt: {config['optimizer']:4s} | Norm: {config['normalization']:6s}")
    
    # Save all results
    os.makedirs("sweep_results", exist_ok=True)
    with open("sweep_results/hyperparameter_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n-- All results saved to: sweep_results/hyperparameter_sweep_results.json")
    
    # Save best configuration
    if results:
        best_config = results[0]['config']
        best_metrics = {k: v for k, v in results[0].items() if k != 'config'}
        
        best_config_data = {
            'best_config': best_config,
            'best_metrics': best_metrics
        }
        
        with open("sweep_results/best_configuration.json", "w") as f:
            json.dump(best_config_data, f, indent=2)
        
        print(f"-- Best configuration found:")
        print(f"   Architecture: {best_config['architecture']}")
        print(f"   Learning Rate: {best_config['learning_rate']}")
        print(f"   Optimizer: {best_config['optimizer']}")
        print(f"   Batch Size: {best_config['batch_size']}")
        print(f"   Normalization: {best_config['normalization']}")
        print(f"   Performance: {results[0]['accuracy']:.1%} accuracy, {results[0]['correct_outputs']} correct outputs")
        
        if results[0]['correct_outputs'] >= 5:
            print("-- SUCCESS! Best configuration meets the target of 5+ correct outputs!")
        
        print(f"-- Best configuration saved to: sweep_results/best_configuration.json")

if __name__ == "__main__":
    main()
