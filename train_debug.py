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
import matplotlib.pyplot as plt
from collections import defaultdict

"""
Enhanced debug training script with comprehensive monitoring.
This will help identify why the model isn't learning.

Usage:
    python train_debug.py --dataset dataset.json --architecture simple
"""

class SimpleModel(nn.Module):
    """Ultra-simple model for debugging"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

class MediumModel(nn.Module):
    """Medium complexity model"""
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
    """Wider model with more neurons"""
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

class DeepModel(nn.Module):
    """Deep model with batch normalization"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)

class ResidualModel(nn.Module):
    """Model with residual connections"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_proj = nn.Linear(input_size, 128)
        self.layer1 = nn.Linear(128, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_proj(x))
        residual = x
        x = self.relu(self.layer1(x))
        x = x + residual  
        residual = x
        x = self.relu(self.layer2(x))
        x = x + residual 
        x = self.output_layer(x)
        return x

def get_model(architecture, input_size, output_size):
    """Get model based on architecture name"""
    models = {
        'simple': SimpleModel,
        'medium': MediumModel,
        'wide': WideModel,
        'deep': DeepModel,
        'residual': ResidualModel
    }
    
    if architecture not in models:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from: {list(models.keys())}")
    
    return models[architecture](input_size, output_size)

def monitor_gradients(model):
    """Monitor gradient magnitudes"""
    total_norm = 0
    param_count = 0
    grad_info = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            grad_info[name] = {
                'grad_norm': param_norm.item(),
                'param_norm': param.data.norm(2).item(),
                'grad_mean': param.grad.data.mean().item(),
                'grad_std': param.grad.data.std().item()
            }
    
    total_norm = total_norm ** (1. / 2)
    return total_norm, grad_info

def monitor_weights(model):
    """Monitor weight statistics"""
    weight_info = {}
    for name, param in model.named_parameters():
        weight_info[name] = {
            'mean': param.data.mean().item(),
            'std': param.data.std().item(),
            'min': param.data.min().item(),
            'max': param.data.max().item()
        }
    return weight_info

def test_tiny_dataset(model, criterion, optimizer, device):
    """Test if model can overfit on tiny dataset"""
    print("-- Testing model on tiny dataset (10 samples)...")
    
    tiny_X = torch.randn(10, 40).to(device)
    tiny_y = torch.sum(tiny_X[:, :5], dim=1, keepdim=True).repeat(1, 12).to(device)
    
    model.train()
    losses = []
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(tiny_X)
        loss = criterion(outputs, tiny_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"   Tiny dataset epoch {epoch}: loss = {loss.item():.6f}")
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = (initial_loss - final_loss) / initial_loss
    
    print(f"   Initial loss: {initial_loss:.6f}")
    print(f"   Final loss: {final_loss:.6f}")
    print(f"   Improvement: {improvement:.1%}")
    
    return improvement > 0.5

def analyze_data_distribution(X, y):
    """Analyze input and output distributions"""
    print("\n-- Data Distribution Analysis:")
    print(f"   Input shape: {X.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Input range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   Output range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"   Input mean: {X.mean():.2f}, std: {X.std():.2f}")
    print(f"   Output mean: {y.mean():.2f}, std: {y.std():.2f}")
    

    unique_outputs = np.unique(y, axis=0)
    print(f"   Unique output patterns: {len(unique_outputs)}")
    
    input_vars = np.var(X, axis=0)
    low_var_features = np.sum(input_vars < 0.1)
    print(f"   Low variance input features: {low_var_features}/{X.shape[1]}")

def main():
    parser = argparse.ArgumentParser(description="Debug training with comprehensive monitoring")
    parser.add_argument("--dataset", type=str, default="dataset.json", help="Path to dataset JSON file")
    parser.add_argument("--architecture", type=str, default="simple", 
                       choices=['simple', 'medium', 'wide', 'deep', 'residual'],
                       help="Model architecture to test")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", choices=['adam', 'sgd', 'rmsprop'])
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--normalization", type=str, default="zscore", 
                       choices=['zscore', 'minmax', 'robust', 'none'], help="Normalization strategy")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    args = parser.parse_args()

    print(f"-- Starting debug training with:")
    print(f"   Architecture: {args.architecture}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Optimizer: {args.optimizer}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Normalization: {args.normalization}")

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"❌ {args.dataset} not found.")

    with open(args.dataset, "r") as f:
        raw_data = json.load(f)

    X = np.array([entry["input"] for entry in raw_data], dtype=np.float32)
    y = np.array([entry["output"] for entry in raw_data], dtype=np.float32)


    analyze_data_distribution(X, y)
    if args.normalization == "zscore":
        input_mean = X.mean(axis=0, keepdims=True)
        input_std = X.std(axis=0, keepdims=True) + 1e-8
        X_norm = (X - input_mean) / input_std
        
        output_mean = y.mean()
        output_std = y.std() + 1e-8
        y_norm = (y - output_mean) / output_std
    elif args.normalization == "minmax":
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
    elif args.normalization == "robust":
        X_median = np.median(X, axis=0)
        X_iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0) + 1e-8
        X_norm = (X - X_median) / X_iqr
        
        y_median = np.median(y)
        y_iqr = np.percentile(y, 75) - np.percentile(y, 25) + 1e-8
        y_norm = (y - y_median) / y_iqr
    else:
        X_norm = X
        y_norm = y

    print(f"\n-- After {args.normalization} normalization:")
    print(f"   Input range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")
    print(f"   Output range: [{y_norm.min():.3f}, {y_norm.max():.3f}]")

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_norm, test_size=0.2, random_state=42
    )


    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X.shape[1]
    output_size = y.shape[1]
    
    model = get_model(args.architecture, input_size, output_size).to(device)
    
    print(f"\n-- Model architecture ({args.architecture}):")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    criterion = nn.MSELoss()

    tiny_success = test_tiny_dataset(model, criterion, optimizer, device)
    if not tiny_success:
        print("-- Model failed to overfit tiny dataset. Check architecture/hyperparameters.")
    else:
        print("✅ Model successfully overfitted tiny dataset.")

    model = get_model(args.architecture, input_size, output_size).to(device)
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    print(f"\n-- Starting training for {args.epochs} epochs...")
    
    train_losses = []
    val_losses = []
    gradient_norms = []
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        epoch_gradient_norms = []
        
        for batch_idx, (inputs_batch, labels_batch) in enumerate(train_loader):
            inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()

            grad_norm, grad_info = monitor_gradients(model)
            epoch_gradient_norms.append(grad_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item() * inputs_batch.size(0)
            
             
            if epoch == 0 and batch_idx < 3:
                print(f"   Batch {batch_idx}: loss={loss.item():.6f}, grad_norm={grad_norm:.6f}")
                sample_output = outputs[0].detach().cpu().numpy()
                sample_target = labels_batch[0].detach().cpu().numpy()
                print(f"      Sample pred: {sample_output[:3]} ... (first 3)")
                print(f"      Sample target: {sample_target[:3]} ... (first 3)")

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_grad_norm = np.mean(epoch_gradient_norms)
        
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
        gradient_norms.append(avg_grad_norm)
        
        if epoch % 10 == 0:
            weight_info = monitor_weights(model)
            first_layer_weights = list(weight_info.values())[0]
            print(f"Epoch {epoch:3d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
                  f"Grad: {avg_grad_norm:.6f} | Weights: {first_layer_weights['std']:.6f}")

        if np.isnan(avg_train_loss):
            print("❌ Loss became NaN. Stopping training.")
            break

    print("\n-- Final Evaluation:")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs_batch, labels_batch in test_loader:
            inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
            outputs = model(inputs_batch)
            
            if args.normalization == "zscore":
                preds_denorm = outputs.cpu().numpy() * output_std + output_mean
                targets_denorm = labels_batch.cpu().numpy() * output_std + output_mean
            elif args.normalization == "minmax":
                preds_denorm = outputs.cpu().numpy() * (y.max() - y.min()) + y.min()
                targets_denorm = labels_batch.cpu().numpy() * (y.max() - y.min()) + y.min()
            elif args.normalization == "robust":
                preds_denorm = outputs.cpu().numpy() * y_iqr + y_median
                targets_denorm = labels_batch.cpu().numpy() * y_iqr + y_median
            else:
                preds_denorm = outputs.cpu().numpy()
                targets_denorm = labels_batch.cpu().numpy()
            
            all_preds.extend(preds_denorm)
            all_targets.extend(targets_denorm)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mse = mean_squared_error(all_targets.flatten(), all_preds.flatten())
    mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
    

    tolerance = 5
    correct_outputs = np.sum(np.abs(all_preds - all_targets) <= tolerance)
    total_outputs = all_preds.size
    accuracy = correct_outputs / total_outputs
    
    print(f"   MSE: {mse:.2f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   Correct outputs (±{tolerance}): {correct_outputs}/{total_outputs} ({accuracy:.1%})")
    
    print("\n-- Sample predictions:")
    for i in range(min(3, len(all_preds))):
        print(f"   Sample {i+1}:")
        for j in range(min(6, output_size)):
            pred = all_preds[i][j]
            actual = all_targets[i][j]
            error = abs(pred - actual)
            status = "✅" if error <= tolerance else "❌"
            print(f"      Output {j+1}: {status} Pred={pred:6.1f}, Actual={actual:6.1f}, Error={error:5.1f}")

    results = {
        'architecture': args.architecture,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'batch_size': args.batch_size,
        'normalization': args.normalization,
        'final_mse': mse,
        'final_mae': mae,
        'accuracy': accuracy,
        'correct_outputs': int(correct_outputs),
        'total_outputs': int(total_outputs),
        'tiny_dataset_success': tiny_success,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'gradient_norms': gradient_norms
    }
    
    os.makedirs("debug_results", exist_ok=True)
    result_file = f"debug_results/{args.architecture}_{args.optimizer}_{args.learning_rate}_{args.normalization}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"-- Results saved to: {result_file}")
    
    if correct_outputs >= 5:
        print("-- SUCCESS! Model achieved 5+ correct outputs!")
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/debug_success_model.pt")
        print("-- Successful model saved as: models/debug_success_model.pt")
    else:
        print("-- Model needs improvement. Try different hyperparameters.")

if __name__ == "__main__":
    main()
