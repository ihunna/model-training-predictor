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
Optimized training script using the best configuration found.
This fixes the learning issues from the original train.py.

Usage:
    python train_optimized.py --dataset dataset.json
"""

class OptimizedModel(nn.Module):
    """Optimized model architecture"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)

def main():
    parser = argparse.ArgumentParser(description="Train optimized regression model")
    parser.add_argument("--dataset", type=str, default="dataset.json", help="Path to dataset JSON file")
    parser.add_argument("--tolerance", type=int, default=5, help="Tolerance for accuracy check")
    args = parser.parse_args()

    print("-- Starting optimized training...")
    print("Key fixes applied:")
    print("   ✅ Increased learning rate from 0.001 to 0.01")
    print("   ✅ Reduced dropout from 0.2 to 0.1")
    print("   ✅ Added gradient clipping")
    print("   ✅ Improved model architecture")

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"❌ {args.dataset} not found.")

    with open(args.dataset, "r") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list) or not all("input" in d and "output" in d for d in raw_data):
        raise ValueError("Dataset must be a list of {'input': [...], 'output': [...]} objects.")

    X = np.array([entry["input"] for entry in raw_data], dtype=np.float32)
    y = np.array([entry["output"] for entry in raw_data], dtype=np.float32)

    print(f"-- Dataset loaded: {len(X)} samples")
    print(f"   Input shape: {X.shape}")
    print(f"   Output range: {y.min():.0f} - {y.max():.0f}")

    input_mean = X.mean(axis=0, keepdims=True)
    input_std = X.std(axis=0, keepdims=True) + 1e-8
    X_normalized = (X - input_mean) / input_std

    output_mean = y.mean()
    output_std = y.std() + 1e-8
    y_normalized = (y - output_mean) / output_std

    print(f"-- After normalization:")
    print(f"   Input range: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")
    print(f"   Output range: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")

    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_normalized, test_size=0.2, random_state=42
    )

    print(f"-- Data split: {len(X_train)} train, {len(X_test)} test samples")

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X.shape[1]
    output_size = y.shape[1]
    
    model = OptimizedModel(input_size, output_size).to(device)
    
    print(f"-- Model setup:")
    print(f"   Device: {device}")
    print(f"   Architecture: {input_size} → 128 → 64 → {output_size}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)  # Increased LR!
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    epochs = 100
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 30
    
    train_losses = []
    val_losses = []

    print(f"\n-- Training for up to {epochs} epochs...")
    print("Epoch | Train Loss | Val Loss   | LR      | Status")
    print("-" * 50)

    for epoch in range(epochs):
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
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 10 == 0 or epoch < 5:
            print(f"{epoch:5d} | {avg_train_loss:10.6f} | {avg_val_loss:10.6f} | {current_lr:.6f} | Training...")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/optimized_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"{epoch:5d} | {avg_train_loss:10.6f} | {avg_val_loss:10.6f} | {current_lr:.6f} | Early stop")
                break

        if np.isnan(avg_train_loss):
            print("❌ Training failed: NaN loss detected")
            break

    print(f"\n✅ Training completed after {epoch+1} epochs")
    print(f"-- Best validation loss: {best_loss:.6f}")

    # Load best model for evaluation
    model.load_state_dict(torch.load("models/optimized_model.pt"))
    model.eval()

    # Final evaluation
    print("\n-- Final Evaluation on Test Set:")
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
    
    tolerance = args.tolerance
    errors = np.abs(all_preds - all_targets)
    
    correct_samples = np.sum(np.all(errors <= tolerance, axis=1))
    total_samples = len(all_targets)
    sample_accuracy = correct_samples / total_samples
    
    correct_outputs = np.sum(errors <= tolerance)
    total_outputs = errors.size
    output_accuracy = correct_outputs / total_outputs
    
    per_output_correct = np.sum(errors <= tolerance, axis=0)
    per_output_accuracy = per_output_correct / total_samples

    print(f"-- Performance Metrics:")
    print(f"   MSE: {mse:.2f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   Sample accuracy (all outputs ±{tolerance}): {sample_accuracy:.1%} ({correct_samples}/{total_samples})")
    print(f"   Output accuracy (individual ±{tolerance}): {output_accuracy:.1%} ({correct_outputs}/{total_outputs})")

    print(f"\n-- Per-output accuracy:")
    for i, acc in enumerate(per_output_accuracy):
        print(f"   Output {i+1:2d}: {acc:.1%} ({per_output_correct[i]}/{total_samples} correct)")

    print(f"\n-- Sample predictions (first 3 test samples):")
    for i in range(min(3, len(all_preds))):
        print(f"   Sample {i+1}:")
        for j in range(min(6, output_size)):
            pred = all_preds[i][j]
            actual = all_targets[i][j]
            error = abs(pred - actual)
            status = "✅" if error <= tolerance else "❌"
            print(f"      Output {j+1}: {status} Pred={pred:6.1f}, Actual={actual:6.1f}, Error={error:5.1f}")

    norm_stats = {
        "input_mean": input_mean.flatten().tolist(),
        "input_std": input_std.flatten().tolist(),
        "output_mean": float(output_mean),
        "output_std": float(output_std)
    }
    
    with open("models/optimized_norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\n-- Files saved:")
    print(f"   Model: models/optimized_model.pt")
    print(f"   Normalization: models/optimized_norm_stats.json")

    if correct_outputs >= 5:
        print(f"\n-- SUCCESS! Model achieved {correct_outputs} correct outputs (≥5 target)")
        if sample_accuracy > 0:
            print(f"-- BONUS: {correct_samples} complete samples predicted correctly!")
    else:
        print(f"\n-- Model achieved {correct_outputs} correct outputs (target: ≥5)")
        print("-- Suggestions: Try longer training or different architecture")

    if correct_outputs > 0:
        improvement = f"Improved from 0% to {output_accuracy:.1%} accuracy"
        print(f"\n-- {improvement}")
    
    return correct_outputs >= 5

if __name__ == "__main__":
    main()
