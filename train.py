import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


""" 
This script trains a simple regression model on your dataset (just drop in `dataset.json`) and gives you a ready-to-use `.pt` file.

- Reads your input/output data from JSON
- Automatically figures out how to shape it for training
- Normalizes inputs/outputs for better training
- Trains a small neural net using PyTorch
- Prints out loss per epoch + simple exact match accuracy (rounded predictions)
- Saves your trained model to the `models/` folder

Just upload your dataset, run the file, and download your model. No headaches.
Perfect if you need something working fast.

– https://github.com/ihunna
"""


def main():
    dataset_path = "dataset.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ {dataset_path} not found. Please upload it.")

    with open(dataset_path, "r") as f:
        data = json.load(f)

    inputs = data["input"]
    outputs = data["output"]

    num_outputs = len(outputs)
    num_features = len(inputs) // num_outputs
    if len(inputs) % num_outputs != 0:
        print(f"⚠️ Warning: {len(inputs) % num_outputs} leftover inputs ignored to match output count.")
    inputs = inputs[:num_outputs * num_features]

    X = np.array(inputs, dtype=np.float32).reshape(num_outputs, num_features)
    y = np.array(outputs, dtype=np.float32)

    # Normalize inputs and outputs for better training
    X = X / 21.0  # input range: 0-21
    y = y / 2047.0  # output range: 0-2047

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)


    class RegressionModel(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        def forward(self, x):
            return self.model(x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegressionModel(num_features).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(10):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset):.6f}")


    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()

            # Denormalize predictions and labels before rounding
            preds = torch.round(outputs * 2047).int()
            targets = torch.round(labels.squeeze() * 2047).int()

            correct += (preds == targets).sum().item()
            total += targets.size(0)

    print(f"\n✅ Exact match accuracy: {correct / total:.6f}")

    os.makedirs("models", exist_ok=True)
    model_path = "models/trained_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"📦 Model saved as: {model_path}")

if __name__ == "__main__":
    main()
