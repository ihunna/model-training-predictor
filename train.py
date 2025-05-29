import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def main():
    with open("dataset.json", "r") as f:
        data = json.load(f)

    X = np.array(data["input"], dtype=np.float32).reshape(-1, 40)
    y = np.array(data["output"], dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)

    class RegressionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(40, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        def forward(self, x):
            return self.model(x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegressionModel().to(device)
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
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset):.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            preds = torch.round(outputs).int()
            targets = labels.squeeze().int()
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    print(f"\n✅ Exact match accuracy: {correct / total:.4f}")

    torch.save(model.state_dict(), "trained_model.pt")
    print("Model saved as trained_model.pt")

if __name__ == "__main__":
    main()
