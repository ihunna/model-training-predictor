import json
import torch
import numpy as np

# Define model class (same as training)
class RegressionModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

# Load normalization parameters
with open("models/norm_stats.json", "r") as f:
    norm_params = json.load(f)

input_mean = np.array(norm_params["input_mean"])
input_std = np.array(norm_params["input_std"])
output_mean = norm_params["output_mean"]
output_std = norm_params["output_std"]

# Load dataset inputs
with open("dataset.json", "r") as f:
    data = json.load(f)

inputs = np.array(data["input"], dtype=np.float32)

num_outputs = len(data["output"])
num_features = len(inputs) // num_outputs

X = inputs.reshape(num_outputs, num_features)

# Normalize inputs
X_norm = (X - input_mean) / input_std

# Load model and weights
model = RegressionModel(num_features)
model.load_state_dict(torch.load("models/trained_model.pt"))
model.eval()

# Predict
with torch.no_grad():
    inputs_tensor = torch.tensor(X_norm, dtype=torch.float32)
    outputs = model(inputs_tensor).squeeze().numpy()

# Denormalize outputs
outputs_denorm = outputs * output_std + output_mean

print("Predictions:", outputs_denorm)
