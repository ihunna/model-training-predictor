import json
import torch
import numpy as np
import argparse

"""
Prediction script for multi-output regression (40 inputs → 12 outputs).

Usage:
    python predict.py --input "[1,2,3,...,40]" (40 values)
    python predict.py --test_dataset
"""

class RegressionModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

def load_model_and_stats():
    """Load the trained model and normalization statistics."""
    
    try:
        with open("models/norm_stats.json", "r") as f:
            norm_stats = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("❌ Normalization stats not found. Please train the model first.")
    
    input_mean = np.array(norm_stats["input_mean"])
    input_std = np.array(norm_stats["input_std"])
    output_mean = norm_stats["output_mean"]
    output_std = norm_stats["output_std"]
    
    try:
        input_size = len(input_mean) 
        output_size = 12 
        model = RegressionModel(input_size, output_size)
        model.load_state_dict(torch.load("models/trained_model.pt", map_location='cpu'))
        model.eval()
    except FileNotFoundError:
        raise FileNotFoundError("❌ Trained model not found. Please train the model first.")
    
    return model, input_mean, input_std, output_mean, output_std

def predict_single(input_vector, model, input_mean, input_std, output_mean, output_std):
    """Make prediction for a single input vector."""
    
    input_array = np.array(input_vector, dtype=np.float32)
    
    if len(input_array) != len(input_mean):
        raise ValueError(f"❌ Input size mismatch. Expected {len(input_mean)}, got {len(input_array)}")
    
    input_normalized = (input_array - input_mean) / input_std
    
    with torch.no_grad():
        input_tensor = torch.tensor(input_normalized, dtype=torch.float32).unsqueeze(0)
        output_normalized = model(input_tensor).squeeze().numpy()
    

    outputs = output_normalized * output_std + output_mean
    
    return outputs.tolist()

def test_on_dataset():
    """Test the model on the original dataset."""
    try:
        with open("dataset.json", "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("❌ Dataset not found.")
    
    model, input_mean, input_std, output_mean, output_std = load_model_and_stats()
    
    print("🧪 Testing model on dataset...")
    print("=" * 80)
    
    correct_samples = 0
    total_correct_outputs = 0
    total_outputs = 0
    tolerance = 5
    
    for i, entry in enumerate(dataset):
        input_vec = entry["input"]
        actual_outputs = entry["output"]
        
        predicted_outputs = predict_single(input_vec, model, input_mean, input_std, output_mean, output_std)
        
        # Check each output
        sample_correct = True
        sample_correct_count = 0
        
        print(f"\n-- Sample {i+1}:")
        for j in range(len(actual_outputs)):
            error = abs(predicted_outputs[j] - actual_outputs[j])
            is_correct = error <= tolerance
            
            if is_correct:
                sample_correct_count += 1
                total_correct_outputs += 1
            else:
                sample_correct = False
            
            total_outputs += 1
            status = "✅" if is_correct else "❌"
            print(f"   Output {j+1:2d}: {status} Pred={predicted_outputs[j]:6.1f}, Actual={actual_outputs[j]:4.0f}, Error={error:5.1f}")
        
        if sample_correct:
            correct_samples += 1
        
        print(f"   → Sample accuracy: {sample_correct_count}/{len(actual_outputs)} outputs correct")
    
    sample_accuracy = correct_samples / len(dataset)
    output_accuracy = total_correct_outputs / total_outputs
    
    print("=" * 80)
    print(f"-- RESULTS:")
    print(f"   Complete samples correct: {correct_samples}/{len(dataset)} ({sample_accuracy:.1%})")
    print(f"   Individual outputs correct: {total_correct_outputs}/{total_outputs} ({output_accuracy:.1%})")
    print(f"   Tolerance: ±{tolerance}")
    
    if total_correct_outputs >= 5:
        print("-- SUCCESS! Model achieved the target of 5+ correct individual outputs!")
    else:
        print("⚠️  Model needs improvement to reach 5+ correct individual outputs.")
    
    return output_accuracy, total_correct_outputs

def main():
    parser = argparse.ArgumentParser(description="Make predictions using trained multi-output model")
    parser.add_argument("--input", type=str, help="Input vector as JSON string with 40 values")
    parser.add_argument("--test_dataset", action="store_true", help="Test model on the entire dataset")
    args = parser.parse_args()
    
    if args.test_dataset:
        test_on_dataset()
    elif args.input:
        try:
            # Parse input vector
            input_vector = json.loads(args.input)
            
            if len(input_vector) != 40:
                print(f"❌ Input must have exactly 40 values, got {len(input_vector)}")
                return
            
            # Load model and stats
            model, input_mean, input_std, output_mean, output_std = load_model_and_stats()
            
            # Make prediction
            predictions = predict_single(input_vector, model, input_mean, input_std, output_mean, output_std)
            
            print(f"-- Input (first 10): {input_vector[:10]}")
            print(f"-- Predictions (12 outputs):")
            for i, pred in enumerate(predictions):
                print(f"   Output {i+1:2d}: {pred:6.1f}")
            
        except json.JSONDecodeError:
            print("❌ Invalid input format. Please provide input as JSON array.")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("❌ Please provide either --input or --test_dataset flag")
        print("Examples:")
        print("  python predict.py --input '[12,7,9,21,18,20,13,21,13,14,14,7,0,1,18,15,0,14,10,5,15,18,3,0,1,2,4,4,10,18,5,3,21,5,0,6,1,9,4,2]'")
        print("  python predict.py --test_dataset")

if __name__ == "__main__":
    main()
