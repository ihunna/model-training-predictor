import json
import numpy as np
import random

def generate_realistic_dataset(num_samples=100):
    """
    Generate a dataset with 40 input features and 12 output values.
    This creates patterns that a neural network can learn.
    """
    
    np.random.seed(42)
    random.seed(42)
    
    dataset = []
    
    for i in range(num_samples):
        inputs = [random.randint(0, 21) for _ in range(40)]
        
        outputs = []
        
        for output_idx in range(12):
            start_idx = (output_idx * 3) % 40
            end_idx = min(start_idx + 8, 40)
            
            if output_idx < 4:
                weight_sum = (
                    sum(inputs[0:10]) * 8 +
                    sum(inputs[10:20]) * 5 +
                    inputs[output_idx] * 20
                )
            elif output_idx < 8:
         
                weight_sum = (
                    sum(inputs[15:25]) * 7 +
                    sum(inputs[25:35]) * 4 +
                    inputs[output_idx + 10] * 15
                )
            else:
        
                weight_sum = (
                    sum(inputs[20:30]) * 6 +
                    sum(inputs[30:40]) * 8 +
                    inputs[output_idx + 20] * 12
                )
            
        
            max_in_range = max(inputs[start_idx:end_idx])
            min_in_range = min(inputs[start_idx:end_idx])
            
           
            interaction = inputs[output_idx] * inputs[(output_idx + 10) % 40]
            
    
            base_output = (
                weight_sum * 0.4 +
                (max_in_range - min_in_range) * 25 +
                interaction * 1.5 +
                (max_in_range ** 1.2) * 8 +
                output_idx * 50
            )
            

            noise = np.random.normal(0, base_output * 0.03)
            final_output = max(0, min(2047, base_output + noise))
            
            outputs.append(int(round(final_output)))
        
        dataset.append({
            "input": inputs,
            "output": outputs
        })
    
    return dataset

if __name__ == "__main__":
    data = generate_realistic_dataset(120)
    
    with open("dataset.json", "w") as f:
        json.dump(data, f, indent=2)
    
    inputs = np.array([d["input"] for d in data])
    outputs = np.array([d["output"] for d in data])
    
    print(f"✅ Generated {len(data)} samples")
    print(f"-- Input shape: {inputs.shape} (range: {inputs.min()}-{inputs.max()})")
    print(f"-- Output shape: {outputs.shape} (range: {outputs.min()}-{outputs.max()})")
    print(f"-- Output means: {outputs.mean(axis=0).round(1)}")
    print(f"-- Output stds: {outputs.std(axis=0).round(1)}")
    
    input_sums = inputs.sum(axis=1)
    correlations = [np.corrcoef(input_sums, outputs[:, i])[0, 1] for i in range(12)]
    print(f"-- Input sum correlations: {[f'{c:.3f}' for c in correlations]}")
    
    print("\n-- Sample entry:")
    print(f"   Input (first 10): {data[0]['input'][:10]}")
    print(f"   Output: {data[0]['output']}")
