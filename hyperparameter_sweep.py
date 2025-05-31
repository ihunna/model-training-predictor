import itertools
import torch
from train import train_model

def hyperparameter_sweep(dataset_path, device):
    # Hyperparameter space
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [16, 32, 64]
    dropouts = [0.0, 0.1, 0.2]
    hidden_layer_configs = [
        [128, 64, 32],
        [256, 128, 64, 32],
        [64, 32, 16]
    ]
    weight_decays = [1e-5, 1e-6]

    best_config = None
    best_mse = float('inf')
    best_results = None

    total_runs = len(learning_rates) * len(batch_sizes) * len(dropouts) * len(hidden_layer_configs) * len(weight_decays)
    run_counter = 1

    for lr, bs, dropout, hidden_dims, wd in itertools.product(learning_rates, batch_sizes, dropouts, hidden_layer_configs, weight_decays):
        print(f"\nRun {run_counter}/{total_runs}: LR={lr}, Batch Size={bs}, Dropout={dropout}, Hidden={hidden_dims}, WD={wd}")
        run_counter += 1

        results = train_model(
            dataset_path=dataset_path,
            device=device,
            max_samples=None,
            lr=lr,
            batch_size=bs,
            dropout=dropout,
            hidden_dims=hidden_dims,
            weight_decay=wd,
            max_epochs=50,    # limit epochs for sweep to speed up
            use_progress_bar=False
        )

        print(f" --> MSE: {results['mse']:.4f}, MAE: {results['mae']:.4f}, Accuracy: {results['accuracy']:.4f}")

        if results['mse'] < best_mse:
            best_mse = results['mse']
            best_config = {
                'lr': lr,
                'batch_size': bs,
                'dropout': dropout,
                'hidden_dims': hidden_dims,
                'weight_decay': wd
            }
            best_results = results

    print("\nBest hyperparameters found:")
    print(best_config)
    print(f"Best MSE: {best_mse:.4f}")

    # Save best model (already saved in train_model), save config
    import json
    with open('models/best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)

    return best_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument('--dataset', required=True, help="Path to JSON dataset")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    best_results = hyperparameter_sweep(args.dataset, device)
    print("Sweep complete.")
