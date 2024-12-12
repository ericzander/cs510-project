import os
import time

import torch
from torch_geometric.datasets import HeterophilousGraphDataset

import pandas as pd

from models.gat import GAT
from models.gcn import GCN

from utils.trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Define the CSV file path
    csv_file = "experiment_results.csv"

    # Initialize saved results
    if os.path.exists(csv_file):
        completed_tests = pd.read_csv(csv_file)[["model_type", "configuration", "dataset"]]
    if not os.path.exists(csv_file):
        pd.DataFrame(columns=[
            "model_type", "configuration", "dataset", 
            "accuracy", "precision", "recall", "f1", "roc", 
            "runtime", "notes"
        ]).to_csv(csv_file, index=False)
        completed_tests = pd.DataFrame(columns=["model_type", "configuration", "dataset"])

    # Define test parameters
    model_types = ["GCN", "GAT"]
    configurations = [(512,), (512, 256), (512, 256, 128)]
    datasets = ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]

    # Iterate through all combinations
    for data_name in datasets:
        dataset = HeterophilousGraphDataset(root='data/', name=data_name).to(device)

        for model_type in model_types:
            for config in configurations:
                print(f"Running '{data_name}' with '{model_type}{config}'...")

                if _already_completed(completed_tests, model_type, config, data_name):
                    print("Results already saved! Skipping...")
                    continue

                t0 = time.time()
                
                # Simulate running the test
                try:
                    model = _init_model(model_type, config, dataset).to(device)

                    trainer = Trainer(model, dataset)

                    print(f"Number of parameters: {trainer.get_num_params():,}")
                    trainer.run(1000, print_freq=20, timeout=900)  # 15 min timeout

                    metrics = trainer.get_metrics()
                    trainer.save_weights()

                    print("Finished! Metrics:")
                    print(metrics)

                    print("=" * 40)

                    notes = "Success"
                except Exception as e:
                    metrics = {"accuracy": None, "precision": None, "recall": None, "f1": None, "roc": None}
                    notes = f"Error: {e}"
                
                t1 = time.time()

                # Prepare the result row
                result = {
                    "model_type": model_type,
                    "configuration": str(config),
                    "dataset": data_name,
                    **metrics,
                    "runtime": t1 - t0,
                    "notes": notes
                }

                # Append to the CSV file incrementally
                pd.DataFrame([result]).to_csv(csv_file, mode='a', header=False, index=False)

        del dataset
        torch.cuda.empty_cache()

    # Load results from CSV
    results_df = pd.read_csv(csv_file)

    summary = results_df.groupby("model_type")[["accuracy", "precision", "recall", "f1", "roc"]].mean()
    print(summary)

def _init_model(model_type, config, dataset):
    mlp_channels_list = [128, 128]
    num_heads = 4

    if model_type == "GCN":
        model = GCN(
            in_channels=dataset.num_node_features,
            conv_channels=config,
            mlp_channels=mlp_channels_list,
            out_channels=dataset.num_classes,
        )
    elif model_type == "GAT":
        model = GAT(
            in_channels=dataset.num_node_features,
            conv_channels=config,
            mlp_channels=mlp_channels_list,
            out_channels=dataset.num_classes,
            heads=num_heads,
            concat=True
        )

    return model

def _already_completed(completed_tests, model_type, config, dataset):
    config_str = str(config)
    return not completed_tests.query(
        f"model_type == '{model_type}' and configuration == '{config_str}' and dataset == '{dataset}'"
    ).empty

if __name__ == "__main__":
    main()
