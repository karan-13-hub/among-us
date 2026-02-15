# train and evaluate logistic probes on all datasets (and get results for the paper)
# NOTE: this assumes all activations have already been cached (run cache_activations.py first)

################### SETUP

import os
import sys
import pickle
from typing import Dict, Any, List
from configs import config_phi4, config_gpt2, config_llama3

sys.path.append(os.path.dirname(os.path.abspath('.')))
sys.path.append('.')

from probe_datasets import (
    TruthfulQADataset,
    DishonestQADataset, 
    AmongUsDataset,
    RepEngDataset,
)
from probes import LinearProbe

datasets: List[str] = [
    "TruthfulQADataset",
    "DishonestQADataset",
    "AmongUsDataset",
    "RepEngDataset",
]

config = config_phi4
model, tokenizer, device = None, None, 'cpu'
amongus_expt_name: str = "2025-02-01_phi_phi_100_games_v3"

################### TRAINING PROBES

for dataset_name in datasets:
    print(f"Loading {dataset_name}...")
    dataset = eval(f"{dataset_name}")(
        config,
        model=model,
        tokenizer=tokenizer, 
        device=device, 
        test_split=0.2,
        expt_name=amongus_expt_name
        )
    train_loader = dataset.get_train(
        batch_size=config["probe_training_batch_size"],
        num_tokens=config["probe_training_num_tokens"],
        chunk_idx=config["probe_training_chunk_idx"],
    )
    probe = LinearProbe(
        input_dim=dataset.activation_size, 
        device=device, 
        lr=config["probe_training_learning_rate"]
        )
    print(f'Training probe on {len(train_loader)} batches and {len(train_loader.dataset)} samples.')
    probe.fit(train_loader, epochs=config["probe_training_epochs"])

    checkpoint_path = f'checkpoints/{dataset_name}_probe_{config["short_name"]}.pkl'
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(probe, f)
        print(f"Probe saved to {checkpoint_path}")

print(f"Probes trained and saved for all datasets.")