import sys
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as t
import importlib
import os

sys.path.append(os.path.dirname(os.path.abspath('.')))

import datasets, plots, configs, evaluate_utils
for module in [datasets, plots, configs, evaluate_utils]:
    importlib.reload(module)

from probe_datasets import AmongUsDataset, TruthfulQADataset, DishonestQADataset, RepEngDataset, RolePlayingDataset, ApolloProbeDataset
from configs import config_phi4, config_gpt2, config_llama3

def main(dataset_name: str):
    config = config_llama3
    model_name = config["model_name"]
    load_models = True

    if load_models:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", torch_dtype=t.bfloat16)
        device = model.device
    else:
        model, tokenizer, device = None, None, 'cpu'

    # Create dataset based on the provided name
    if dataset_name == "AmongUsDataset":
        dataset = eval(dataset_name)(config, model=model, tokenizer=tokenizer, device=device, 
                                    expt_name=config['expt_name'], test_split=1.0)
    else:
        dataset = eval(dataset_name)(config, model=model, tokenizer=tokenizer, device=device, test_split=1.0)
    
    eval(f"model.{config['hook_component']}").register_forward_hook(dataset.activation_cache.hook_fn)
    num_tokens = 5 if dataset_name == "ApolloProbeDataset" else None
    dataset.populate_dataset(force_redo=True, just_load=False, max_rows=1500, seq_len=config["seq_len"], num_tokens=num_tokens, chunk_size=500)
    print(f'Done! Cached activations for {dataset.num_total_chunks} chunks.')

if __name__ == "__main__":
    # datasets_to_cache = ["AmongUsDataset", "TruthfulQADataset", "DishonestQADataset", "RepEngDataset", "RolePlayingDataset"]
    datasets_to_cache = ["RolePlayingDataset"]
    for dataset_name in datasets_to_cache:
        print(f"Caching activations for {dataset_name}...")
        main(dataset_name)
    print("All dataset activations cached.")