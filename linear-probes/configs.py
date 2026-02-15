from typing import Dict, Any

config_gpt2: Dict[str, Any] = {
    "short_name": "gpt2",
    "model_name": "openai-community/gpt2",
    "activation_size": 768,
    "seq_len": 1024,
    "hook_component": "transformer.h[6].mlp",
    "test_split": 0.2,
    "batch_size": 32,
    "learning_rate": 0.001,
    }

config_phi4: Dict[str, Any] = {
    "short_name": "phi4",
    "model_name": "microsoft/phi-4",
    "activation_size": 5120,
    "seq_len": 16384,
    # "hook_component": "model.layers[20].mlp",
    "hook_component": "model.model.layers[20]",
    "layer": 20,
    "num_layers": 40,
    "test_split": 0.2,
    "batch_size": 32,
    "learning_rate": 0.001,
    "expt_name": "2025-02-01_phi_phi_100_games_v3",
    "probe_training_epochs": 4,
    "probe_training_batch_size": 32,
    "probe_training_learning_rate": 0.001,
    "probe_training_num_tokens": 10,
    "probe_training_chunk_idx": 0,
}

config_llama3: Dict[str, Any] = {
    "short_name": "llama3",
    "model_name": "meta-llama/Llama-3.3-70B-Instruct",
    "activation_size": 8192,
    "seq_len": 131072,
    "hook_component": "model.layers[40].mlp",
    "test_split": 0.2,
    "batch_size": 32,
    "learning_rate": 0.001,
    "expt_name": "2025-02-01_llama_llama_100_games_v3",
    "probe_training_epochs": 4,
    "probe_training_batch_size": 32,
    "probe_training_learning_rate": 0.001,
    "probe_training_num_tokens": 10,
    "probe_training_chunk_idx": 0,
}