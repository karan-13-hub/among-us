from torch.utils.data import Dataset, DataLoader, random_split
import torch as t
from typing import List, Tuple, Dict, Any
import pickle
import pandas as pd
import os
import random
import sys
from probe_utils import free_unused_memory
import yaml

def phi4_format(system, user, assistant):
    system_part = f"<|im_start|>system<|im_sep|>{system}<|im_end|>" if system else ""
    user_part = f"<|im_start|>user<|im_sep|>{user}<|im_end|>" if user else ""
    assistant_part = f"<|im_start|>assistant<|im_sep|>{assistant}" if assistant else "<|im_start|>assistant<|im_sep|>"
    return f"{system_part}{user_part}{assistant_part}"

def llama3_format(system, user, assistant):
    system_part = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|>" if system else ""
    user_part = f"<|start_header_id|>user<|end_header_id|>{user}<|eot_id|>" if user else ""
    assistant_part = f"<|start_header_id|>assistant<|end_header_id|>{assistant}" if assistant else "<|start_header_id|>assistant<|end_header_id|>"
    return f"{system_part}{user_part}{assistant_part}"

class ActivationCache:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.activations = []
        self.handles = []

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output
        self.activations.append(output_tensor.detach().cpu().float().numpy())

    def register_hook(self, layer):
        handle = layer.register_forward_hook(self.hook_fn)
        self.handles.append(handle)
        
    def clear_activations(self):
        self.activations = []

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

class ActivationDataset(Dataset):
    def __init__(self, test_split, name: str = "", model=None, tokenizer=None, device=None, activation_size: int = 768, **kwargs):
        """
        Initialize empty dataset with configurable test split ratio
        
        Args:
            test_split (float): Proportion of data to use for testing (0-1)
            name (str): Name of the dataset
            model: Model to use for generating activations
            tokenizer: Tokenizer to use for processing text
            device: Device to run model on
            activation_size (int): Size of activation vectors
            **kwargs: Additional keyword arguments that may be needed by specific dataset implementations
        """
        self.test_split = test_split
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.activation_size = activation_size
        self.activation_cache = ActivationCache(model, tokenizer, device)
        self.num_total_chunks = 0
        if model is not None:
            self.activation_cache.remove_hooks()

    def get_chunk_path(self, chunk_idx: int) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), self.activations_dir, f"chunk_{chunk_idx}.pkl")

    def save_chunk(self, chunk_data: List[Tuple[List[t.Tensor], int]], chunk_idx: int):
        full_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.activations_dir)
        os.makedirs(full_dir_path, exist_ok=True)
        chunk_path = self.get_chunk_path(chunk_idx)
        with open(chunk_path, 'wb') as f:
            pickle.dump(chunk_data, f)
            print(f"Saved chunk {chunk_idx} to {chunk_path}")

    def load_chunk(self, chunk_idx: int) -> List[Tuple[List[t.Tensor], int]]:
        chunk_path = self.get_chunk_path(chunk_idx)
        if not os.path.exists(chunk_path):
            raise ValueError(f"Chunk {chunk_idx} does not exist at {chunk_path}")
        with open(chunk_path, 'rb') as f:
            return pickle.load(f)

    def get_train(self, chunk_idx: int = 0, batch_size: int = 32, shuffle: bool = True,
                    num_workers: int = 0, pin_memory: bool = True, num_tokens: int = None,  
                    keep_frac: float = 1.0, get_val: bool = False) -> DataLoader:
        """
        Get train DataLoader for a specific chunk, taking specified number of tokens from each prompt
        
        Args:
            chunk_idx: Which chunk to load
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data
            num_workers: Number of workers for DataLoader
            pin_memory: Whether to pin memory for DataLoader
            num_tokens: Number of tokens to take from end of each prompt. If None, take all tokens.
            keep_frac: Fraction of data to keep (for subsampling)
            get_val: If True, also return a validation DataLoader. If False, only return train DataLoader.
        """
        chunk_data = self.load_chunk(chunk_idx)
        train_size = min(int(len(chunk_data) * (1 - self.test_split)), len(chunk_data) - 1)
        train_data = chunk_data[:train_size]
        
        # Take specified number of tokens from each prompt
        flat_data = []
        for acts, label in train_data:
            if random.random() > keep_frac:
                continue
            acts_to_use = acts[-num_tokens:] if num_tokens else acts
            for act in acts_to_use:
                flat_data.append((act, label))
                
        train_loader = DataLoader(
            flat_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        if not get_val:
            return train_loader
            
        # Create validation set from remaining data
        val_size = min(max(len(flat_data) // 2, 1), len(chunk_data) - train_size)
        if val_size == 0:
            return train_loader, None
            
        val_data = chunk_data[train_size:train_size + val_size]
        print(f"Validation size: {val_size}")
        val_flat_data = []
        
        # Keep trying until we get at least one sample
        max_attempts = 10
        for attempt in range(max_attempts):
            val_flat_data = []
            for acts, label in val_data:
                acts_to_use = acts[-num_tokens:] if num_tokens else acts
                for act in acts_to_use:
                    val_flat_data.append((act, label))
            if val_flat_data:  # If we got at least one sample, break
                break
            if attempt == max_attempts - 1:  # If we've tried max_attempts times and still have no samples
                return train_loader, None
                
        val_loader = DataLoader(
            val_flat_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader

    def get_test_acts(self, chunk_idx: int = 0, batch_size: int = 32, shuffle: bool = False,
                    num_workers: int = 0, pin_memory: bool = True) -> DataLoader:
        """
        Get test DataLoader for a specific chunk keeping activations for each prompt together
        """
        chunk_data = self.load_chunk(chunk_idx)
        test_size = int(len(chunk_data) * self.test_split)
        test_data = chunk_data[-test_size:]
        
        return test_data
    
    def get_train_data_stats(self, chunk_idx: int = 0) -> dict:
        """
        Get basic statistics about a specific chunk
        """
        chunk_data = self.load_chunk(chunk_idx)
        if not chunk_data:
            return {"total_samples": 0, "class_distribution": {}}
            
        train_size = int(len(chunk_data) * (1 - self.test_split))
        train_data = chunk_data[:train_size]
        labels = [y for _, y in train_data]
        unique, counts = t.tensor(labels).unique(return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist()))
        
        return {
            "total_samples": train_size,
            "class_distribution": class_dist
        }

class TruthfulQADataset(ActivationDataset):
    def __init__(self, config: Dict[str, Any]=None, model=None, tokenizer=None, device=None, test_split=None, **kwargs):
        super().__init__(test_split, "TruthfulQA", model, tokenizer, device, config["activation_size"])
        self.data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), './data/TruthfulQA/TruthfulQA.csv')
        self.tqa_df = pd.read_csv(self.data_path)
        self.activations_dir: str = f'data/{self.name}_{config["short_name"]}_acts_{config["layer"]}/'
        self.num_total_chunks = 1  # TruthfulQA uses single chunk
        self.format = eval(config["short_name"] + "_format")

    def row_to_prompts(self, row: pd.Series) -> str:
        question = row['Question']
        best_answer = row['Best Answer']
        best_incorrect_answer = row['Best Incorrect Answer']
        correct_qa = self.format(None, question, best_answer)
        incorrect_qa = self.format(None, question, best_incorrect_answer)
        return correct_qa, incorrect_qa

    def process_row(self, row, num_tokens: int = 5, seq_len: int = 1024):
        correct_prompt, incorrect_prompt = self.row_to_prompts(row)
        correct_inputs = self.tokenizer(correct_prompt, return_tensors="pt", max_length=seq_len, truncation=True).to(self.device)
        incorrect_inputs = self.tokenizer(incorrect_prompt, return_tensors="pt", max_length=seq_len, truncation=True).to(self.device)
        chunk_data = []
        with t.no_grad():
            self.activation_cache.clear_activations()
            self.model.forward(correct_inputs['input_ids'], attention_mask=correct_inputs['attention_mask'], pad_token_id=self.tokenizer.eos_token_id)
            correct_activations = self.activation_cache.activations[0][0]
            correct_acts = [correct_activations[i] for i in range(-num_tokens, 0)] if num_tokens else [correct_activations[i] for i in range(len(correct_activations))]
            chunk_data.append((correct_acts, 1))
                
            self.activation_cache.clear_activations()
            self.model.forward(incorrect_inputs['input_ids'], attention_mask=incorrect_inputs['attention_mask'], pad_token_id=self.tokenizer.eos_token_id)
            incorrect_activations = self.activation_cache.activations[0][0]
            incorrect_acts = [incorrect_activations[i] for i in range(-num_tokens, 0)] if num_tokens else [incorrect_activations[i] for i in range(len(incorrect_activations))]
            chunk_data.append((incorrect_acts, 0))
        return chunk_data

    def populate_dataset(self, force_redo: bool = False, num_tokens: int = 5, max_rows: int = 0, seq_len: int = 1024, chunk_size=None, **kwargs):
        if os.path.exists(self.get_chunk_path(0)) and not force_redo:
            print(f"Loading existing chunk from {self.get_chunk_path(0)}")
            return
            
        chunk_data = []
        for idx, row in self.tqa_df.iterrows():
            chunk_data.extend(self.process_row(row, num_tokens, seq_len))
            if idx % (len(self.tqa_df) // 10) == 0:
                print(f"Processed {idx} rows of {len(self.tqa_df)}")
        self.save_chunk(chunk_data, 0)

class DishonestQADataset(ActivationDataset):
    def __init__(self, config: Dict[str, Any]=None, model=None, tokenizer=None, device=None, test_split=None, **kwargs):
        super().__init__(test_split, "DishonestQA", model, tokenizer, device, config["activation_size"])
        self.data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), './data/TruthfulQA/TruthfulQA.csv')
        self.tqa_df = pd.read_csv(self.data_path)
        self.activations_dir: str = f'data/{self.name}_{config["short_name"]}_acts_{config["layer"]}/'
        self.num_total_chunks = 1  # DishonestQA uses single chunk
        self.format = eval(config["short_name"] + "_format")

    def row_to_prompts(self, row: pd.Series) -> str:
        question = row['Question']
        correct_q = self.format("Pretend you're an honest assistant.", question, None)
        incorrect_q = self.format("Pretend you're a dishonest assistant.", question, None)
        return correct_q, incorrect_q

    def process_row(self, row, num_tokens: int = 5, seq_len: int = 1024):
        correct_prompt, incorrect_prompt = self.row_to_prompts(row)
        correct_inputs = self.tokenizer(correct_prompt, return_tensors="pt", max_length=seq_len, truncation=True).to(self.device)
        incorrect_inputs = self.tokenizer(incorrect_prompt, return_tensors="pt", max_length=seq_len, truncation=True).to(self.device)
        chunk_data = []
        with t.no_grad():
            self.activation_cache.clear_activations()
            out_tokens = self.model.generate(correct_inputs['input_ids'], attention_mask=correct_inputs['attention_mask'], pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=10)
            out_token_ids = out_tokens.cpu().tolist()[0]
            full_output_str = self.tokenizer.decode(out_token_ids)
            self.activation_cache.clear_activations()
            # run a single forward pass to get activations
            full_output_inputs = self.tokenizer(full_output_str, return_tensors="pt", max_length=seq_len, truncation=True).to(self.device)
            self.model.forward(full_output_inputs['input_ids'], attention_mask=full_output_inputs['attention_mask'], pad_token_id=self.tokenizer.eos_token_id)
            correct_activations = self.activation_cache.activations[0][0]
            correct_acts = [correct_activations[i] for i in range(-num_tokens, 0)] if num_tokens else [correct_activations[i] for i in range(len(correct_activations))]
            chunk_data.append((correct_acts, 1))
                
            self.activation_cache.clear_activations()
            out_tokens = self.model.generate(incorrect_inputs['input_ids'], attention_mask=incorrect_inputs['attention_mask'], pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=10)
            out_token_ids = out_tokens.cpu().tolist()[0]
            full_output_str = self.tokenizer.decode(out_token_ids)
            self.activation_cache.clear_activations()
            # run a single forward pass to get activations
            full_output_inputs = self.tokenizer(full_output_str, return_tensors="pt", max_length=seq_len, truncation=True).to(self.device)
            self.model.forward(full_output_inputs['input_ids'], attention_mask=full_output_inputs['attention_mask'], pad_token_id=self.tokenizer.eos_token_id)
            incorrect_activations = self.activation_cache.activations[0][0]
            incorrect_acts = [incorrect_activations[i] for i in range(-num_tokens, 0)] if num_tokens else [incorrect_activations[i] for i in range(len(incorrect_activations))]
            chunk_data.append((incorrect_acts, 0))
        return chunk_data

    def populate_dataset(self, force_redo: bool = False, num_tokens: int = 5, max_rows: int = 0, seq_len: int = 1024, chunk_size=None, **kwargs):
        if os.path.exists(self.get_chunk_path(0)) and not force_redo:
            print(f"Loading existing chunk from {self.get_chunk_path(0)}")
            return
            
        chunk_data = []
        for idx, row in self.tqa_df.iterrows():
            chunk_data.extend(self.process_row(row, num_tokens, seq_len))
            if idx % (len(self.tqa_df) // 10) == 0:
                print(f"Processed {idx} rows of {len(self.tqa_df)}")
        self.save_chunk(chunk_data, 0)

class AmongUsDataset(ActivationDataset):
    def __init__(
            self, 
            config: Dict[str, Any], 
            model=None, 
            tokenizer=None, 
            device=None, 
            raw_path: str = "../expt-logs/",
            expt_name: str = None,
            test_split: float = None,
            **kwargs
            ):
        super().__init__(test_split, "AmongUs", model, tokenizer, device, config["activation_size"])
        self.name: str = "AmongUs"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.agent_logs_path: str = os.path.join(base_dir, raw_path, expt_name + "/agent-logs-compact.json")
        sys.path.append(os.path.join(base_dir, ".."))
        from utils import load_agent_logs_df
        self.agent_logs_df = load_agent_logs_df(self.agent_logs_path)
        self.activations_dir: str = f'data/{self.name}_{config["short_name"]}_acts_{config["layer"]}/'
        # load number of chunks from existing directory
        self.num_total_chunks = 0
        self.format = eval(config["short_name"] + "_format")
        while os.path.exists(self.get_chunk_path(self.num_total_chunks)):
            self.num_total_chunks += 1

    def agent_logs_row_to_full_prompt(self, row: pd.Series) -> str:
        system_prompt = row["interaction.system_prompt"]
        summarization = row["interaction.prompt.Summarization"]
        processed_memory = row["interaction.prompt.Memory"]
        phase = row["interaction.prompt.Phase"]
        all_info = row["interaction.prompt.All Info"]

        user_prompt = f"Summarization: {summarization}\n\n{all_info}\n\nMemory: {processed_memory}\
                        \n\nPhase: {phase}. Return your output."

        assistant_prompt = row["interaction.full_response"]

        formatted_prompt = self.format(system_prompt if system_prompt else None, user_prompt, assistant_prompt)
        return formatted_prompt

    def process_row(self, row, num_tokens, seq_len):
        free_unused_memory()
        phi_format_prompt = self.agent_logs_row_to_full_prompt(row)
        inputs = self.tokenizer(phi_format_prompt, return_tensors="pt", max_length=seq_len, truncation=True).to(self.device)
        with t.no_grad():
            self.activation_cache.clear_activations()
            self.model.forward(inputs['input_ids'], attention_mask=inputs['attention_mask'], pad_token_id=self.tokenizer.eos_token_id)
            if not self.activation_cache.activations:
                raise ValueError("No activations found. Ensure the model and activation cache are set up correctly.")
            activations = self.activation_cache.activations[0][0]
            if num_tokens:
                if len(activations) < num_tokens:
                    raise ValueError(f"Not enough activations to extract {num_tokens} tokens. Available: {len(activations)}")
                acts = [activations[i] for i in range(-num_tokens, 0)]
            else:
                acts = [activations[i] for i in range(len(activations))]
            label = 0 if row["player.identity"] == "Impostor" else 1
            return acts, label

    def populate_dataset(
        self, 
        force_redo: bool = False,
        num_tokens: int = 5,
        max_rows: int = 0,
        seq_len: int = 1024,
        chunk_size: int = 100,
        just_load: bool = True,
        ):
        """Get activations row by row and save in chunks"""
        if just_load:
            chunk_idx = 0
            while os.path.exists(self.get_chunk_path(chunk_idx)):
                chunk_idx += 1
            self.num_total_chunks = chunk_idx
            print(f"Loaded {self.num_total_chunks} existing chunks")
            return

        if force_redo:
            full_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.activations_dir)
            if os.path.exists(full_dir_path):
                import shutil
                shutil.rmtree(full_dir_path)
            
        full_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.activations_dir)
        os.makedirs(full_dir_path, exist_ok=True)
        
        # Find last processed chunk
        chunk_idx = 0
        while os.path.exists(self.get_chunk_path(chunk_idx)):
            chunk_idx += 1
        print(f"Starting from chunk {chunk_idx}")
        
        # Calculate rows to process
        processed_rows = chunk_idx * chunk_size
        rows_to_cache = max_rows if max_rows > 0 else len(self.agent_logs_df)
        
        # Process remaining rows
        current_chunk = []
        for idx in range(processed_rows, rows_to_cache):
            if idx >= len(self.agent_logs_df):
                break
                
            # Get activations for this row
            row = self.agent_logs_df.iloc[idx]
            acts, label = self.process_row(row, num_tokens, seq_len)
            current_chunk.append((acts, label))
            free_unused_memory()
                
            # Save chunk when it reaches chunk_size
            if len(current_chunk) >= chunk_size:
                self.save_chunk(current_chunk, chunk_idx)
                chunk_idx += 1
                current_chunk = []
                free_unused_memory()
                
            if (idx - processed_rows) % (chunk_size // 10) == 0:
                print(f"Processed {idx} rows of {rows_to_cache}")
        
        # Save final partial chunk
        if current_chunk:
            self.save_chunk(current_chunk, chunk_idx)
            
        self.num_total_chunks = chunk_idx + 1
        print(f"Finished processing {rows_to_cache} rows into {self.num_total_chunks} chunks")

class RolePlayingDataset(ActivationDataset):
    def __init__(self, config: Dict[str, Any]=None, model=None, tokenizer=None, device=None, test_split=None, **kwargs):
        super().__init__(test_split, "Roleplaying", model, tokenizer, device, config["activation_size"])
        self.data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), './data/Roleplaying/phi4_rollouts.csv')
        self.df = pd.read_csv(self.data_path)
        self.activations_dir: str = f'data/{self.name}_{config["short_name"]}_acts_{config["layer"]}/'
        self.num_total_chunks = 1  # Roleplaying uses single chunk
        self.format = eval(config["short_name"] + "_format")

    def row_to_prompts(self, row: pd.Series) -> str:
        system: str = row['scenario']
        question: str = row['question']
        rollout: str = row['answer_prefix'] + row['phi4_rollout']
        formatted = self.format(system, question, rollout)
        return formatted, row['llm_eval_gt']

    def process_row(self, row, num_tokens: int = 5, seq_len: int = 1024):
        free_unused_memory()
        prompt, label = self.row_to_prompts(row)
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=seq_len, truncation=True).to(self.device)
        chunk_data = []
        with t.no_grad():
            self.activation_cache.clear_activations()
            self.model.forward(inputs['input_ids'], attention_mask=inputs['attention_mask'], pad_token_id=self.tokenizer.eos_token_id)
            activations = self.activation_cache.activations[0][0]
            acts = [activations[i] for i in range(-num_tokens, 0)] if num_tokens else [activations[i] for i in range(len(activations))]
            chunk_data.append((acts, label))
        return chunk_data
    
    def populate_dataset(self, force_redo: bool = False, num_tokens: int = 5, max_rows: int = 0, seq_len: int = 1024, chunk_size=None, **kwargs):
        if os.path.exists(self.get_chunk_path(0)) and not force_redo:
            print(f"Loading existing chunk from {self.get_chunk_path(0)}")
            return
        chunk_data = []
        for idx, row in self.df.iterrows():
            chunk_data.extend(self.process_row(row, num_tokens, seq_len))
            if idx % (len(self.df) // 10) == 0:
                print(f"Processed {idx} rows of {len(self.df)}")
        self.save_chunk(chunk_data, 0)
        free_unused_memory()
        
class RepEngDataset(ActivationDataset):
    def __init__(self, config: Dict[str, Any]=None, model=None, tokenizer=None, device=None, test_split=None, **kwargs):
        super().__init__(test_split, "RepEng", model, tokenizer, device, config["activation_size"])
        self.data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), './data/RepE/true_false_facts.csv')
        self.df = pd.read_csv(self.data_path)
        self.activations_dir: str = f'data/{self.name}_{config["short_name"]}_acts_{config["layer"]}/'
        self.num_total_chunks = 1 # RepEng uses single chunk
        self.format = eval(config["short_name"] + "_format")

    def row_to_prompts(self, row: pd.Series) -> str:
        statement = row['statement']
        system: str = "You are an AI assistant."
        label: str = "honest" if row['label'] == 1 else "dishonest"
        user: str = f"Pretend you're a {label} person making statements about the world."
        formatted = self.format(system, user, statement)
        return formatted, row['label']

    def process_row(self, row, num_tokens: int = 5, seq_len: int = 1024):
        free_unused_memory()
        phi_format_prompt, label = self.row_to_prompts(row)
        inputs = self.tokenizer(phi_format_prompt, return_tensors="pt", max_length=seq_len, truncation=True).to(self.device)
        with t.no_grad():
            self.activation_cache.clear_activations()
            self.model.forward(inputs['input_ids'], attention_mask=inputs['attention_mask'], pad_token_id=self.tokenizer.eos_token_id)
            activations = self.activation_cache.activations[0][0]
            acts = [activations[i] for i in range(-num_tokens, 0)] if num_tokens else [activations[i] for i in range(len(activations))]
            return (acts, label)

    def populate_dataset(self, force_redo: bool = False, num_tokens: int = 5, max_rows: int = 0, seq_len: int = 1024, chunk_size=None, **kwargs):
        if os.path.exists(self.get_chunk_path(0)) and not force_redo:
            print(f"Loading existing chunk from {self.get_chunk_path(0)}")
            return
        chunk_data = []
        for idx, row in self.df.iterrows():
            chunk_data.append(self.process_row(row, num_tokens, seq_len))
            if idx % (len(self.df) // 10) == 0:
                print(f"Processed {idx} rows of {len(self.df)}")
        self.save_chunk(chunk_data, 0)
        free_unused_memory()
        self.num_total_chunks = 1

class ApolloProbeDataset(ActivationDataset):
    def __init__(self, config: Dict[str, Any]=None, model=None, tokenizer=None, device=None, test_split=None, **kwargs):
        super().__init__(test_split, "ApolloProbe", model, tokenizer, device, config["activation_size"])
        self.data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), './data/ApolloProbe/common_claim_true_false.csv')
        self.df = pd.read_csv(self.data_path)
        self.activations_dir: str = f'data/{self.name}_{config["short_name"]}_acts_{config["layer"]}/'
        self.num_total_chunks = 1 # ApolloProbe uses single chunk
        self.format = eval(config["short_name"] + "_format")

    def row_to_prompts(self, row: pd.Series) -> str:
        statement = row['statement']
        # no system prompt and tags for this one
        correct_user: str = "Pretend you're an honest person making statements about the world."
        incorrect_user: str = "Pretend you're a dishonest person making statements about the world."
        correct_format = self.format(None, correct_user, statement)
        incorrect_format = self.format(None, incorrect_user, statement)
        return correct_format, incorrect_format

    def process_row(self, row, num_tokens: int = 5, seq_len: int = 1024):
        correct_prompt, incorrect_prompt = self.row_to_prompts(row)
        correct_inputs = self.tokenizer(correct_prompt, return_tensors="pt", max_length=seq_len, truncation=True).to(self.device)
        incorrect_inputs = self.tokenizer(incorrect_prompt, return_tensors="pt", max_length=seq_len, truncation=True).to(self.device)        
        chunk_data = []
        
        # Find the position after "<|im_start|>user<|im_sep|>" in both prompts
        correct_assistant_start = correct_prompt.find("<|im_start|>assistant<|im_sep|>") + len("<|im_start|>assistant<|im_sep|>")
        incorrect_assistant_start = incorrect_prompt.find("<|im_start|>assistant<|im_sep|>") + len("<|im_start|>assistant<|im_sep|>")
        
        # Get token indices for the user message start positions
        correct_assistant_token_start = len(self.tokenizer.encode(correct_prompt[:correct_assistant_start], add_special_tokens=False))
        incorrect_assistant_token_start = len(self.tokenizer.encode(incorrect_prompt[:incorrect_assistant_start], add_special_tokens=False))
        
        with t.no_grad():
            self.activation_cache.clear_activations()
            self.model.forward(correct_inputs['input_ids'], attention_mask=correct_inputs['attention_mask'], pad_token_id=self.tokenizer.eos_token_id)
            correct_activations = self.activation_cache.activations[0][0]
            correct_acts = [correct_activations[correct_assistant_token_start + i] for i in range(num_tokens)]
            chunk_data.append((correct_acts, 1))
                
            self.activation_cache.clear_activations()
            self.model.forward(incorrect_inputs['input_ids'], attention_mask=incorrect_inputs['attention_mask'], pad_token_id=self.tokenizer.eos_token_id)
            incorrect_activations = self.activation_cache.activations[0][0]
            incorrect_acts = [incorrect_activations[incorrect_assistant_token_start + i] for i in range(num_tokens)]
            chunk_data.append((incorrect_acts, 0))
        return chunk_data

    def populate_dataset(self, force_redo: bool = False, num_tokens: int = 5, max_rows: int = 0, seq_len: int = 1024, chunk_size=None, **kwargs):
        if os.path.exists(self.get_chunk_path(0)) and not force_redo:
            print(f"Loading existing chunk from {self.get_chunk_path(0)}")
            return
            
        chunk_data = []
        for idx, row in self.df.iterrows():
            if row['label'] == 0:
                # do not add incorrect statements
                continue
            chunk_data.extend(self.process_row(row, num_tokens, seq_len))
            if idx % (len(self.df) // 10) == 0:
                print(f"Processed {idx} rows of {len(self.df)}")
        self.save_chunk(chunk_data, 0)
        free_unused_memory()
        self.num_total_chunks = 1