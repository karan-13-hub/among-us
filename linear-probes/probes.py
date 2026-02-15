import torch.nn as nn
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np

class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.register_buffer('mean', t.zeros(input_dim))
        self.register_buffer('std', t.ones(input_dim))
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        # Normalize input using running statistics
        clamped_std = t.clamp(self.std, min=1e-8)
        x_normalized = (x.to(self.std.device) - self.mean) / clamped_std
        return self.linear(x_normalized)

class LinearProbe:
    def __init__(self, input_dim, criterion=None, optimizer_cls=optim.Adam, lr=0.001, step_size=100, gamma=0.7, device='cpu', seed=42, verbose=False, weight_decay=1e-3):
        # Set seeds for reproducibility
        self.seed = seed
        self._set_seeds(seed)
        
        self.device = device
        self.model = LinearModel(input_dim).to(self.device)
        self.criterion = criterion if criterion else nn.BCEWithLogitsLoss()
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr, weight_decay=weight_decay)  # L2 regularization
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.train_accs = []
        self.verbose = verbose
    def _set_seeds(self, seed):
        """Set seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        t.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False

    def update_normalization_stats(self, loader):
        mean_sum = t.zeros(self.model.linear.weight.shape[1]).to(self.device)
        var_sum = t.zeros(self.model.linear.weight.shape[1]).to(self.device)
        total_samples = 0

        # First pass for mean
        for X_batch, _ in loader:
            X_batch = X_batch.to(self.device, dtype=t.float32)
            mean_sum += X_batch.sum(dim=0)
            total_samples += X_batch.shape[0]
        
        mean = mean_sum / total_samples
        
        # Second pass for variance
        for X_batch, _ in loader:
            X_batch = X_batch.to(self.device, dtype=t.float32)
            var_sum += ((X_batch - mean) ** 2).sum(dim=0)
        
        std = t.sqrt(var_sum / total_samples)
        
        # Update model's buffers
        self.model.mean.copy_(mean)
        self.model.std.copy_(std)

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device, dtype=t.float32), y_batch.to(self.device, dtype=t.float32)
            
            y_pred = self.model(X_batch).squeeze()
            loss = self.criterion(y_pred, y_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            correct += ((y_pred > 0) == y_batch).sum().item()  # Changed from round() to > 0
            total += y_batch.size(0)
        
        return total_loss / len(train_loader), correct / total
    
    def evaluate_single_activation(self, activation):
        """Evaluate a single activation and return its score.
        
        Args:
            activation: torch.Tensor of shape (input_dim,) or (1, input_dim)
            
        Returns:
            float: The model's score (logit) for this activation
        """
        if activation.dim() == 1:
            activation = activation.unsqueeze(0)
        self.model.eval()
        with t.no_grad():
            activation = activation.to(self.device, dtype=t.float32)
            
            # Get model's prediction
            score = t.sigmoid(self.model(activation)).item()
            return score

    def fit(self, train_loader: DataLoader, epochs=1000):
        # Reset seeds before training for reproducibility
        self._set_seeds(self.seed)
        
        # Calculate normalization statistics from training data
        self.update_normalization_stats(train_loader)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            self.lr_scheduler.step()
            
            self.train_accs.append(train_acc)
            
            if self.verbose:
                if epochs > 10 and epoch % (epochs // 10) == 0 or epochs <= 10:
                    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}")
        
        if self.verbose:
            print(f"Final Train Acc: {self.train_accs[-1]:.4f}")
        return self.train_accs[-1]

    def accuracy(self, val_loader: DataLoader):
        self.model.eval()
        total_correct, total_samples = 0, 0
        
        with t.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device, dtype=t.float32)
                y_pred = self.model(X_batch).squeeze()
                total_correct += ((y_pred > 0) == y_batch).sum().item()
                total_samples += y_batch.size(0)
        
        return total_correct / total_samples