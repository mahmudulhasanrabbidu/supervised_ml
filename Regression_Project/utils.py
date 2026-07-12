"""
utils.py
Utility functions for data scaling, batching, splitting, and saving/loading models.
"""

import torch
import random
import os
from typing import Tuple, Generator

def set_seed(seed: int = 42) -> None:
    """Sets deterministic seed for reproducibility across all libraries."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def train_test_split(X: torch.Tensor, y: torch.Tensor, test_size: float = 0.2) -> Tuple[torch.Tensor, ...]:
    """
    Splits tensors into random train and test subsets using PyTorch.
    """
    num_samples = X.size(0)
    indices = torch.randperm(num_samples)
    
    split_idx = int(num_samples * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    z = (x - u) / s
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: torch.Tensor) -> None:
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, unbiased=False, keepdim=True)
        self.std[self.std == 0] = 1e-8 # Prevent division by zero

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)

class MinMaxScaler:
    """
    Transforms features by scaling each feature to a given range [0, 1].
    X_std = (X - X.min) / (X.max - X.min)
    """
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X: torch.Tensor) -> None:
        self.min = X.min(dim=0, keepdim=True)[0]
        self.max = X.max(dim=0, keepdim=True)[0]
        # Prevent division by zero
        self.max = torch.where(self.max == self.min, self.max + 1e-8, self.max)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)

def batch_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True) -> Generator:
    """
    Yields mini-batches from the dataset.
    """
    num_samples = X.size(0)
    indices = torch.randperm(num_samples) if shuffle else torch.arange(num_samples)
    
    for start_idx in range(0, num_samples, batch_size):
        batch_idx = indices[start_idx:start_idx + batch_size]
        yield X[batch_idx], y[batch_idx]