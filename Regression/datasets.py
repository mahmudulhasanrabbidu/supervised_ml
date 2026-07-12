"""
Dataset module for the Regression Project.
Downloads, caches, and prepares the Diabetes dataset.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# Global cache variables
_X_full = None
_y_full = None
_feature_names = None
_target_name = "disease_progression"
_y_mean = 0.0
_y_std = 1.0

def _fetch_diabetes():
    """Fetches the diabetes dataset from scikit-learn and caches it locally."""
    global _X_full, _y_full, _feature_names
    if _X_full is None:
        data = load_diabetes()
        _X_full = torch.tensor(data.data, dtype=torch.float32)
        _y_full = torch.tensor(data.target, dtype=torch.float32).view(-1, 1)
        _feature_names = data.feature_names
    return _X_full, _y_full

def train_test_split(X: torch.Tensor, y: torch.Tensor, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the dataset into training and testing sets.
    
    Args:
        X (torch.Tensor): Feature tensor.
        y (torch.Tensor): Target tensor.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Use deterministic random seed
    torch.manual_seed(random_state)
    n_samples = X.shape[0]
    indices = torch.randperm(n_samples)
    split_idx = int(n_samples * (1 - test_size))
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def normalize_features(X_train: torch.Tensor, X_test: torch.Tensor):
    """
    Normalizes features based on the training data statistics.
    
    Args:
        X_train (torch.Tensor): Training features.
        X_test (torch.Tensor): Testing features.
        
    Returns:
        tuple: (X_train_norm, X_test_norm)
    """
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)
    
    # Avoid division by zero
    std[std == 0] = 1e-8
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm

def inverse_transform_targets(y_norm: torch.Tensor):
    """
    Reverts target normalization if it was applied.
    
    Args:
        y_norm (torch.Tensor): Normalized target tensor.
        
    Returns:
        torch.Tensor: Target tensor in original scale.
    """
    global _y_mean, _y_std
    return y_norm * _y_std + _y_mean

def get_feature_names():
    """
    Returns the list of feature names for the dataset.
    
    Returns:
        list: List of feature names.
    """
    _fetch_diabetes()
    return _feature_names

def get_target_name():
    """
    Returns the name of the target variable.
    
    Returns:
        str: Target variable name.
    """
    return _target_name

def visualize_dataset():
    """
    Visualizes the dataset features against the target variable.
    Produces publication-quality scatter plots for each feature.
    """
    X, y = _fetch_diabetes()
    features = get_feature_names()
    
    # Calculate grid size (2 rows, 5 columns for 10 features)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Diabetes Dataset: Features vs Target", fontsize=16, y=1.02)
    
    for i, ax in enumerate(axes.flatten()):
        if i < len(features):
            ax.scatter(X[:, i].numpy(), y.numpy(), alpha=0.6, s=15, edgecolors='none', c='#1f77b4')
            ax.set_xlabel(features[i], fontsize=12)
            if i % 5 == 0:
                ax.set_ylabel(get_target_name(), fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title(f"Target vs {features[i]}", fontsize=10)
    
    plt.tight_layout()
    plt.show()

def load_dataset(test_size: float = 0.2, normalize: bool = True, random_state: int = 42):
    """
    Loads and prepares the Diabetes regression dataset.
    
    Args:
        test_size (float): Proportion of the dataset to include in the test split.
        normalize (bool): Whether to normalize the features.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) as PyTorch tensors.
    """
    global _y_mean, _y_std
    X, y = _fetch_diabetes()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)
    
    if normalize:
        X_train, X_test = normalize_features(X_train, X_test)
        
        # Optional: target normalization (commented out by default to keep target scale interpretable)
        # _y_mean = y_train.mean().item()
        # _y_std = y_train.std().item()
        # y_train = (y_train - _y_mean) / _y_std
        # y_test = (y_test - _y_mean) / _y_std
        
    return X_train, X_test, y_train, y_test