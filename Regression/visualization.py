"""
visualization.py
Matplotlib visualization functions for evaluating regression models.
"""

import matplotlib.pyplot as plt
import torch
import os
from typing import Dict, List, Any

# Ensure output directory exists
os.makedirs("plots", exist_ok=True)

def plot_dataset(X: torch.Tensor, y: torch.Tensor, title: str = "Dataset", save_path: str = None) -> None:
    """Plots the generated dataset."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X.numpy(), y.numpy(), color="black", s=15, alpha=0.6, label="Observed")
    plt.title(title)
    plt.xlabel("Feature (X)")
    plt.ylabel("Target (y)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_prediction(X: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, 
                    title: str = "Model Prediction", save_path: str = None) -> None:
    """
    Plots the observed data points (black) and the predicted curve (blue).
    X should be sorted before calling this for a smooth line.
    """
    # Sort X for smooth line plotting
    sorted_indices = torch.argsort(X, dim=0).squeeze()
    X_sorted = X[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    plt.figure(figsize=(8, 6))
    plt.scatter(X.numpy(), y_true.numpy(), color="black", s=15, alpha=0.5, label="Observed")
    plt.plot(X_sorted.numpy(), y_pred_sorted.numpy(), color="tab:blue", linewidth=2.5, label="Predicted / Imputed")
    
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_residuals(y_true: torch.Tensor, y_pred: torch.Tensor, save_path: str = None) -> None:
    """Plots the residual errors ($ y - \hat{y} $)."""
    residuals = (y_true - y_pred).numpy()
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter of residuals
    ax[0].scatter(y_pred.numpy(), residuals, color="tab:red", alpha=0.6, s=15)
    ax[0].axhline(0, color="black", linestyle="--")
    ax[0].set_title("Residuals vs Predicted")
    ax[0].set_xlabel("Predicted Values")
    ax[0].set_ylabel("Residuals")
    
    # Histogram of residuals
    ax[1].hist(residuals, bins=20, color="tab:red", alpha=0.6, edgecolor="black")
    ax[1].set_title("Error Histogram")
    ax[1].set_xlabel("Residual Error")
    ax[1].set_ylabel("Frequency")
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def compare_metrics(metrics_dict: Dict[str, Dict[str, float]], metric_name: str = "R2 Score", save_path: str = None) -> None:
    """
    Bar chart comparing different models on a specific metric.
    """
    models = list(metrics_dict.keys())
    scores = [metrics_dict[m][metric_name] for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color="tab:green", alpha=0.7, edgecolor="black")
    plt.title(f"Model Comparison: {metric_name}")
    plt.ylabel(metric_name)
    plt.xticks(rotation=45, ha="right")
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha="center", va="bottom")
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()