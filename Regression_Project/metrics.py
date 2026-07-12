"""
metrics.py
Evaluation metrics for regression analysis.
"""

import torch

def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean Squared Error: $ \frac{1}{n} \sum (y_i - \hat{y}_i)^2 $"""
    return torch.mean((y_true - y_pred) ** 2).item()

def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Root Mean Squared Error: $ \sqrt{MSE} $"""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean Absolute Error: $ \frac{1}{n} \sum |y_i - \hat{y}_i| $"""
    return torch.mean(torch.abs(y_true - y_pred)).item()

def mape(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean Absolute Percentage Error."""
    # Add epsilon to prevent division by zero
    epsilon = 1e-8
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))).item() * 100

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    R^2 (Coefficient of Determination).
    $ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} $
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return (1 - (ss_res / ss_tot)).item()

def adjusted_r2(y_true: torch.Tensor, y_pred: torch.Tensor, num_features: int) -> float:
    """
    Adjusted R^2 Score penalizes the addition of unnecessary features.
    """
    r2 = r2_score(y_true, y_pred)
    n = y_true.size(0)
    p = num_features
    if n - p - 1 <= 0:
        return float('nan')
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Explained Variance Score.
    $ EV = 1 - \frac{Var(y - \hat{y})}{Var(y)} $
    """
    var_res = torch.var(y_true - y_pred, unbiased=False)
    var_tot = torch.var(y_true, unbiased=False)
    if var_tot == 0:
        return 0.0
    return (1 - (var_res / var_tot)).item()