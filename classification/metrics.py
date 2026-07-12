from __future__ import annotations

import torch


def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true == y_pred).float().mean().item()


def precision_score(y_true: torch.Tensor, y_pred: torch.Tensor, average: str = "binary") -> float:
    if average == "binary":
        tp = ((y_true == 1) & (y_pred == 1)).sum().item()
        fp = ((y_true == 0) & (y_pred == 1)).sum().item()
        return tp / (tp + fp + 1e-12)
    labels = torch.unique(y_true)
    scores = []
    for label in labels:
        tp = ((y_true == label) & (y_pred == label)).sum().item()
        fp = ((y_true != label) & (y_pred == label)).sum().item()
        scores.append(tp / (tp + fp + 1e-12))
    return sum(scores) / len(scores)


def recall_score(y_true: torch.Tensor, y_pred: torch.Tensor, average: str = "binary") -> float:
    if average == "binary":
        tp = ((y_true == 1) & (y_pred == 1)).sum().item()
        fn = ((y_true == 1) & (y_pred == 0)).sum().item()
        return tp / (tp + fn + 1e-12)
    labels = torch.unique(y_true)
    scores = []
    for label in labels:
        tp = ((y_true == label) & (y_pred == label)).sum().item()
        fn = ((y_true == label) & (y_pred != label)).sum().item()
        scores.append(tp / (tp + fn + 1e-12))
    return sum(scores) / len(scores)


def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor, average: str = "binary") -> float:
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    return 2 * p * r / (p + r + 1e-12)


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, labels=None) -> torch.Tensor:
    if labels is None:
        labels = torch.unique(torch.cat([y_true, y_pred]))
    cm = torch.zeros((len(labels), len(labels)), dtype=torch.long)
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            cm[i, j] = ((y_true == l1) & (y_pred == l2)).sum().item()
    return cm


def log_loss(y_true: torch.Tensor, y_pred_proba: torch.Tensor) -> float:
    eps = 1e-12
    y_true_one_hot = torch.nn.functional.one_hot(y_true, num_classes=y_pred_proba.size(1)).float()
    return (-torch.sum(y_true_one_hot * torch.log(y_pred_proba + eps)) / y_true.size(0)).item()
