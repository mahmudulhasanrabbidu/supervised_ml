from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

__all__ = [
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_class_distribution",
    "plot_probability_distribution",
    "plot_learning_curve",
    "plot_accuracy_curve",
    "plot_loss_curve",
    "plot_feature_importance",
    "plot_decision_boundary",
    "plot_misclassified_samples",
    "compare_metrics",
    "plot_predictions",
    "plot_residuals",
    "plot_error_histogram",
    "plot_learning_curve_regression",
    "compare_regression_metrics",
    "plot_feature_importance_regression",
    "save_figure",
    "set_plot_style",
    "close_all_figures",
]


def _as_numpy(data: Any) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, pd.Series):
        return data.to_numpy()
    if isinstance(data, (list, tuple)):
        return np.asarray(data)
    return np.asarray(data)


def _get_axis(ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    return ax


def set_plot_style(style_name: str = "seaborn-v0_8-darkgrid") -> None:
    try:
        plt.style.use(style_name)
    except OSError:
        plt.style.use("default")


def save_figure(
    filename: str,
    dpi: int = 300,
    bbox_inches: str = "tight",
    show: bool = False,
    **kwargs: Any,
) -> None:
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    if show:
        plt.show()


def close_all_figures() -> None:
    plt.close("all")


def _format_axes(
    ax: plt.Axes,
    title: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: Optional[Sequence[Any]] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    y_true_arr = _as_numpy(y_true).astype(int).ravel()
    y_pred_arr = _as_numpy(y_pred).astype(int).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    labels = list(labels)
    mapping = {label: idx for idx, label in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(y_true_arr, y_pred_arr):
        cm[mapping[int(actual)], mapping[int(predicted)]] += 1

    ax = _get_axis(ax)
    image = ax.imshow(cm, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.grid(False)
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = cm[row, col]
            color = "white" if value > cm.max() / 2 else "black"
            ax.text(col, row, int(value), ha="center", va="center", color=color)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_roc_curve(
    y_true: Any,
    y_score: Any,
    pos_label: int = 1,
    title: str = "ROC Curve",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    y_true_arr = _as_numpy(y_true).astype(int).ravel()
    scores = _as_numpy(y_score)
    if scores.ndim > 1:
        scores = scores[:, -1] if scores.shape[1] > 1 else scores[:, 0]
    scores = scores.ravel()

    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = y_true_arr[order]
    eps = 1e-12

    thresholds = np.unique(np.concatenate(([sorted_scores.min() - 1e-6], sorted_scores, [sorted_scores.max() + 1e-6])))
    fpr = []
    tpr = []
    for threshold in thresholds:
        predicted = (sorted_scores >= threshold)
        tp = np.sum((predicted == 1) & (sorted_labels == pos_label))
        fp = np.sum((predicted == 1) & (sorted_labels != pos_label))
        fn = np.sum((predicted == 0) & (sorted_labels == pos_label))
        tn = np.sum((predicted == 0) & (sorted_labels != pos_label))
        tpr.append(tp / (tp + fn + eps))
        fpr.append(fp / (fp + tn + eps))

    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    order_idx = np.argsort(fpr)
    fpr = fpr[order_idx]
    tpr = tpr[order_idx]
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    auc_score = float(np.trapz(tpr, fpr))

    ax = _get_axis(ax)
    ax.plot(fpr, tpr, linewidth=2, color="#4c78a8")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    ax.set_title(title + f" (AUC = {auc_score:.3f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    if show:
        plt.show()
    return fpr, tpr, auc_score


def plot_precision_recall_curve(
    y_true: Any,
    y_score: Any,
    pos_label: int = 1,
    title: str = "Precision-Recall Curve",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true_arr = _as_numpy(y_true).astype(int).ravel()
    scores = _as_numpy(y_score)
    if scores.ndim > 1:
        scores = scores[:, -1] if scores.shape[1] > 1 else scores[:, 0]
    scores = scores.ravel()

    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = y_true_arr[order]
    eps = 1e-12

    thresholds = np.unique(np.concatenate(([sorted_scores.min() - 1e-6], sorted_scores, [sorted_scores.max() + 1e-6])))
    precision = []
    recall = []
    for threshold in thresholds:
        predicted = (sorted_scores >= threshold)
        tp = np.sum((predicted == 1) & (sorted_labels == pos_label))
        fp = np.sum((predicted == 1) & (sorted_labels != pos_label))
        fn = np.sum((predicted == 0) & (sorted_labels == pos_label))
        precision.append(tp / (tp + fp + eps))
        recall.append(tp / (tp + fn + eps))

    precision = np.asarray(precision)
    recall = np.asarray(recall)
    order_idx = np.argsort(recall)
    precision = precision[order_idx]
    recall = recall[order_idx]
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))

    ax = _get_axis(ax)
    ax.plot(recall, precision, linewidth=2, color="#f58518")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    if show:
        plt.show()
    return recall, precision


def plot_class_distribution(
    y_true: Any,
    labels: Optional[Sequence[Any]] = None,
    title: str = "Class Distribution",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    y_true_arr = _as_numpy(y_true).astype(int).ravel()
    if labels is None:
        labels = np.unique(y_true_arr)
    counts = [np.sum(y_true_arr == label) for label in labels]

    ax = _get_axis(ax)
    ax.bar(labels, counts, color=["#4c78a8", "#f58518"][: len(labels)], edgecolor="black")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, count in zip(ax.patches, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.01, int(count), ha="center", va="bottom")
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_probability_distribution(
    probabilities: Any,
    title: str = "Probability Distribution",
    bins: int = 20,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    scores = _as_numpy(probabilities)
    if scores.ndim > 1:
        scores = scores[:, -1] if scores.shape[1] > 1 else scores[:, 0]
    scores = scores.ravel()

    ax = _get_axis(ax)
    ax.hist(scores, bins=bins, color="#4c78a8", edgecolor="black", alpha=0.9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_learning_curve(
    train_scores: Any,
    val_scores: Optional[Any] = None,
    title: str = "Learning Curve",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    train_scores = _as_numpy(train_scores).ravel()
    epoch_index = np.arange(1, len(train_scores) + 1)

    ax = _get_axis(ax)
    ax.plot(epoch_index, train_scores, marker="o", linewidth=2, label="Training")
    if val_scores is not None:
        val_scores = _as_numpy(val_scores).ravel()
        ax.plot(epoch_index[: len(val_scores)], val_scores, marker="s", linewidth=2, label="Validation")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_accuracy_curve(
    history: Any,
    title: str = "Accuracy Curve",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    if isinstance(history, dict):
        keys = [key for key in history.keys() if "acc" in key.lower()]
        if not keys:
            raise ValueError("No accuracy history found in the provided dictionary.")
        values = _as_numpy(history[keys[0]]).ravel()
    else:
        values = _as_numpy(history).ravel()
    epoch_index = np.arange(1, len(values) + 1)
    ax = _get_axis(ax)
    ax.plot(epoch_index, values, marker="o", linewidth=2, color="#4c78a8")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_loss_curve(
    history: Any,
    title: str = "Loss Curve",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    if isinstance(history, dict):
        keys = [key for key in history.keys() if "loss" in key.lower()]
        if not keys:
            raise ValueError("No loss history found in the provided dictionary.")
        values = _as_numpy(history[keys[0]]).ravel()
    else:
        values = _as_numpy(history).ravel()
    epoch_index = np.arange(1, len(values) + 1)
    ax = _get_axis(ax)
    ax.plot(epoch_index, values, marker="o", linewidth=2, color="#f58518")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_feature_importance(
    importances: Any,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Feature Importance",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    values = _as_numpy(importances).ravel()
    if feature_names is None:
        feature_names = [f"Feature {idx}" for idx in range(len(values))]
    order = np.argsort(values)[::-1]
    ordered_values = values[order]
    ordered_names = [feature_names[idx] for idx in order]

    ax = _get_axis(ax)
    ax.barh(ordered_names, ordered_values, color="#4c78a8")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_decision_boundary(
    X: Any,
    y: Any,
    model: Optional[Any] = None,
    title: str = "Decision Boundary",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    feature_names: Optional[Sequence[str]] = None,
    grid_size: int = 200,
) -> plt.Axes:
    X_arr = _as_numpy(X)
    y_arr = _as_numpy(y).astype(int).ravel()
    if X_arr.ndim != 2 or X_arr.shape[1] < 2:
        raise ValueError("X must be a 2D array with at least two features.")

    X_two = X_arr[:, :2]
    ax = _get_axis(ax)
    ax.scatter(
        X_two[y_arr == 0, 0],
        X_two[y_arr == 0, 1],
        c="#4c78a8",
        edgecolors="black",
        label="Class 0",
        alpha=0.8,
    )
    ax.scatter(
        X_two[y_arr == 1, 0],
        X_two[y_arr == 1, 1],
        c="#f58518",
        edgecolors="black",
        label="Class 1",
        alpha=0.8,
    )

    if model is not None:
        x_min, x_max = X_two[:, 0].min() - 0.5, X_two[:, 0].max() + 0.5
        y_min, y_max = X_two[:, 1].min() - 0.5, X_two[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(grid)
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                prediction_labels = np.argmax(predictions, axis=1)
            else:
                prediction_labels = (predictions[:, 0] > 0.5).astype(int)
        elif hasattr(model, "predict"):
            prediction_labels = model.predict(grid)
        else:
            prediction_labels = np.zeros(len(grid), dtype=int)
        ax.contourf(xx, yy, prediction_labels.reshape(xx.shape), alpha=0.2, cmap="coolwarm")

    if feature_names is not None and len(feature_names) >= 2:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_misclassified_samples(
    X: Any,
    y_true: Any,
    y_pred: Any,
    title: str = "Misclassified Samples",
    sample_limit: int = 20,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    X_arr = _as_numpy(X)
    y_true_arr = _as_numpy(y_true).astype(int).ravel()
    y_pred_arr = _as_numpy(y_pred).astype(int).ravel()
    if X_arr.ndim != 2 or X_arr.shape[1] < 2:
        raise ValueError("X must be a 2D array with at least two features.")

    X_two = X_arr[:, :2]
    mask = y_true_arr != y_pred_arr
    indices = np.flatnonzero(mask)[:sample_limit]
    ax = _get_axis(ax)
    ax.scatter(X_two[:, 0], X_two[:, 1], c="#4c78a8", alpha=0.6, edgecolors="black")
    ax.scatter(X_two[indices, 0], X_two[indices, 1], c="#f58518", edgecolors="black", s=90, label="Misclassified")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def compare_metrics(
    results: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True,
) -> None:
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be provided as a pandas.DataFrame")
    if "Algorithm" not in results.columns:
        raise ValueError("results must include an 'Algorithm' column")

    if metrics is None:
        metrics = ["Accuracy", "Precision", "Recall", "F1", "F1 Score", "Training Time", "Prediction Time"]

    for metric in metrics:
        metric_name = None
        for possible in [metric, metric.lower(), metric.replace(" ", " ").lower(), metric.replace(" ", "_").lower()]:
            if possible in {col.lower() for col in results.columns}:
                metric_name = next(col for col in results.columns if col.lower() == possible)
                break
        if metric_name is None:
            continue
        ordered = results[["Algorithm", metric_name]].copy()
        ordered = ordered.sort_values(by=metric_name, ascending=False)
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(ordered["Algorithm"], ordered[metric_name], color="#4c78a8", edgecolor="black")
        _format_axes(ax, title=f"{metric_name} Comparison", xlabel="Algorithm", ylabel=metric_name)
        for bar, value in zip(bars, ordered[metric_name]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
        plt.tight_layout()
        if show:
            plt.show()


def plot_predictions(
    y_true: Any,
    y_pred: Any,
    title: str = "Predictions vs Actual",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    y_true_arr = _as_numpy(y_true).ravel()
    y_pred_arr = _as_numpy(y_pred).ravel()
    ax = _get_axis(ax)
    ax.scatter(y_true_arr, y_pred_arr, color="#4c78a8", edgecolors="black", alpha=0.8)
    min_val = min(y_true_arr.min(), y_pred_arr.min())
    max_val = max(y_true_arr.max(), y_pred_arr.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1.2)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_residuals(
    y_true: Any,
    y_pred: Any,
    title: str = "Residual Plot",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    y_true_arr = _as_numpy(y_true).ravel()
    y_pred_arr = _as_numpy(y_pred).ravel()
    residuals = y_true_arr - y_pred_arr
    ax = _get_axis(ax)
    ax.axhline(0, color="black", linestyle="--")
    ax.scatter(y_pred_arr, residuals, color="#f58518", edgecolors="black", alpha=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_error_histogram(
    y_true: Any,
    y_pred: Any,
    bins: int = 30,
    title: str = "Error Distribution",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    y_true_arr = _as_numpy(y_true).ravel()
    y_pred_arr = _as_numpy(y_pred).ravel()
    errors = y_true_arr - y_pred_arr
    ax = _get_axis(ax)
    ax.hist(errors, bins=bins, color="#4c78a8", edgecolor="black", alpha=0.9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Error")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_learning_curve_regression(
    train_scores: Any,
    val_scores: Optional[Any] = None,
    title: str = "Regression Learning Curve",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    return plot_learning_curve(train_scores, val_scores=val_scores, title=title, ax=ax, show=show)


def compare_regression_metrics(
    results: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True,
) -> None:
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be provided as a pandas.DataFrame")
    if "Model" not in results.columns and "Algorithm" not in results.columns:
        raise ValueError("results must include an 'Algorithm' or 'Model' column")

    if metrics is None:
        metrics = ["MAE", "RMSE", "R2", "Training Time", "Prediction Time"]

    algorithm_col = "Algorithm" if "Algorithm" in results.columns else "Model"
    for metric in metrics:
        metric_name = next((col for col in results.columns if col.lower() == metric.lower()), None)
        if metric_name is None:
            continue
        ordered = results[[algorithm_col, metric_name]].copy()
        ordered = ordered.sort_values(by=metric_name, ascending=True)
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(ordered[algorithm_col], ordered[metric_name], color="#4c78a8", edgecolor="black")
        _format_axes(ax, title=f"{metric_name} Comparison", xlabel=algorithm_col, ylabel=metric_name)
        for bar, value in zip(bars, ordered[metric_name]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
        plt.tight_layout()
        if show:
            plt.show()


def plot_feature_importance_regression(
    importances: Any,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Regression Feature Importance",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    return plot_feature_importance(importances, feature_names=feature_names, title=title, ax=ax, show=show)
