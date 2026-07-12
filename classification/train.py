from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent))

from algorithm import (
    LogisticRegression,
    Perceptron,
    KNNClassifier,
    GaussianNB,
    DecisionTreeClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    LightGBMClassifier,
    CatBoostClassifier,
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
from datasets import load_dataset
from metrics import accuracy_score, f1_score, confusion_matrix


def run_experiment(model_name: str, model, X_train: torch.Tensor, X_test: torch.Tensor, y_train: torch.Tensor, y_test: torch.Tensor) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "confusion": confusion_matrix(y_test, preds),
    }


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    X_train, X_test, y_train, y_test = load_dataset("moons", test_size=0.3, random_state=7)
    models = {
        "logistic": LogisticRegression(lr=0.05, epochs=300, batch_size=32),
        "perceptron": Perceptron(lr=0.1, epochs=200),
        "knn": KNNClassifier(k=5),
        "gnb": GaussianNB(),
        "tree": DecisionTreeClassifier(max_depth=4),
        "forest": RandomForestClassifier(n_estimators=10, max_depth=4),
        "adaboost": AdaBoostClassifier(n_estimators=8),
        "lightgbm": LightGBMClassifier(n_estimators=8, learning_rate=0.1),
        "catboost": CatBoostClassifier(n_estimators=8, learning_rate=0.1),
        "qda": QuadraticDiscriminantAnalysis(),
        "lda": LinearDiscriminantAnalysis(),
    }
    for name, model in models.items():
        metrics = run_experiment(name, model, X_train, X_test, y_train, y_test)
        print(name, metrics["accuracy"], metrics["f1"])
