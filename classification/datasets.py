from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_moons, load_wine, load_digits, fetch_openml
from sklearn.model_selection import train_test_split as skl_train_test_split


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensor(X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long).view(-1)
    return X_t, y_t


def load_heart_disease_dataset(test_size: float = 0.3, random_state: int = 42):
    set_seed(random_state)
    heart_data = fetch_openml(name="heart-disease", version=1, as_frame=True, parser="auto")
    df = heart_data.frame
    target_col = "target" if "target" in df.columns else df.columns[-1]
    feature_frame = df.drop(columns=[target_col])
    target = df[target_col].astype(int)

    feature_frame = pd.get_dummies(feature_frame, drop_first=True)
    feature_frame = feature_frame.fillna(feature_frame.mean())
    feature_frame = feature_frame.astype(float)

    X_train, X_test, y_train, y_test = skl_train_test_split(
        feature_frame, target, test_size=test_size, random_state=random_state, stratify=target
    )
    X_train_t, y_train_t = _to_tensor(X_train.to_numpy(), y_train.to_numpy())
    X_test_t, y_test_t = _to_tensor(X_test.to_numpy(), y_test.to_numpy())

    mean = X_train_t.mean(dim=0, keepdim=True)
    std = X_train_t.std(dim=0, keepdim=True)
    std[std < 1e-8] = 1.0
    X_train_t = (X_train_t - mean) / std
    X_test_t = (X_test_t - mean) / std
    return X_train_t, X_test_t, y_train_t, y_test_t


def load_dataset(name: str = "moons", test_size: float = 0.3, random_state: int = 42):
    set_seed(random_state)
    if name.lower() == "moons":
        X, y = make_moons(n_samples=400, noise=0.2, random_state=random_state)
        X_train, X_test, y_train, y_test = skl_train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    elif name.lower() == "wine":
        data = load_wine()
        X_train, X_test, y_train, y_test = skl_train_test_split(
            data.data, data.target, test_size=test_size, random_state=random_state, stratify=data.target
        )
    elif name.lower() == "digits":
        data = load_digits()
        X_train, X_test, y_train, y_test = skl_train_test_split(
            data.data, data.target, test_size=test_size, random_state=random_state, stratify=data.target
        )
    elif name.lower() == "heart":
        return load_heart_disease_dataset(test_size=test_size, random_state=random_state)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    X_train_t, y_train_t = _to_tensor(X_train, y_train)
    X_test_t, y_test_t = _to_tensor(X_test, y_test)

    mean = X_train_t.mean(dim=0, keepdim=True)
    std = X_train_t.std(dim=0, keepdim=True)
    std[std < 1e-8] = 1.0
    X_train_t = (X_train_t - mean) / std
    X_test_t = (X_test_t - mean) / std
    return X_train_t, X_test_t, y_train_t, y_test_t
