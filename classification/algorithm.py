from __future__ import annotations

import math
import os
import pickle
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseClassifier:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.is_fitted_ = False

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        raise NotImplementedError

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        from metrics import accuracy_score

        return accuracy_score(y, self.predict(X))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self.__dict__, handle)

    def load(self, path: str) -> None:
        with open(path, "rb") as handle:
            self.__dict__.update(pickle.load(handle))


class LogisticRegression(BaseClassifier):
    def __init__(self, lr: float = 0.05, epochs: int = 200, batch_size: int = 32, device: str = "cpu") -> None:
        super().__init__(device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.device)
        y = y.to(self.device).float().view(-1, 1)
        n_features = X.size(1)
        self.weights = torch.zeros((n_features, 1), device=self.device, dtype=torch.float32, requires_grad=True)
        self.bias = torch.zeros(1, device=self.device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([self.weights, self.bias], lr=self.lr)
        criterion = nn.BCELoss()

        for _ in range(self.epochs):
            indices = torch.randperm(X.size(0), device=self.device)
            for start in range(0, X.size(0), self.batch_size):
                idx = indices[start:start + self.batch_size]
                xb = X[idx]
                yb = y[idx]
                logits = torch.sigmoid(xb @ self.weights + self.bias)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.is_fitted_ = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(X)
        return (probs >= 0.5).long().view(-1)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fit before calling predict_proba")
        X = X.to(self.device)
        probs = torch.sigmoid(X @ self.weights + self.bias)
        return probs.detach().cpu().view(-1)


class Perceptron(BaseClassifier):
    def __init__(self, lr: float = 0.1, epochs: int = 200, device: str = "cpu") -> None:
        super().__init__(device)
        self.lr = lr
        self.epochs = epochs
        self.weights: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.device)
        y = y.to(self.device).float().view(-1)
        self.weights = torch.zeros(X.size(1), device=self.device, dtype=torch.float32)
        self.bias = torch.zeros(1, device=self.device, dtype=torch.float32)
        for _ in range(self.epochs):
            for i in range(X.size(0)):
                x = X[i]
                pred = 1 if (x @ self.weights + self.bias).item() >= 0 else 0
                target = int(y[i].item())
                if pred != target:
                    self.weights += self.lr * (target - pred) * x
                    self.bias += self.lr * (target - pred)
        self.is_fitted_ = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        scores = X @ self.weights + self.bias
        return (scores >= 0).long().view(-1)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        scores = X @ self.weights + self.bias
        probs = torch.sigmoid(scores)
        return probs.detach().cpu().view(-1)


class KNNClassifier(BaseClassifier):
    def __init__(self, k: int = 5, device: str = "cpu") -> None:
        super().__init__(device)
        self.k = k
        self.X_train: Optional[torch.Tensor] = None
        self.y_train: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X_train = X.to(self.device)
        self.y_train = y.to(self.device).view(-1)
        self.is_fitted_ = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(X)
        return probs.argmax(dim=1).long()

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        distances = torch.cdist(X, self.X_train, p=2.0)
        topk = torch.topk(distances, k=min(self.k, self.X_train.size(0)), largest=False, dim=1)
        labels = self.y_train[topk.indices]
        classes = torch.unique(self.y_train)
        probs = torch.zeros((X.size(0), len(classes)), device=self.device, dtype=torch.float32)
        for idx, label in enumerate(classes.tolist()):
            probs[:, idx] = (labels == label).sum(dim=1).float() / topk.indices.size(1)
        return probs.detach().cpu()


class GaussianNB(BaseClassifier):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device)
        self.class_means: Optional[torch.Tensor] = None
        self.class_stds: Optional[torch.Tensor] = None
        self.class_priors: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.device)
        y = y.to(self.device)
        classes = torch.unique(y)
        self.class_means = torch.stack([X[y == c].mean(dim=0) for c in classes], dim=0)
        self.class_stds = torch.stack([X[y == c].std(dim=0, unbiased=False) for c in classes], dim=0)
        self.class_stds[self.class_stds < 1e-8] = 1.0
        self.class_priors = torch.tensor([torch.sum(y == c).item() / len(y) for c in classes], device=self.device)
        self.is_fitted_ = True

    def _log_prob(self, X: torch.Tensor) -> torch.Tensor:
        n_samples = X.size(0)
        n_classes = self.class_means.size(0)
        log_probs = torch.empty((n_samples, n_classes), device=self.device, dtype=torch.float32)
        for class_idx in range(n_classes):
            mean = self.class_means[class_idx]
            var = self.class_stds[class_idx].square()
            log_norm = -0.5 * torch.log(2 * torch.pi * var)
            diff = X - mean
            exponent = log_norm - (diff.square() / (2 * var))
            log_probs[:, class_idx] = exponent.sum(dim=1)
        return log_probs

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(X)
        return probs.argmax(dim=1).long()

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        log_probs = self._log_prob(X) + torch.log(self.class_priors).to(self.device)
        return torch.softmax(log_probs, dim=1).detach().cpu()


class DecisionTreeClassifier(BaseClassifier):
    def __init__(self, max_depth: int = 4, device: str = "cpu") -> None:
        super().__init__(device)
        self.max_depth = max_depth
        self.tree: Optional[Dict[str, Any]] = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.tree = self._grow_tree(X.cpu(), y.cpu(), depth=0)
        self.is_fitted_ = True

    def _grow_tree(self, X: torch.Tensor, y: torch.Tensor, depth: int) -> Dict[str, Any]:
        if depth >= self.max_depth or X.size(0) <= 1 or torch.unique(y).numel() == 1:
            counts = torch.bincount(y.long())
            label = int(counts.argmax().item())
            return {"is_leaf": True, "label": label}
        best_split = self._find_best_split(X, y)
        if best_split is None:
            counts = torch.bincount(y.long())
            label = int(counts.argmax().item())
            return {"is_leaf": True, "label": label}
        left_mask = X[:, best_split["feature_idx"]] <= best_split["threshold"]
        right_mask = ~left_mask
        return {
            "is_leaf": False,
            "feature_idx": best_split["feature_idx"],
            "threshold": best_split["threshold"],
            "left": self._grow_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._grow_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def _find_best_split(self, X: torch.Tensor, y: torch.Tensor) -> Optional[Dict[str, Any]]:
        best_score = float("inf")
        best = None
        for feature_idx in range(X.size(1)):
            thresholds = torch.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                left_y = y[left_mask]
                right_y = y[right_mask]
                gini = self._gini(left_y) * len(left_y) / len(y) + self._gini(right_y) * len(right_y) / len(y)
                if gini < best_score:
                    best_score = float(gini)
                    best = {"feature_idx": feature_idx, "threshold": threshold.item()}
        return best

    def _gini(self, y: torch.Tensor) -> float:
        counts = torch.bincount(y.long())
        probs = counts.float() / max(len(y), 1)
        return 1.0 - torch.sum(probs.square()).item()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        preds = [self._predict_single(x, self.tree) for x in X]
        return torch.tensor(preds, dtype=torch.long)

    def _predict_single(self, x: torch.Tensor, tree: Dict[str, Any]) -> int:
        if tree["is_leaf"]:
            return tree["label"]
        if x[tree["feature_idx"]] <= tree["threshold"]:
            return self._predict_single(x, tree["left"])
        return self._predict_single(x, tree["right"])

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        preds = self.predict(X)
        classes = torch.unique(preds)
        return torch.nn.functional.one_hot(preds, num_classes=int(classes.max().item()) + 1).float()


class RandomForestClassifier(BaseClassifier):
    def __init__(self, n_estimators: int = 10, max_depth: int = 4, device: str = "cpu") -> None:
        super().__init__(device)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees: list[DecisionTreeClassifier] = []

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.trees = []
        n_samples = X.size(0)
        for _ in range(self.n_estimators):
            indices = torch.randint(0, n_samples, (n_samples,))
            tree = DecisionTreeClassifier(max_depth=self.max_depth, device=self.device)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
        self.is_fitted_ = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(X)
        return probs.argmax(dim=1).long()

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        all_probs = torch.stack([tree.predict_proba(X) for tree in self.trees], dim=0)
        return all_probs.mean(dim=0)


class AdaBoostClassifier(BaseClassifier):
    def __init__(self, n_estimators: int = 8, device: str = "cpu") -> None:
        super().__init__(device)
        self.n_estimators = n_estimators
        self.stumps: list[Dict[str, Any]] = []
        self.alphas: list[float] = []
        self.classes: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.device)
        y = y.to(self.device).view(-1)
        self.classes = torch.unique(y)
        if self.classes.numel() != 2:
            raise NotImplementedError("AdaBoost is implemented for binary classification only")
        y_bin = torch.where(y == self.classes[0], -1.0, 1.0)
        sample_weights = torch.full((X.size(0),), 1.0 / X.size(0), device=self.device)
        self.stumps = []
        self.alphas = []

        for _ in range(self.n_estimators):
            best_stump: Optional[Dict[str, Any]] = None
            best_error = float("inf")
            for feature_idx in range(X.size(1)):
                thresholds = torch.unique(X[:, feature_idx])
                for threshold in thresholds:
                    left_pred = torch.where(X[:, feature_idx] <= threshold, -1.0, 1.0)
                    wrong = (left_pred != y_bin).float()
                    error = (sample_weights * wrong).sum().item()
                    if error < best_error:
                        best_error = float(error)
                        best_stump = {
                            "feature_idx": feature_idx,
                            "threshold": float(threshold.item()),
                            "prediction": left_pred.detach().cpu(),
                        }
            if best_stump is None or best_error >= 0.5:
                break
            alpha = 0.5 * math.log((1.0 - best_error) / max(best_error, 1e-12))
            pred = best_stump["prediction"].to(self.device)
            sample_weights = sample_weights * torch.exp(-alpha * y_bin * pred)
            sample_weights = sample_weights / sample_weights.sum().clamp_min(1e-12)
            best_stump["alpha"] = alpha
            self.stumps.append(best_stump)
            self.alphas.append(alpha)
            if best_error < 1e-8:
                break
        self.is_fitted_ = True

    def _margin(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        margin = torch.zeros(X.size(0), device=self.device, dtype=torch.float32)
        for stump, alpha in zip(self.stumps, self.alphas):
            pred = torch.where(X[:, stump["feature_idx"]] <= stump["threshold"], -1.0, 1.0).to(self.device)
            margin += alpha * pred
        return margin

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(X)
        return probs.argmax(dim=1).long()

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        margin = self._margin(X)
        probs = torch.sigmoid(margin)
        return torch.stack([1.0 - probs, probs], dim=1).detach().cpu()


class LightGBMClassifier(BaseClassifier):
    def __init__(self, n_estimators: int = 8, learning_rate: float = 0.1, max_depth: int = 2, device: str = "cpu") -> None:
        super().__init__(device)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees: list[Dict[str, Any]] = []

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.device)
        y = y.to(self.device).float().view(-1)
        y_bin = (y > 0.5).float()
        logits = torch.full((X.size(0),), torch.log((y_bin.mean().clamp_min(1e-6) / (1.0 - y_bin.mean().clamp_min(1e-6)))), device=self.device)
        self.trees = []
        for _ in range(self.n_estimators):
            residuals = y_bin - torch.sigmoid(logits)
            stump = self._fit_regression_stump(X, residuals)
            leaf_value = stump["value"]
            update = torch.where(X[:, stump["feature_idx"]] <= stump["threshold"], leaf_value[0], leaf_value[1]).to(self.device)
            logits += self.learning_rate * update
            self.trees.append(stump)
        self.is_fitted_ = True

    def _fit_regression_stump(self, X: torch.Tensor, residuals: torch.Tensor) -> Dict[str, Any]:
        best_score = float("inf")
        best = None
        for feature_idx in range(X.size(1)):
            thresholds = torch.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                left_value = residuals[left_mask].mean().item() if left_mask.any() else 0.0
                right_value = residuals[right_mask].mean().item() if right_mask.any() else 0.0
                left_pred = torch.full((left_mask.sum(),), left_value, device=X.device)
                right_pred = torch.full((right_mask.sum(),), right_value, device=X.device)
                preds = torch.cat([left_pred, right_pred], dim=0)
                score = ((residuals - preds) ** 2).mean().item()
                if score < best_score:
                    best_score = score
                    best = {"feature_idx": feature_idx, "threshold": float(threshold.item()), "value": (left_value, right_value)}
        return best if best is not None else {"feature_idx": 0, "threshold": 0.0, "value": (0.0, 0.0)}

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).long()

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        logits = torch.zeros(X.size(0), device=self.device, dtype=torch.float32)
        for tree in self.trees:
            update = torch.where(X[:, tree["feature_idx"]] <= tree["threshold"], tree["value"][0], tree["value"][1]).to(self.device)
            logits += self.learning_rate * update
        probs = torch.sigmoid(logits)
        return torch.stack([1.0 - probs, probs], dim=1).detach().cpu()


class CatBoostClassifier(LightGBMClassifier):
    def __init__(self, n_estimators: int = 8, learning_rate: float = 0.1, max_depth: int = 2, device: str = "cpu") -> None:
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, device=device)


class QuadraticDiscriminantAnalysis(BaseClassifier):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device)
        self.class_means: Optional[torch.Tensor] = None
        self.class_covs: Optional[torch.Tensor] = None
        self.class_priors: Optional[torch.Tensor] = None
        self.classes: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.device)
        y = y.to(self.device).view(-1)
        self.classes = torch.unique(y)
        self.class_priors = torch.tensor([torch.sum(y == c).item() / len(y) for c in self.classes], device=self.device)
        self.class_means = torch.stack([X[y == c].mean(dim=0) for c in self.classes], dim=0)
        covs = []
        for c in self.classes:
            class_x = X[y == c]
            centered = class_x - self.class_means[self.classes.tolist().index(int(c.item()))]
            cov = centered.T @ centered / max(class_x.size(0) - 1, 1)
            covs.append(cov + torch.eye(cov.size(0), device=self.device) * 1e-6)
        self.class_covs = torch.stack(covs, dim=0)
        self.is_fitted_ = True

    def _log_prob(self, X: torch.Tensor) -> torch.Tensor:
        log_probs = torch.empty((X.size(0), self.class_covs.size(0)), device=self.device, dtype=torch.float32)
        for class_idx in range(self.class_covs.size(0)):
            cov = self.class_covs[class_idx]
            inv_cov = torch.linalg.inv(cov)
            diff = X - self.class_means[class_idx]
            mahal = torch.einsum("bi,ij,bj->b", diff, inv_cov, diff)
            log_det = torch.logdet(cov)
            log_probs[:, class_idx] = -0.5 * (diff.size(1) * math.log(2 * math.pi) + log_det + mahal)
        return log_probs + torch.log(self.class_priors).to(self.device)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(X)
        return probs.argmax(dim=1).long()

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        log_probs = self._log_prob(X)
        probs = torch.softmax(log_probs, dim=1)
        return probs.detach().cpu()


class LinearDiscriminantAnalysis(BaseClassifier):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device)
        self.class_means: Optional[torch.Tensor] = None
        self.shared_cov: Optional[torch.Tensor] = None
        self.class_priors: Optional[torch.Tensor] = None
        self.classes: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.device)
        y = y.to(self.device).view(-1)
        self.classes = torch.unique(y)
        self.class_priors = torch.tensor([torch.sum(y == c).item() / len(y) for c in self.classes], device=self.device)
        self.class_means = torch.stack([X[y == c].mean(dim=0) for c in self.classes], dim=0)
        pooled = []
        for c in self.classes:
            class_x = X[y == c]
            centered = class_x - self.class_means[self.classes.tolist().index(int(c.item()))]
            pooled.append(centered.T @ centered)
        self.shared_cov = sum(pooled) / max(len(y) - 1, 1) + torch.eye(X.size(1), device=self.device) * 1e-6
        self.shared_cov = self.shared_cov / self.shared_cov.size(0)
        self.is_fitted_ = True

    def _log_prob(self, X: torch.Tensor) -> torch.Tensor:
        inv_cov = torch.linalg.inv(self.shared_cov)
        log_probs = torch.empty((X.size(0), self.class_means.size(0)), device=self.device, dtype=torch.float32)
        for class_idx in range(self.class_means.size(0)):
            diff = X - self.class_means[class_idx]
            mahal = torch.einsum("bi,ij,bj->b", diff, inv_cov, diff)
            log_probs[:, class_idx] = -0.5 * mahal
        return log_probs + torch.log(self.class_priors).to(self.device)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(X)
        return probs.argmax(dim=1).long()

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        log_probs = self._log_prob(X)
        probs = torch.softmax(log_probs, dim=1)
        return probs.detach().cpu()
