"""
algorithm.py
Implementation of 9 distinct regression algorithms using Python and PyTorch from scratch.
"""

import torch
import torch.nn as nn
import os
import pickle
from typing import Dict, Any, Optional

# Import metrics to use in the score() method
from metrics import mse, r2_score

class BaseRegressor:
    """
    Base class for all regression models. Enforces a consistent scikit-learn-like API.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        raise NotImplementedError

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Returns the R^2 score by default."""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def save(self, path: str) -> None:
        """Saves the model state or object."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(self.model, nn.Module):
            torch.save(self.model.state_dict(), path)
        else:
            with open(path, 'wb') as f:
                pickle.dump(self.__dict__, f)

    def load(self, path: str) -> None:
        """Loads the model state or object."""
        if isinstance(self.model, nn.Module):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            with open(path, 'rb') as f:
                self.__dict__.update(pickle.load(f))



class LinearRegression(BaseRegressor):
    """
    Ordinary Least Squares (OLS) Linear Regression.
    
    Mathematical Formulation:
    $ \hat{\beta} = (X^T X)^{-1} X^T y $
    
    Intuition: Finds the hyperplane that minimizes the sum of squared distances 
    between the observed and predicted values.
    
    Optimization Objective: Minimize $ \sum (y_i - x_i^T \beta)^2 $
    
    Computational Complexity: $ O(nd^2 + d^3) $ for exact inverse (n=samples, d=features).
    Advantages: Fast, interpretable, deterministic.
    Disadvantages: Assumes linear relationship; sensitive to outliers.
    """
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.weights = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Add bias term (column of 1s)
        ones = torch.ones(X.size(0), 1, device=self.device)
        X_b = torch.cat([ones, X], dim=1)
        
        # Compute exact closed-form solution using pseudo-inverse for stability
        # torch.linalg.pinv handles singular matrices better than direct inverse
        X_pinv = torch.linalg.pinv(X_b)
        self.weights = X_pinv @ y

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        ones = torch.ones(X.size(0), 1, device=self.device)
        X_b = torch.cat([ones, X], dim=1)
        return X_b @ self.weights


class SGDRegression(BaseRegressor):
    """
    Linear Regression optimized via Stochastic/Mini-Batch Gradient Descent.
    
    Mathematical Formulation:
    $ \beta := \beta - \eta \nabla_\beta L(\beta) $
    where $ L(\beta) = \frac{1}{2n} \sum (y_i - \hat{y}_i)^2 $
    
    Intuition: Iteratively updates weights by calculating the gradient of the loss 
    using autograd on small subsets of data.
    
    Computational Complexity: $ O(n \cdot d \cdot epochs) $
    Advantages: Scales well to massive datasets that don't fit in memory.
    """
    def __init__(self, lr: float = 0.01, epochs: int = 1000, batch_size: int = 32, device: str = "cpu"):
        super().__init__(device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None # Will be initialized in fit

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X, y = X.to(self.device), y.to(self.device)
        num_features = X.size(1)
        
        # Use PyTorch's native linear layer
        self.model = nn.Linear(num_features, 1).to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            # Manual Mini-batching
            indices = torch.randperm(X.size(0))
            for start_idx in range(0, X.size(0), self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                X_batch, y_batch = X[batch_idx], y[batch_idx]

                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X)


class PolynomialRegression(LinearRegression):
    """
    Polynomial Regression mapping features to higher dimensions.
    
    Mathematical Formulation:
    Maps $ x \rightarrow [1, x, x^2, ..., x^d] $, then applies OLS.
    $ \hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_d x^d $
    
    Intuition: By engineering polynomial features, a linear model can fit non-linear curves.
    Advantages: Simple extension of linear models to capture curvature.
    Disadvantages: High degrees cause severe overfitting (Runge's phenomenon).
    """
    def __init__(self, degree: int = 3, device: str = "cpu"):
        super().__init__(device)
        self.degree = degree

    def _transform_features(self, X: torch.Tensor) -> torch.Tensor:
        """Expands X into polynomial features up to self.degree."""
        features = [X ** i for i in range(1, self.degree + 1)]
        return torch.cat(features, dim=1)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X_poly = self._transform_features(X)
        super().fit(X_poly, y)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X_poly = self._transform_features(X)
        return super().predict(X_poly)


class DecisionTreeRegression(BaseRegressor):
    """
    Decision Tree Regression using variance reduction (CART algorithm).
    
    Mathematical Formulation:
    Splits data by maximizing Variance Reduction (VR):
    $ VR = Var(S) - ( \frac{|S_L|}{|S|} Var(S_L) + \frac{|S_R|}{|S|} Var(S_R) ) $
    
    Intuition: Recursively partitions the input space into axis-aligned rectangles.
    Prediction is the mean of target values in the leaf node.
    
    Complexity: $ O(n \log n \cdot d) $ for training per depth.
    Advantages: Highly interpretable, non-linear, handles unscaled data well.
    Disadvantages: Prone to overfitting (high variance).
    """
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2, device: str = "cpu"):
        super().__init__(device)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.tree = self._grow_tree(X.cpu(), y.cpu(), depth=0)

    def _grow_tree(self, X: torch.Tensor, y: torch.Tensor, depth: int) -> dict:
        num_samples, num_features = X.shape
        # Stopping criteria
        if depth >= self.max_depth or num_samples < self.min_samples_split or torch.var(y) == 0:
            return {'is_leaf': True, 'value': torch.mean(y).item()}

        best_split = self._find_best_split(X, y, num_features)
        
        if not best_split:
            return {'is_leaf': True, 'value': torch.mean(y).item()}

        left_subtree = self._grow_tree(best_split['X_left'], best_split['y_left'], depth + 1)
        right_subtree = self._grow_tree(best_split['X_right'], best_split['y_right'], depth + 1)

        return {
            'is_leaf': False,
            'feature_idx': best_split['feature_idx'],
            'threshold': best_split['threshold'],
            'left': left_subtree,
            'right': right_subtree
        }

    def _find_best_split(self, X: torch.Tensor, y: torch.Tensor, num_features: int) -> dict:
        best_vr = -float('inf')
        best_split = {}
        base_var = torch.var(y, unbiased=False) * len(y)

        for feature_idx in range(num_features):
            # Try splitting on all unique values of the feature
            thresholds = torch.unique(X[:, feature_idx])
            for thresh in thresholds:
                left_mask = X[:, feature_idx] <= thresh
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_l, y_r = y[left_mask], y[right_mask]
                var_l = torch.var(y_l, unbiased=False) * len(y_l) if len(y_l) > 1 else 0
                var_r = torch.var(y_r, unbiased=False) * len(y_r) if len(y_r) > 1 else 0
                
                # Variance Reduction
                vr = base_var - (var_l + var_r)

                if vr > best_vr:
                    best_vr = vr
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': thresh,
                        'X_left': X[left_mask], 'y_left': y_l,
                        'X_right': X[right_mask], 'y_right': y_r
                    }
                    
        return best_split if best_vr > 0 else None

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.cpu()
        preds = [self._predict_single(x, self.tree) for x in X]
        return torch.tensor(preds, dtype=torch.float32, device=self.device).view(-1, 1)

    def _predict_single(self, x: torch.Tensor, tree: dict) -> float:
        if tree['is_leaf']:
            return tree['value']
        
        if x[tree['feature_idx']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])


class RandomForestRegression(BaseRegressor):
    """
    Ensemble of Decision Trees via Bagging (Bootstrap Aggregating).
    
    Mathematical Formulation:
    $ \hat{y} = \frac{1}{B} \sum_{b=1}^{B} f_b(x) $
    
    Intuition: Trains multiple deep trees on random subsets of the data (with replacement).
    Averaging their predictions reduces the overall variance without increasing bias.
    
    Advantages: Robust to overfitting, handles missing data, highly accurate.
    """
    def __init__(self, n_estimators: int = 10, max_depth: int = 5, device: str = "cpu"):
        super().__init__(device)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.trees = []
        n_samples = X.size(0)
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling (sampling with replacement)
            indices = torch.randint(0, n_samples, (n_samples,))
            X_sample, y_sample = X[indices], y[indices]
            
            tree = DecisionTreeRegression(max_depth=self.max_depth, device=self.device)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        # Stack predictions from all trees and take the mean across dimension 0
        predictions = torch.stack([tree.predict(X) for tree in self.trees])
        return torch.mean(predictions, dim=0)


class KNNRegression(BaseRegressor):
    """
    Instance-based learning using distance metrics.
    
    Mathematical Formulation:
    $ \hat{y} = \frac{1}{K} \sum_{i \in N_k(x)} y_i $
    where $ N_k(x) $ is the set of K nearest points in training data based on Euclidean distance.
    
    Intuition: "Show me who your friends are, and I'll tell you who you are."
    Predicts based on the average of the k closest data points.
    
    Advantages: No explicit training phase, highly non-linear.
    Disadvantages: $ O(nd) $ inference time, memory intensive.
    """
    def __init__(self, k: int = 5, device: str = "cpu"):
        super().__init__(device)
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        # KNN simply memorizes the training data
        self.X_train = X.to(self.device)
        self.y_train = y.to(self.device)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        # Compute pairwise Euclidean distance matrix
        # Output shape: [len(X_test), len(X_train)]
        distances = torch.cdist(X, self.X_train, p=2.0)
        
        # Get indices of the top K smallest distances
        # topk returns largest by default, so we set largest=False
        top_k_indices = torch.topk(distances, self.k, largest=False, dim=1).indices
        
        # Gather target values of the K nearest neighbors and average them
        k_nearest_targets = self.y_train[top_k_indices] # shape: [len(X), K, 1]
        return torch.mean(k_nearest_targets, dim=1)


class SupportVectorRegression(BaseRegressor):
    """
    Support Vector Regression using primal formulation and SGD.
    
    Mathematical Formulation (Epsilon-Insensitive Loss):
    $ L(\beta) = C \max(0, |y - x^T\beta| - \epsilon) + \frac{1}{2}||\beta||^2 $
    
    Intuition: Finds a "tube" of radius epsilon around the true data. Errors inside 
    the tube are ignored (0 loss). Errors outside are penalized linearly.
    
    Assumptions: The data must be appropriately scaled to map features to the kernel space effectively.
    """
    def __init__(self, epsilon: float = 0.1, C: float = 1.0, lr: float = 0.01, epochs: int = 1000, device: str = "cpu"):
        super().__init__(device)
        self.epsilon = epsilon
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.model = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X, y = X.to(self.device), y.to(self.device)
        self.model = nn.Linear(X.size(1), 1).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            y_pred = self.model(X)
            
            # Epsilon-Insensitive Loss Calculation
            abs_diff = torch.abs(y_pred - y)
            hinge_loss = torch.clamp(abs_diff - self.epsilon, min=0.0)
            
            # L2 Regularization on weights
            l2_reg = 0.5 * torch.sum(self.model.weight ** 2)
            
            loss = self.C * torch.mean(hinge_loss) + l2_reg
            loss.backward()
            optimizer.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device))



class NeuralNetworkRegression(BaseRegressor):
    """
    Multi-Layer Perceptron (MLP) for Regression.
    
    Mathematical Formulation:
    $ \hat{y} = W_2 \sigma(W_1 x + b_1) + b_2 $ (for 1 hidden layer)
    
    Intuition: Uses hidden layers with non-linear activation functions (ReLU) to 
    learn complex representation manifolds. Optimized via backpropagation (Chain Rule).
    
    Advantages: Universal approximator, handles massive datasets, highly flexible.
    Disadvantages: Requires tuning, data hungry, acts as a "black box".
    """
    def __init__(self, hidden_dim: int = 64, lr: float = 0.01, epochs: int = 1000, device: str = "cpu"):
        super().__init__(device)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.model = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X, y = X.to(self.device), y.to(self.device)
        
        self.model = nn.Sequential(
            nn.Linear(X.size(1), self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self.model(X), y)
            loss.backward()
            optimizer.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device))



class XGBoostRegression(BaseRegressor):
    """
    Gradient Boosting Machine (Foundation of XGBoost) via Additive Modeling.
    
    Mathematical Formulation:
    $ F_m(x) = F_{m-1}(x) + \nu \cdot \gamma_m h_m(x) $
    Fits subsequent trees to the negative gradient of the loss function (pseudo-residuals).
    For MSE, negative gradient is exactly $ y - \hat{y} $.
    
    Intuition: "Golf analogy." The first tree gets the ball near the hole. The second 
    tree putts from there. Sequential error correction.
    
    Advantages: State-of-the-art accuracy on tabular data.
    """
    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1, max_depth: int = 3, device: str = "cpu"):
        super().__init__(device)
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        # Base prediction is the mean of the target
        self.initial_prediction = torch.mean(y).item()
        current_predictions = torch.full_like(y, self.initial_prediction)
        
        self.trees = []
        for _ in range(self.n_estimators):
            # Calculate negative gradient for MSE (Residuals)
            residuals = y - current_predictions
            
            # Fit a decision tree to the residuals
            tree = DecisionTreeRegression(max_depth=self.max_depth, device=self.device)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Update predictions
            update = tree.predict(X).to(self.device)
            current_predictions += self.lr * update

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        # Start with the base prediction
        preds = torch.full((X.size(0), 1), self.initial_prediction, device=self.device)
        
        # Iteratively add scaled tree predictions
        for tree in self.trees:
            preds += self.lr * tree.predict(X).to(self.device)
            
        return preds