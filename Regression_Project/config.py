"""
config.py
Contains all hyperparameters and global configurations for the regression project.
"""

import torch

# Global Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Training Hyperparameters
LR = 0.01
EPOCHS = 10000000000
BATCH_SIZE = 64
PATIENCE = 500 # For early stopping

# Algorithm-Specific Parameters
K = 5                  # For KNN
POLY_DEGREE = 3        # For Polynomial Regression
TREE_DEPTH = 3         # For Decision Tree, RF, XGBoost
NUM_TREES = 50         # For Random Forest and XGBoost
HIDDEN_DIM = 64        # For Neural Network (MLP)
SVR_EPSILON = 0.1      # For Support Vector Regression
SVR_C = 1.0            # Regularization for SVR
XGB_LR = 0.1           # Learning rate for Gradient Boosting