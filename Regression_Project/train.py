import os
import torch
from config import DEVICE, RANDOM_SEED, BATCH_SIZE
from utils import set_seed, train_test_split, StandardScaler
from datasets import make_sine, make_spiral, make_polynomial
from algorithm import (
    LinearRegression, SGDRegression, PolynomialRegression, 
    DecisionTreeRegression, RandomForestRegression, KNNRegression, 
    SupportVectorRegression, NeuralNetworkRegression, XGBoostRegression
)
from metrics import mse, rmse, mae, r2_score, explained_variance
from visualization import plot_prediction, plot_residuals

# Setup directories
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def evaluate_model(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """Computes all metrics natively."""
    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2 Score": r2_score(y_true, y_pred),
        "Explained Var": explained_variance(y_true, y_pred)
    }

def run_experiment(model_name: str, model, X: torch.Tensor, y: torch.Tensor):
    """End-to-end training and evaluation pipeline for a single model."""
    print(f"\n{'='*40}")
    print(f"Training: {model_name}")
    print(f"{'='*40}")
    
    # 1. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 2. Scaling (Trees and RF don't strictly need it, but it helps Neural Nets/KNN/SVR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X) # For plotting full curve
    
    # 3. Fit
    model.fit(X_train_scaled, y_train)
    
    # 4. Predict
    y_pred_test = model.predict(X_test_scaled)
    y_pred_full = model.predict(X_scaled)
    
    # 5. Evaluate
    metrics = evaluate_model(y_test, y_pred_test)
    print(f"Results for {model_name}:")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
        
    # 6. Save Model
    model_path = f"models/{model_name.replace(' ', '_')}.pth"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # 7. Visualize
    plot_prediction(X, y, y_pred_full, title=model_name, save_path=f"plots/{model_name.replace(' ', '_')}_pred.png")
    plot_residuals(y, y_pred_full, save_path=f"plots/{model_name.replace(' ', '_')}_res.png")
    
    return metrics

if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    
    # Generate non-linear data (Sine wave) to test algorithm flexibility
    X, y = make_sine(n_samples=300, noise=0.15)
    
    # Dictionary of all models
    models = {
        "Linear Regression": LinearRegression(device=DEVICE),
        "Polynomial Regression (Deg 3)": PolynomialRegression(degree=3, device=DEVICE),
        "SGD Regression": SGDRegression(lr=0.01, epochs=500, device=DEVICE),
        "Decision Tree": DecisionTreeRegression(max_depth=4, device=DEVICE),
        "Random Forest": RandomForestRegression(n_estimators=20, max_depth=4, device=DEVICE),
        "KNN Regression": KNNRegression(k=5, device=DEVICE),
        "SVR": SupportVectorRegression(epsilon=0.1, C=1.0, epochs=500, device=DEVICE),
        "Neural Network": NeuralNetworkRegression(hidden_dim=64, epochs=800, device=DEVICE),
        "XGBoost Regression": XGBoostRegression(n_estimators=30, learning_rate=0.1, max_depth=3, device=DEVICE)
    }
    
    all_metrics = {}
    for name, model in models.items():
        metrics = run_experiment(name, model, X, y)
        all_metrics[name] = metrics
        
    print("\nAll Training Complete.")