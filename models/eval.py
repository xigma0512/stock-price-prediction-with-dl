import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from utils.plot import plot_price_predictions

def eval(model, test_loader, scaler_y):
    model.eval()
    test_loss = 0.0
    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)

            loss = nn.MSELoss()(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
            y_true_test.extend(labels.flatten().tolist())
            y_pred_test.extend(outputs.flatten().tolist())

    y_true_test_orig = scaler_y.inverse_transform(np.array(y_true_test).reshape(-1, 1))
    y_pred_test_orig = scaler_y.inverse_transform(np.array(y_pred_test).reshape(-1, 1))

    r2 = r2_score(y_true_test_orig, y_pred_test_orig)
    mae = mean_absolute_error(y_true_test_orig, y_pred_test_orig)
    rmse = np.sqrt(root_mean_squared_error(y_true_test_orig, y_pred_test_orig))
    avg_test_loss = test_loss / len(test_loader.dataset)

    print("Evaluation Result:")
    print(f"- Loss: {avg_test_loss:.4f}")
    print(f"- R-squared: {r2:.4f}")
    print(f"- MAE: {mae:.2f}")
    print(f"- RMSE: {rmse:.2f}")

    plot_price_predictions(y_true_test_orig, y_pred_test_orig)