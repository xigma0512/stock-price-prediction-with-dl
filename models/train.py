from aiosignal import Signal
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from utils.plot import plot_train_result

import signal

_interrupted = False
def interrupt_handler(sig, frame):
    global _interrupted
    print("interrupt...")
    _interrupted = True

def train(model, optimizer, train_loader, val_loader, num_epochs = 100):
    signal.signal(signal.SIGINT, interrupt_handler)

    train_losses, val_losses = [], []
    train_r2, val_r2 = [], []
    train_mae, val_mae = [], []
    train_rmse, val_rmse = [], []

    for epoch in range(num_epochs):

        if _interrupted: break

        model.train()
        running_loss = 0.0
        y_true_train, y_pred_train = [], []

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            y_true_train.extend(labels.flatten().tolist())
            y_pred_train.extend(outputs.flatten().tolist())

        avg_train_loss = running_loss / len(train_loader.dataset)
        train_r2_score = r2_score(y_true_train, y_pred_train)
        train_mae_score = mean_absolute_error(y_true_train, y_pred_train)
        train_rmse_score = np.sqrt(root_mean_squared_error(y_true_train, y_pred_train))

        model.eval()
        val_loss = 0.0
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = nn.MSELoss()(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                y_true_val.extend(labels.flatten().tolist())
                y_pred_val.extend(outputs.flatten().tolist())

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_r2_score = r2_score(y_true_val, y_pred_val)
        val_mae_score = mean_absolute_error(y_true_val, y_pred_val)
        val_rmse_score = np.sqrt(root_mean_squared_error(y_true_val, y_pred_val))

        train_losses.append(avg_train_loss)
        train_r2.append(train_r2_score)
        train_mae.append(train_mae_score)
        train_rmse.append(train_rmse_score)

        val_losses.append(avg_val_loss)
        val_r2.append(val_r2_score)
        val_mae.append(val_mae_score)
        val_rmse.append(val_rmse_score)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"{'Metric':<10} | {'Train':>10} | {'Validation':>10}")
        print(f"{'-----------':<10}|{'------------':>10}|{'------------':>10}")
        print(f"{'Loss':<10} | {avg_train_loss:>10.4f} | {avg_val_loss:>10.4f}")
        print(f"{'R2':<10} | {train_r2_score:>10.4f} | {val_r2_score:>10.4f}")
        print(f"{'MAE':<10} | {train_mae_score:>10.2f} | {val_mae_score:>10.2f}")
        print(f"{'RMSE':<10} | {train_rmse_score:>10.2f} | {val_rmse_score:>10.2f}")
        print(f"{'-----------':<10}-{'------------':>10}-{'------------':>10}")

    plot_train_result('Loss', train_losses, val_losses)
    plot_train_result('R-squared', train_r2, val_r2)
    plot_train_result('MAE', train_mae, val_mae)
    plot_train_result('RMSE', train_rmse, val_rmse)

    return model