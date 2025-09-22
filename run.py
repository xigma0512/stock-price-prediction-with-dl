import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from utils.preprocess import get_data_loaders
from utils.plot import plot_train_result, plot_price_predictions
from models.lstm import LSTMModel

import signal

def train(model, optimizer, train_loader, val_loader, scaler_y, num_epochs = 100):
    train_losses, val_losses = [], []
    train_r2, val_r2 = [], []
    train_mape, val_mape = [], []
    
    for epoch in range(num_epochs):
        if _interrupted:
            print('interrupted')
            break

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

            y_true_train_orig = scaler_y.inverse_transform(labels.detach().cpu().numpy())
            y_pred_train_orig = scaler_y.inverse_transform(outputs.detach().cpu().numpy())
            
            y_true_train.extend(y_true_train_orig.flatten().tolist())
            y_pred_train.extend(y_pred_train_orig.flatten().tolist())

        avg_train_loss = running_loss / len(train_loader.dataset)
        train_r2_score = r2_score(y_true_train, y_pred_train)
        train_mape_score = mean_absolute_percentage_error(y_true_train, y_pred_train) * 100

        model.eval()
        val_loss = 0.0
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = nn.MSELoss()(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
    
                y_true_val_orig = scaler_y.inverse_transform(labels.detach().cpu().numpy())
                y_pred_val_orig = scaler_y.inverse_transform(outputs.detach().cpu().numpy())
                
                y_true_val.extend(y_true_val_orig.flatten().tolist())
                y_pred_val.extend(y_pred_val_orig.flatten().tolist())

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_r2_score = r2_score(y_true_val, y_pred_val)
        val_mape_score = mean_absolute_percentage_error(y_true_val, y_pred_val) * 100

        train_losses.append(avg_train_loss)
        train_r2.append(train_r2_score)
        train_mape.append(train_mape_score)

        val_losses.append(avg_val_loss)
        val_r2.append(val_r2_score)
        val_mape.append(val_mape_score)

        print(
            f"\nEpoch {epoch+1}/{num_epochs}: "
            f"\nTrain Loss: {avg_train_loss:.4f}, Train R2: {train_r2_score:.4f}, Train MAPE: {train_mape_score:.2f}% || "
            f"\nVal Loss: {avg_val_loss:.4f}, Val R2: {val_r2_score:.4f}, Val MAPE: {val_mape_score:.2f}%"
        )

    plot_train_result('Loss', train_losses, val_losses)
    plot_train_result('R-squared', train_r2, val_r2)
    plot_train_result('Mean-absolute-precentage-error(%)', train_mape, val_mape)

    return model


def eval(model, test_loader, scaler):
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

    y_true_test_orig = scaler.inverse_transform(np.array(y_true_test).reshape(-1, 1))
    y_pred_test_orig = scaler.inverse_transform(np.array(y_pred_test).reshape(-1, 1))

    r2 = r2_score(y_true_test_orig, y_pred_test_orig)
    mape = mean_absolute_percentage_error(y_true_test_orig, y_pred_test_orig) * 100
    avg_test_loss = test_loss / len(test_loader.dataset)

    print("Evaluation Result:")
    print(f"- Loss: {avg_test_loss:.4f}")
    print(f"- R-squared: {r2:.4f}")
    print(f"- Mean absolute percentage error: {mape:.2f}%")

    plot_price_predictions(y_true_test_orig, y_pred_test_orig)


_interrupted = False

def interrupt_handler(sig, frame):
    global _interrupted
    _interrupted = True

if __name__ == "__main__":
    signal.signal(signal.SIGINT, interrupt_handler)

    train_loader, val_loader, test_loader, scaler = get_data_loaders(
        n=60,
        test_size=0.2,
        val_size=0.1,
        batch_size=20
    )

    lstm_model = LSTMModel(input_size=4)
    lstm_optimizer = optim.AdamW(lstm_model.parameters(), lr=0.001)

    trained_lstm_model = train(lstm_model, lstm_optimizer, train_loader, val_loader, scaler, num_epochs=300)
    
    eval(trained_lstm_model, test_loader, scaler)