import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import signal

from utils.preprocess import get_data_loaders
from utils.plot import plot_graph
from models.lstm import LSTMModel

_interrupted = False

def interrupt_handler(sig, frame):
    global _interrupted
    _interrupted = True

def train(model, optimizer, num_epochs = 100):
    train_losses, val_losses = [], []
    train_r2, val_r2 = [], []

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

            y_true_train.extend(labels.flatten().tolist())
            y_pred_train.extend(outputs.flatten().tolist())

        avg_train_loss = running_loss / len(train_loader.dataset)
        train_r2_score = r2_score(y_true_train, y_pred_train)

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

        train_losses.append(avg_train_loss)
        train_r2.append(train_r2_score)

        val_losses.append(avg_val_loss)
        val_r2.append(val_r2_score)

        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, Train R2: {train_r2_score:.4f} || "
            f"Val Loss: {avg_val_loss:.4f}, Val R2: {val_r2_score:.4f}"
        )

    plot_graph('Loss', train_losses, val_losses)
    plot_graph('R-squared', train_r2, val_r2)
    return model


def eval(model):
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

    avg_test_loss = test_loss / len(test_loader.dataset)
    r2 = r2_score(y_true_test, y_pred_test)

    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print(f"R-squared: {r2:.4f}")

if __name__ == "__main__":
    
    signal.signal(signal.SIGINT, interrupt_handler)

    train_loader, val_loader, test_loader = get_data_loaders(n=60, test_size=0.2, val_size=0.1, batch_size=20)

    lstm_model = LSTMModel(input_size=5)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    trained_lstm_model = train(lstm_model, lstm_optimizer, num_epochs = 1000)
    eval(trained_lstm_model)