import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from utils.preprocess import get_data_loaders
from utils.plot import plot_graph
from models.lstm import LSTMModel

train_loader, val_loader, test_loader = get_data_loaders(n=100, test_size=0.2, val_size=0.1, batch_size=64)

def train(model, optimizer, num_epochs = 100):

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        y_true_train, y_pred_train = [], []

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = nn.BCELoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            preds = (outputs > 0.5).int().flatten()
            y_true_train.extend(labels.flatten().tolist())
            y_pred_train.extend(preds.tolist())

        avg_train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = accuracy_score(y_true_train, y_pred_train)

        model.eval()
        val_loss = 0.0
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = nn.BCELoss()(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                preds = (outputs > 0.5).int().flatten()
                y_true_val.extend(labels.flatten().tolist())
                y_pred_val.extend(preds.tolist())

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = accuracy_score(y_true_val, y_pred_val)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}")

    plot_graph('Loss', train_losses, val_losses)
    plot_graph('Accuracy', train_accuracies, val_accuracies)
    return model


def eval(model):
    model.eval()
    test_loss = 0.0
    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = nn.BCELoss()(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            preds = (outputs > 0.5).int().flatten()
            y_true_test.extend(labels.flatten().tolist())
            y_pred_test.extend(preds.tolist())

    avg_test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = accuracy_score(y_true_test, y_pred_test)

    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":

    lstm_model = LSTMModel(input_size=5)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    trained_lstm_model = train(lstm_model, lstm_optimizer, num_epochs = 100)
    eval(trained_lstm_model)