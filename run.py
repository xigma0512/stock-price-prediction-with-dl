import argparse
import torch
import torch.optim as optim

from models.train import train
from models.eval import eval
from data.preprocess import test_data_preprocess, train_data_preprocess
from models.lstm import LSTMModel

MODEL_SAVE_PATH = "results/model"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train or evaluate a pre-trained LSTM model.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use this flag to load a pre-trained model instead of training a new one.')
    args = parser.parse_args()

    if args.pretrained:
        print("Loading pre-trained model...")
        try:
            trained_model = LSTMModel(input_size=4)
            trained_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Pre-trained model not found at {MODEL_SAVE_PATH}. Please train a model first.")
            exit()
    else:
        train_loader, val_loader = train_data_preprocess(n=60, val_size=0.1, batch_size=20)

        lstm_model = LSTMModel(input_size=4)
        lstm_optimizer = optim.AdamW(lstm_model.parameters(), lr=0.001)
        trained_model = train(lstm_model, lstm_optimizer, train_loader, val_loader, num_epochs=200)
        torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)

    test_loader, scaler_y_test = test_data_preprocess(n=60, batch_size=20)
    eval(trained_model, test_loader, scaler_y_test)
