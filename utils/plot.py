import os
import matplotlib.pyplot as plt
import time

if not os.path.exists('results'):
    os.makedirs('results')

def plot_train_result(name, train_values, val_values):

    plt.figure(figsize=(10, 6))
    plt.title(name)
    x_axis = range(1, len(train_values) + 1)
    
    plt.plot(x_axis, train_values, label='Training')
    plt.plot(x_axis, val_values, label='Validation')
    
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./results/{name}_{time.time()}.png')
    plt.close()

def plot_price_predictions(y_true, y_pred):
    plt.figure(figsize=(12, 6))
    plt.title("Price History")
    
    plt.plot(y_true, label='Price History', color='blue')
    plt.plot(y_pred, label='Predictions', color='red')
    
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./results/predictions_{time.time()}.png')
    plt.close()