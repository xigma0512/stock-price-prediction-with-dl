import os
import matplotlib.pyplot as plt

if not os.path.exists('results'):
    os.makedirs('results')

def plot_graph(name, train_values, val_values):

    plt.figure(figsize=(10, 6))
    plt.title(name)
    x_axis = range(1, len(train_values) + 1)
    
    plt.plot(x_axis, train_values, label='Training')
    plt.plot(x_axis, val_values, label='Validation')
    
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./results/{name}.png')
    plt.close()
