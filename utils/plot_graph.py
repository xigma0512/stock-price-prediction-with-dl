import matplotlib.pyplot as plt

def plot_graph(train_history):
    history_dict = train_history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    # loss
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    plt.figure(1)
    plt.plot(epochs, loss_values, label='Training loss', color='blue', linestyle='-')
    plt.plot(epochs, val_loss_values, label='Validation loss', color='red', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # accuracy
    accuracy_values = history_dict['accuracy']
    val_accuracy_values = history_dict['val_accuracy']

    plt.figure(2)
    plt.plot(epochs, accuracy_values, label='Training accuracy', color='blue', linestyle='-')
    plt.plot(epochs, val_accuracy_values, label='Validation accuracy', color='red', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # show
    plt.legend()
    plt.show()