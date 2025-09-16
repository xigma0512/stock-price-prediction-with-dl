import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from utils.preprocess import data_preprocessing

X_train, y_train, X_val, y_val, X_test, y_test = data_preprocessing()

n_zeros = np.sum(y_train == 0)
n_ones = np.sum(y_train == 1)

if n_ones > n_zeros:
    most_frequent = 1
else:
    most_frequent = 0

predictions = np.full(y_test.shape, most_frequent)
accuracy = accuracy_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f'baseline accuracy score: {accuracy}')
print(f'baseline mean squared error: {mse}')