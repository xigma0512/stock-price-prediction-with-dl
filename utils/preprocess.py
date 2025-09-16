import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import TensorDataset, DataLoader

DATA_FILE = "data/price_AAPL.csv"

def data_preprocessing(n=100, test_size=0.2, val_size=0.1):
    df = pd.read_csv(DATA_FILE)
    df = df.sort_values(by="Date").reset_index(drop=True)
    df = df.ffill()
    
    df['label'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)
    df = df.iloc[:-1]
    features = ["Open", "High", "Low", "Close"]
    ohlc = df[features].values
    
    X, y = [], []
    for i in range(len(ohlc) - n):
        ohlc_group = ohlc[i:i+n]
        group_flattened = ohlc_group.flatten()
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_group = scaler.fit_transform(group_flattened.reshape(-1, 1)).flatten()
        X.append(normalized_group.reshape(n, 4))
        y.append(df['label'].iloc[i+n-1])

    X, y = np.array(X), np.array(y)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, shuffle=False)
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_ratio, shuffle=False)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_data_loaders(n=100, test_size=0.2, val_size=0.1, batch_size = 64):
    X_train, y_train, X_val, y_val, X_test, y_test = data_preprocessing(n, test_size, val_size)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader