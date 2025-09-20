import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import TensorDataset, DataLoader

DATA_FILE = "data/price_AAPL.csv"

def vwap(df):
    v = df['Volume']
    typical_price = (df['Close'] + df['Low'] + df['High']).div(3).values
    vwap_series = (typical_price * v).cumsum() / v.cumsum()
    return vwap_series

def data_preprocessing(n, test_size, val_size):
    df = pd.read_csv(DATA_FILE)
    df = df.sort_values(by="Date").reset_index(drop=True)
    df = df.ffill()
    
    df['VWAP'] = vwap(df)
    df['Labels'] = df['Close'].shift(-1)
    df = df.iloc[:-1]

    df['Close'] = np.log1p(df['Close'])
    df['VWAP'] = np.log1p(df['VWAP'])
    df['Labels'] = np.log1p(df['Labels'])

    features_tags = ["Close", "VWAP"]
    features = df[features_tags].values
    
    X, y = [], []
    for i in range(len(features) - n):
        features_group = features[i:i+n]
        X.append(features_group)
        y.append(df['Labels'].iloc[i+n-1])

    X, y = np.array(X), np.array(y)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=ratio, shuffle=True)
    
    scaler_X = MinMaxScaler(feature_range=(0, 10))
    scaler_X.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train_scaled = scaler_X.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    scaler_y = MinMaxScaler(feature_range=(0, 10))
    scaler_y.fit(y_train.reshape(-1, 1))
    y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return (
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        X_test_scaled, y_test_scaled,
        scaler_X, scaler_y
    )


def get_data_loaders(n=100, test_size=0.2, val_size=0.1, batch_size = 64):
    X_train, y_train, X_val, y_val, X_test, y_test, _, scaler_y = data_preprocessing(n, test_size, val_size)

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

    return train_loader, val_loader, test_loader, scaler_y