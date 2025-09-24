import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.data_loader import get_data_loader
from typing import Tuple


TRAIN_DATA_FILE = 'data/train_dataset.csv'
TEST_DATA_FILE = 'data/test_dataset.csv'

def get_preprocess_data(df: pd.DataFrame, n: int, val_size: float) -> Tuple[np.ndarray, ...]:
    feature_tags = ["High", "Low", "Close", "Volume"]
    labels = df['Close'].shift(-1); df = df.iloc[:-1]
    features = df[feature_tags].values
    
    seqs_X = sliding_window_view(features, (n, len(feature_tags)))
    seqs_X = np.squeeze(seqs_X, axis=1)
    seqs_y = labels[n-1:-1]

    seqs_X = np.array(seqs_X); seqs_y = np.array(seqs_y)

    X_raw, X_split_raw, y_raw, y_split_raw = train_test_split(seqs_X, seqs_y, test_size=val_size, shuffle=False)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw.reshape(-1, X_raw.shape[-1])).reshape(X_raw.shape)
    X_split_scaled = scaler_X.transform(X_split_raw.reshape(-1, X_split_raw.shape[-1])).reshape(X_split_raw.shape)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
    y_split_scaled = scaler_y.transform(y_split_raw.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, X_split_scaled, y_split_scaled

def test_data_preprocess(n: int, batch_size: int):
    test_dataset = pd.read_csv(TEST_DATA_FILE)
    df = test_dataset.reset_index(drop=True)
    
    feature_tags = ["High", "Low", "Close", "Volume"]
    labels = df['Close'].shift(-1); df = df.iloc[:-1]
    features = df[feature_tags].values
    
    seqs_X = sliding_window_view(features, (n, len(feature_tags)))
    seqs_X = np.squeeze(seqs_X, axis=1)
    seqs_y = labels[n-1:-1]

    X_test_raw = np.array(seqs_X); y_test_raw = np.array(seqs_y)
    
    scaler_X_test = StandardScaler()
    X_test_scaled = scaler_X_test.fit_transform(X_test_raw.reshape(-1, X_test_raw.shape[-1])).reshape(X_test_raw.shape)

    scaler_y_test = StandardScaler()
    y_test_scaled = scaler_y_test.fit_transform(y_test_raw.reshape(-1, 1)).flatten()

    test_loader = get_data_loader(X_test_scaled, y_test_scaled, batch_size)
    
    return test_loader, scaler_y_test

def train_data_preprocess(n: int, val_size: float, batch_size: int):
    train_dataset = pd.read_csv(TRAIN_DATA_FILE)
    tickers = train_dataset['Ticker'].unique().tolist()

    all_data = {
        'X_train': [], 'y_train': [],
        'X_val': [], 'y_val': []
    }

    for ticker_name in tickers:
        df_ticker = train_dataset[train_dataset['Ticker'] == ticker_name].reset_index(drop=True)
        X_train, y_train, X_val, y_val = get_preprocess_data(df_ticker, n, val_size)

        all_data['X_train'].append(X_train)
        all_data['y_train'].append(y_train)
        all_data['X_val'].append(X_val)
        all_data['y_val'].append(y_val)

    X_train_combined = np.concatenate(all_data['X_train'])
    y_train_combined = np.concatenate(all_data['y_train'])
    X_val_combined = np.concatenate(all_data['X_val'])
    y_val_combined = np.concatenate(all_data['y_val'])
    
    train_loader = get_data_loader(X_train_combined, y_train_combined, batch_size)
    val_loader = get_data_loader(X_val_combined, y_val_combined, batch_size)

    return train_loader, val_loader