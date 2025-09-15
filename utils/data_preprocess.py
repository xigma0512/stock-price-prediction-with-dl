import pandas
import numpy
from sklearn.model_selection import train_test_split

DATA_FILE = "data/price_2330.csv"

def preprocessing(n=20, test_size=0.2, val_size=0.1):
    df = pandas.read_csv(DATA_FILE)
    df = df.sort_values(by="Date").reset_index(drop=True)
    
    ohlc = df[["Open", "High", "Low", "Close"]].values
    print(ohlc)
    
    X, y = [], []
    for i in range(len(ohlc) - n - 1):
        X.append(ohlc[i:i+n])
        next_open, _, _, next_close = ohlc[i+n]
        label = 1 if (next_close - next_open) > 0 else 0
        y.append(label)
    
    X, y = numpy.array(X), numpy.array(y)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size+val_size, shuffle=False)
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-val_ratio, shuffle=False)

    return X_train, y_train, X_val, y_val, X_test, y_test