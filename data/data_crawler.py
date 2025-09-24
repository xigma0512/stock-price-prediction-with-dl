import numpy as np
import pandas as pd
import yfinance as yf

STOCK_LIST = [
    "AAPL",
    "GOOG",
    "MSFT",
    "TSLA",
    "META",
    "AMZN",
    "NVDA"
]

TEST_STOCK = "NFLX"
TEST_STOCK_PERIOD = "3y"

def get_ticker(name) -> pd.DataFrame:
    df = yf.Ticker(name).history(period="max", interval="1d", start='2015-01-01')
    for col in ['Close', 'High', 'Low', 'Open']:
        df[col] = np.floor(df[col] * 100) / 100
    df["Ticker"] = name
    df = df.ffill()

    return df

stocks_df = []
for name in STOCK_LIST:
    stocks_df.append(get_ticker(name))

all_stocks_df = pd.concat(stocks_df)
all_stocks_df = all_stocks_df.sort_values(by=['Ticker', 'Date']).reset_index()
all_stocks_df.to_csv('data/train_dataset.csv', sep=',')

test_stock_df = get_ticker(name)
test_stock_df = test_stock_df.sort_values(by=['Ticker', 'Date']).reset_index()
test_stock_df.to_csv('data/test_dataset.csv', sep=',')

print("done")