import pandas as pd
import yfinance as yf

stocks = [
    "AAPL",
    "GOOG",
    "MSFT",
    "TSLA",
    "META"
]

stocks_df = []
for name in stocks:
    df = yf.Ticker(name).history(period="max", interval="1d", start='2015-01-01')
    df["Ticker"] = name
    df = df.ffill()
    stocks_df.append(df)

all_stocks_df = pd.concat(stocks_df)
all_stocks_df = all_stocks_df.sort_values(by=['Ticker', 'Date']).reset_index()
all_stocks_df.to_csv('train_dataset.csv', sep=',')
print(all_stocks_df.head())