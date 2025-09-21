import yfinance as yf

stocks = [
    "AAPL",
    "GOOG",
    "MSFT",
    "TSLA",
    "META"
]

for name in stocks:
    df = yf.Ticker(name).history(period="max", interval="1d", start='2015-01-01')
    df.to_csv(f'{name}.csv', sep=',')