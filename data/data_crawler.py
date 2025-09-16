import yfinance as yf

df = yf.Ticker("AAPL").history(period="max")
df.to_csv('price_AAPL.csv', sep=',')