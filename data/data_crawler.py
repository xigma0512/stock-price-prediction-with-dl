import yfinance as yf

df = yf.Ticker("AAPL").history(period="max", interval="1d", start='2010-01-01')
df.to_csv('price_AAPL.csv', sep=',')