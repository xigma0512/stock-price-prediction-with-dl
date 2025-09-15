import yfinance as yf

df = yf.Ticker("2330.TW").history(period="max")
df.to_csv('price_2330.csv', sep=',')