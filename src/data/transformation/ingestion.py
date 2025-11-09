import yfinance as yf


def fetch_stock_data(ticker: str, period: str = "1w", interval: str = "1d"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    return hist
