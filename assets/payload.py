import json
import yfinance as yf


if __name__ == "__main__":
    print("Fetching AAPL stock data...")
    stock = yf.Ticker("AAPL")
    hist = stock.history(period="90d", interval="1d")
    print(f"Retrieved {len(hist)} days of data")

    sequence = []
    for date, row in hist.iterrows():
        sequence.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
        })

    print(f"Processed {len(sequence)} data points")
    payload = {
        "sequence": sequence,
        "steps": 5
    }

    print("Writing payload to ./assets/payload.json")
    with open("./assets/payload.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("Payload written successfully")
