import yfinance as yf
import pandas as pd


def download_data():
    print("Downloading Nifty 50 and India VIX data...")

    # Tickers: ^NSEI is Nifty 50, ^INDIAVIX is India VIX
    # We start from Jan 2020 to match your RBI data
    data = yf.download(["^NSEI", "^INDIAVIX"], start="2020-01-01")

    # We only need the 'Close' prices
    market_df = data['Close'].reset_index()

    # Rename for clarity
    market_df.columns = ['Date', 'India_VIX', 'Nifty_50']

    # Save to raw
    market_df.to_csv("data/raw/market_data_raw.csv", index=False)
    print("Market data saved to data/raw/market_data_raw.csv")


if __name__ == "__main__":
    download_data()
