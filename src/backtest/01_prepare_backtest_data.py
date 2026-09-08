import pandas as pd
import yfinance as yf
import os

# 1. Paths
master_path = os.path.join('data/processed/master_dataset.csv')
backtest_path = os.path.join('data/processed/backtest_dataset.csv')

# 2. Load the Master Dataset
df_master = pd.read_csv(master_path)
df_master['Date'] = pd.to_datetime(df_master['Date'])

# 3. Fetch Nifty 50 Price Data (^NSEI)
# We use the min/max dates from your existing dataset
start_date = df_master['Date'].min().strftime('%Y-%m-%d')
end_date = df_master['Date'].max().strftime('%Y-%m-%d')

print(f"Fetching Nifty 50 data from {start_date} to {end_date}...")
nifty = yf.download('^NSEI', start=start_date, end=end_date)

# Reset index so Date is a column and keep only 'Close'
nifty = nifty[['Close']].reset_index()
nifty.columns = ['Date', 'Nifty_Close']
nifty['Date'] = pd.to_datetime(nifty['Date'])

# 4. Merge
# We use 'outer' to ensure we have all trading days, not just RBI days
df_backtest = pd.merge(nifty, df_master, on='Date', how='left')

# 5. Data Cleanup for Backtesting
# Fill Sentiment_Score with 0 for non-policy days (so we don't 'exit' on random days)
df_backtest['Sentiment_Score'] = df_backtest['Sentiment_Score'].fillna(0)
df_backtest = df_backtest.sort_values('Date')

# 6. Save as a SEPARATE file
df_backtest.to_csv(backtest_path, index=False)

print(f"Successfully created: {backtest_path}")
print(f"Original master_dataset.csv remains unchanged.")
