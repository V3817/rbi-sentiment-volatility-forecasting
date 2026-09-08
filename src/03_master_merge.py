import pandas as pd
import os

# 1. Load the datasets
# Running from root, so we use ./data/
sentiment_df = pd.read_csv('./data/processed/rbi_sentiment_scores.csv')
market_df = pd.read_csv('./data/raw/market_data_raw.csv')

# Clean column names
market_df.columns = market_df.columns.str.strip()

# 2. Rename India_VIX to VIX_Close for consistency
if 'India_VIX' in market_df.columns:
    market_df = market_df.rename(columns={'India_VIX': 'VIX_Close'})
    print("Found 'India_VIX', renamed to 'VIX_Close'")
else:
    print(
        f"Warning: Expected 'India_VIX' but found {market_df.columns.tolist()}")

# 3. Convert Date columns to datetime objects
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
market_df['Date'] = pd.to_datetime(market_df['Date'])

# 4. Sort by date (Required for merge_asof)
sentiment_df = sentiment_df.sort_values('Date')
market_df = market_df.sort_values('Date')

# 5. Feature Engineering: Calculate VIX Returns
# We use .pct_change() to see the % shift in fear
market_df['VIX_Returns'] = market_df['VIX_Close'].pct_change()

# 6. The Master Merge (Weekend/Holiday Logic)
# Maps sentiment to the NEXT available trading day
master_df = pd.merge_asof(
    sentiment_df,
    market_df,
    on='Date',
    direction='forward'
)

# 7. Cleanup & Aggregation
# Average sentiment if multiple docs on the same trading day
master_df = master_df.groupby(['Date', 'Document_Type']).agg({
    'Sentiment_Score': 'mean',
    'VIX_Close': 'first',
    'VIX_Returns': 'first'
}).reset_index()

# Drop future dates/missing values
master_df = master_df.dropna(subset=['VIX_Close', 'VIX_Returns'])

# 8. Save Final Master Dataset
output_dir = './data/processed/'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'master_dataset.csv')
master_df.to_csv(output_path, index=False)

print(f"\nSuccess! Master dataset created with {len(master_df)} rows.")
print(master_df.head())
