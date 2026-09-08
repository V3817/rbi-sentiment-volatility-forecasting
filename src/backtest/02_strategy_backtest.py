import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load Data
file_path = os.path.join('data/processed/backtest_dataset.csv')
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])

# 2. Calculate Market Returns
# fillna(0) is critical or the whole portfolio will result in 'NaN'
df['Nifty_Returns'] = df['Nifty_Close'].pct_change().fillna(0)

# 3. Strategy Parameters
THRESHOLD = 0.4
df['Abs_Sentiment'] = df['Sentiment_Score'].abs()

# 4. Strategy Logic (The "Exit" Signal)
# Step A: Identify the shock days (1 if high intensity, 0 otherwise)
df['Signal'] = (df['Abs_Sentiment'] > THRESHOLD).astype(int)

# Step B: Create the Exit Market flag
# Instead of '|', we use addition on shifted columns and fill NaNs with 0
# s1 is 'was yesterday an RBI day?', s2 is 'was the day before an RBI day?'
s1 = df['Signal'].shift(1).fillna(0)
s2 = df['Signal'].shift(2).fillna(0)

# If either yesterday or day before was 1, the sum will be > 0
df['Exit_Market'] = ((s1 + s2) > 0).astype(int)

# Step C: Apply Strategy
# If Exit_Market is 1, return is 0 (Cash). Otherwise, take the Nifty Return.
df['Strategy_Returns'] = np.where(
    df['Exit_Market'] == 1, 0, df['Nifty_Returns'])

# 5. Performance Metrics Function


def get_metrics(returns):
    cum_ret = (1 + returns).cumprod()

    # Sharpe Ratio (Annualized)
    sharpe = (returns.mean() / returns.std()) * \
        np.sqrt(252) if returns.std() != 0 else 0

    # Max Drawdown
    peak = cum_ret.cummax()
    dd = (cum_ret - peak) / peak

    return cum_ret, (cum_ret.iloc[-1] - 1), dd.min(), sharpe


# 6. Execute Calculation
c_base, r_base, m_base, s_base = get_metrics(df['Nifty_Returns'])
c_strat, r_strat, m_strat, s_strat = get_metrics(df['Strategy_Returns'])

# 7. Print Results
print("="*50)
print(f"BACKTEST RESULTS (Threshold: {THRESHOLD})")
print("="*50)
print(f"{'Metric':<15} | {'Buy & Hold':<12} | {'Strategy'}")
print("-" * 50)
print(f"{'Total Return':<15} | {r_base*100:<12.2f}% | {r_strat*100:.2f}%")
print(f"{'Max Drawdown':<15} | {m_base*100:<12.2f}% | {m_strat*100:.2f}%")
print(f"{'Sharpe Ratio':<15} | {s_base:<12.2f} | {s_strat:.2f}")
print("="*50)

# 8. Visualization
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], c_base * 100,
         label='Nifty 50 (Buy & Hold)', color='gray', alpha=0.4)
plt.plot(df['Date'], c_strat * 100,
         label='Sentiment Shield Strategy', color='blue', linewidth=2)
plt.title(f'Equity Curve: Avoiding RBI Shocks (Threshold {THRESHOLD})')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()
