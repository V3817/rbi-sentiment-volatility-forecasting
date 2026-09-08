import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import os

# 1. Load the dataset using the relative path from the 'src' folder
# .. moves up to the root, then into data/processed
file_path = os.path.join('data', 'processed', 'master_dataset.csv')

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded dataset from: {file_path}")
except FileNotFoundError:
    print(
        f"Error: Could not find the file at {file_path}. Ensure you are running the script from the 'src' directory.")
    exit()

# Convert Date and sort
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Drop NaN values for statistical accuracy
df_clean = df.dropna(subset=['Sentiment_Score', 'VIX_Returns'])

# 2. Correlation Analysis
pearson_corr, p_val_p = pearsonr(
    df_clean['Sentiment_Score'], df_clean['VIX_Returns'])
spearman_corr, p_val_s = spearmanr(
    df_clean['Sentiment_Score'], df_clean['VIX_Returns'])

print("\n" + "="*40)
print("CORRELATION ANALYSIS")
print("="*40)
print(f"Pearson (Linear): {pearson_corr:.4f} (p-value: {p_val_p:.4f})")
print(f"Spearman (Rank):   {spearman_corr:.4f} (p-value: {p_val_s:.4f})")
print("="*40)

# 3. Visualization: Dual-Axis Line Chart (Sentiment vs VIX_Close)
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Sentiment Score', color=color)
ax1.plot(df_clean['Date'], df_clean['Sentiment_Score'],
         color=color, marker='o', label='Sentiment')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('VIX Close', color=color)
ax2.plot(df_clean['Date'], df_clean['VIX_Close'],
         color=color, linestyle='--', label='VIX Index')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('RBI Monetary Policy Sentiment vs. Nifty VIX over Time')
fig.tight_layout()
plt.show()

# 4. Visualization: Scatter Plot (Sentiment vs VIX Returns)
plt.figure(figsize=(10, 6))
sns.regplot(data=df_clean, x='Sentiment_Score', y='VIX_Returns',
            scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
plt.title('Scatter Plot: Sentiment Score vs. VIX Daily Returns')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 5. OLS Regression Analysis
X = df_clean['Sentiment_Score']
y = df_clean['VIX_Returns']
X = sm.add_constant(X)  # Adds an intercept (alpha) to the model

model = sm.OLS(y, X).fit()

print("\nOLS REGRESSION SUMMARY")
print(model.summary())

# 6. Lagged Analysis: Does Sentiment affect the NEXT day's VIX?
df_clean = df_clean.copy()
df_clean['Lagged_Sentiment'] = df_clean['Sentiment_Score'].shift(1)

# Drop the first row which will now be NaN
df_lagged = df_clean.dropna(subset=['Lagged_Sentiment', 'VIX_Returns'])

# Re-run Correlation
lagged_pearson, lag_p = pearsonr(
    df_lagged['Lagged_Sentiment'], df_lagged['VIX_Returns'])

print("\n" + "="*40)
print("LAGGED ANALYSIS (T-1 Sentiment vs T VIX)")
print("="*40)
print(f"Lagged Pearson: {lagged_pearson:.4f} (p-value: {lag_p:.4f})")
print("="*40)

if lag_p < 0.05:
    print("RESULT: Statistically Significant! Sentiment predicts the next day's move.")
else:
    print("RESULT: Still Insignificant. The market might be 'pricing in' the news immediately.")
