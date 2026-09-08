import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
import os

# 1. Setup Path (Moving up two levels from src/analysis/ to root)
file_path = os.path.join('data/processed/master_dataset.csv')

try:
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Sentiment_Score', 'VIX_Returns'])
    print(f"Loaded {len(df)} observations for Shock Analysis.")
except FileNotFoundError:
    print(f"Error: Path not found at {file_path}")
    exit()

# 2. Calculate 'Intensity' (Absolute Values)
# This tests: Does strong language (regardless of good/bad) move the market?
df['Abs_Sentiment'] = df['Sentiment_Score'].abs()
df['Abs_VIX_Returns'] = df['VIX_Returns'].abs()

# 3. Correlation: Speech Intensity vs. Move Magnitude
abs_corr, abs_p = pearsonr(df['Abs_Sentiment'], df['Abs_VIX_Returns'])

print("\n" + "="*40)
print("INTENSITY ANALYSIS (Magnitude vs Magnitude)")
print("="*40)
print(f"Correlation (Abs_Sent vs Abs_VIX): {abs_corr:.4f}")
print(f"P-Value: {abs_p:.4f}")

# 4. Create Volatility Dummy (The Shock Variable)
# Define a shock as a move > 1 Standard Deviation from the mean
vix_std = df['VIX_Returns'].std()
df['Volatility_Shock'] = (df['VIX_Returns'].abs() > vix_std).astype(int)

print(f"\nVolatility Threshold (1 SD): {vix_std:.4f}")
print(
    f"Number of Shocks identified: {df['Volatility_Shock'].sum()} out of {len(df)}")

# 5. Logistic Regression: Can Sentiment Intensity predict a Shock?
# y = Volatility_Shock (0 or 1), X = Abs_Sentiment
X = df['Abs_Sentiment']
X = sm.add_constant(X)
y = df['Volatility_Shock']

logit_model = sm.Logit(y, X).fit()

print("\nLOGISTIC REGRESSION SUMMARY")
print(logit_model.summary())

# 6. Visualizations
plt.figure(figsize=(14, 5))

# Plot A: Absolute Scatter
plt.subplot(1, 2, 1)
sns.regplot(data=df, x='Abs_Sentiment', y='Abs_VIX_Returns',
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'orange'})
plt.title('Speech Intensity vs. Market Move Size')

# Plot B: Boxplot (Distribution of Sentiment for Shocks vs Non-Shocks)
plt.subplot(1, 2, 2)
sns.boxplot(x='Volatility_Shock', y='Abs_Sentiment', data=df, palette='Set2')
plt.title('Sentiment Intensity: Normal vs. Shock Days')
plt.xticks([0, 1], ['Normal', 'Shock (1 SD+)'])

plt.tight_layout()
plt.show()
