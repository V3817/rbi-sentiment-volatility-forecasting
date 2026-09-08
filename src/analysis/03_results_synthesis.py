import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# 1. Load Data
# Moving up two levels from src/analysis/ to root
file_path = os.path.join('data/processed/master_dataset.csv')
df = pd.read_csv(file_path)
df = df.dropna(subset=['Sentiment_Score', 'VIX_Returns'])

# 2. Re-run the Winning Model (Intensity vs. Shock)
df['Abs_Sentiment'] = df['Sentiment_Score'].abs()
vix_std = df['VIX_Returns'].std()
df['Volatility_Shock'] = (df['VIX_Returns'].abs() > vix_std).astype(int)

X = df['Abs_Sentiment']
X = sm.add_constant(X)
y = df['Volatility_Shock']
logit_model = sm.Logit(y, X).fit(disp=0)

# 3. Calculate Odds Ratio
# Use .iloc to safely access by position: [0] is Intercept, [1] is Abs_Sentiment
intercept_coeff = logit_model.params.iloc[0]
slope_coeff = logit_model.params.iloc[1]
p_value = logit_model.pvalues.iloc[1]
odds_ratio = np.exp(slope_coeff)

# 4. Final Executive Summary Output
print("="*50)
print("FINAL QUANTITATIVE RESEARCH SUMMARY")
print("="*50)
print(f"Project: RBI Sentiment vs. Nifty VIX")
print(f"Sample Size: {len(df)} policy statements")

# Logic to determine significance string
sig_status = "YES" if p_value < 0.05 else "NO"
print(f"Statistically Significant: {sig_status} (p = {p_value:.4f})")
print(f"Direction: Positive (Higher intensity = Higher shock probability)")
print("-"*50)
print(f"THE ODDS RATIO: {odds_ratio:.2f}")
print(f"EXPLANATION: For every 1-unit increase in Speech Intensity,")
print(
    f"the odds of a Volatility Shock increase by {((odds_ratio-1)*100):.1f}%.")
print("="*50)

# 5. Create a 'Probability Table' for the Report


def calc_prob(sentiment_level):
    # Formula: p = 1 / (1 + exp(-(const + coeff * x)))
    logit_val = intercept_coeff + (slope_coeff * sentiment_level)
    return 1 / (1 + np.exp(-logit_val))


print("\nSHOCK PROBABILITY TABLE (For your report)")
print(f"{'Sentiment Intensity':<25} | {'Prob. of VIX Shock':<20}")
print("-" * 50)
# We test intensities from 0 (Neutral) to 0.8 (Very Intense)
for level in [0.0, 0.2, 0.4, 0.6, 0.8]:
    prob = calc_prob(level)
    print(f"{level:<25.1f} | {prob*100:<19.2f}%")
