# RBI Sentinel — Monetary Policy Sentiment & Volatility Shock Forecasting

<p align="center">
  <img src="streamlit-app/logo.png" alt="RBI Sentinel" width="280"/>
</p>

<p align="center">
  <strong>Can the tone of an RBI Governor's speech predict a volatility shock in the Indian stock market?</strong>
</p>

<p align="center">
  <strong>                    Try it here: https://rbi-sentinel.streamlit.app/                           </strong>
</p>


<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-findings">Key Findings</a> •
  <a href="#data-collection">Data Collection</a> •
  <a href="#methodology">Methodology</a> •
  <a href="#model-evolution">Model Evolution</a> •
  <a href="#live-dashboard">Live Dashboard</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#getting-started">Getting Started</a>
</p>

---

## Overview

This project investigates the relationship between **Reserve Bank of India (RBI)** monetary policy communications and **India VIX** (volatility index) movements. It combines NLP-based sentiment analysis with market data to build a real-time volatility shock prediction system.

The core hypothesis: *Strongly-worded RBI communications — regardless of whether they are hawkish or dovish — precede larger-than-normal moves in the India VIX.*

### What This Project Does

1. **Mines & digitises** 140 RBI policy documents (Governor speeches + MPC minutes) from 2020–2026.
2. **Scores sentiment** on each document using [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) with a sliding-window chunking approach for long documents.
3. **Merges** sentiment scores with daily India VIX and Nifty 50 market data fetched via Yahoo Finance.
4. **Models** the probability of a VIX shock using a Random Forest classifier trained on 5 engineered features.
5. **Deploys** an interactive Streamlit dashboard (**RBI Sentinel**) that runs real-time inference on any pasted RBI communication.

---

## Key Findings

| Metric | Value |
|--------|-------|
| **Documents Analysed** | 140 (102 speeches + 38 MPC minutes) |
| **Date Range** | January 2020 – February 2026 |
| **Sentiment Score Range** | −0.4512 to +0.7419 |
| **Shock Threshold** | VIX return > 80th percentile of training period |
| **Model** | Random Forest (300 trees, max-depth 4, balanced class weights) |
| **Training / Test Split** | Chronological — train ≤ 2024, test ≥ 2025 |

### The Critical Insight — Intensity Over Direction

The initial OLS regression of raw sentiment score vs. VIX returns showed **no statistically significant linear relationship** (p > 0.05). Positive speeches didn't reliably lower VIX, and negative speeches didn't reliably raise it.

The breakthrough came from reframing the problem:

> **It's not whether the RBI sounds positive or negative — it's how _strongly_ they speak.**

Using the **absolute value of sentiment** (intensity) as the predictor and modelling VIX **shocks** (binary: did VIX move more than 1 SD?) via logistic regression yielded a statistically significant relationship. This insight carried forward into the final Random Forest model.

---

## Data Collection

### RBI Communications (140 PDFs)

All documents were manually downloaded from the [RBI website](https://www.rbi.org.in):

| Source | Count | Period | URL Section |
|--------|-------|--------|-------------|
| **Governor Speeches** | 102 | 2020–2026 | RBI → Press Releases → Speeches |
| **MPC Minutes** | 38 | 2020–2026 | RBI → Monetary Policy → MPC Minutes |

Each PDF was named with a date prefix (e.g., `2024-02-08_minutes.pdf`) and stored in `data/raw/speeches/` and `data/raw/minutes/`. Text extraction was performed using `pdfplumber`.

### Market Data

| Ticker | Source | Description |
|--------|--------|-------------|
| `^INDIAVIX` | Yahoo Finance | India VIX (fear gauge) |
| `^NSEI` | Yahoo Finance | Nifty 50 index |
| `^VIX` | Yahoo Finance | CBOE VIX (US volatility, used in live dashboard) |

Market data was fetched from January 2020 onwards. RBI communication dates that fall on weekends or holidays are forward-merged to the next available trading day using `pd.merge_asof(direction='forward')`.

---

## Methodology

### Pipeline Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  01a  Extract PDFs  │────▶│  02  FinBERT        │────▶│  03  Master Merge   │
│  (pdfplumber)       │     │  Sentiment Analysis │     │  (merge_asof)       │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                                                 │
         ┌─────────────────────┐                                 ▼
         │  01b  Market Data   │──────────────────────▶ master_dataset.csv
         │  (yfinance)         │
         └─────────────────────┘
                                                                 │
                                                                 ▼
                            ┌─────────────────────────────────────────────────┐
                            │  Analysis Layer                                 │
                            │  ├── 01  Baseline OLS (raw sentiment)          │
                            │  ├── 02  Volatility Shocks (abs sentiment)     │
                            │  └── 03  Results Synthesis (odds ratio)        │
                            └─────────────────────────────────────────────────┘
                                                                 │
                                                                 ▼
                            ┌─────────────────────────────────────────────────┐
                            │  Model Training                                 │
                            │  └── Random Forest Classifier                  │
                            │      5 features, StandardScaler, 80/20 chrono  │
                            └─────────────────────────────────────────────────┘
                                                                 │
                                                                 ▼
                            ┌─────────────────────────────────────────────────┐
                            │  Deployment: Streamlit Dashboard (RBI Sentinel)│
                            │  Live inference + market context               │
                            └─────────────────────────────────────────────────┘
```

### Sentiment Scoring (FinBERT)

Standard BERT models truncate inputs at 512 tokens. RBI documents are often 3,000–10,000+ tokens long. This project uses a **sliding-window chunking** approach:

1. Tokenise the full document (no truncation).
2. Split into **510-token chunks** (reserving 2 tokens for `[CLS]` and `[SEP]`).
3. Run each chunk through FinBERT independently.
4. **Average** the softmax probability vectors across all chunks.
5. Compute `Score = P(positive) − P(negative)`, yielding a value in `[−1, +1]`.

### Feature Engineering

The final model uses **5 features**, computed on each RBI communication day:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `Sentiment_Intensity` | `\|Sentiment_Score\|` — magnitude of policy tone (the key insight) |
| 2 | `VIX_Lag1` | Previous day's India VIX close (forward-filled for gaps) |
| 3 | `Nifty_5d_Ret` | `ln(Nifty_t / Nifty_{t−5})` — 5-day log return |
| 4 | `Nifty_5d_Std` | Rolling 5-day standard deviation of daily Nifty log returns |
| 5 | `Is_MPC` | Binary flag: 1 if MPC minutes, 0 if Governor speech |

### Target Variable

A **VIX Shock** is defined as a VIX return exceeding the **80th percentile** of training-period VIX returns. The threshold is computed **only on the training set** (≤ 2024) to prevent data leakage.

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=3,
    class_weight='balanced',   # handles class imbalance
    random_state=42
)
```

Features are standardised via `StandardScaler` fit on training data. The chronological train/test split ensures no future information leaks into training.

---

## Model Evolution

The project went through three analytical stages — each one building on the failures of the last:

### Stage 1: Baseline OLS (❌ Failed)

**Hypothesis:** Negative RBI sentiment → VIX goes up; positive → VIX goes down.

```
OLS: Sentiment_Score → VIX_Returns
Result: Pearson r ≈ −0.03, p > 0.05 — NOT significant
```

The raw directional relationship between sentiment and VIX returns was essentially zero. A lagged analysis (T−1 sentiment → T VIX) was also insignificant.

**Lesson:** The market doesn't care whether the RBI sounds positive or negative in a simple linear way.

### Stage 2: Volatility Shocks — The Abs-Value Fix (✅ Breakthrough)

**Hypothesis:** It's the *intensity* (absolute value) of sentiment that matters, not the direction.

```
Logistic Regression: |Sentiment_Score| → Volatility_Shock (binary)
Result: p < 0.05 — SIGNIFICANT
```

Using the absolute value of sentiment as the predictor and defining a VIX shock as a move exceeding 1 standard deviation, the logistic regression found a statistically significant relationship. The **odds ratio** showed that for every 1-unit increase in speech intensity, the odds of a volatility shock increased meaningfully.

**Lesson:** Strong language — hawkish *or* dovish — signals conviction, and markets react to conviction with volatility.

### Stage 3: Random Forest Classifier (✅ Production Model)

Built on the Stage 2 insight, the final model combines sentiment intensity with live market context (VIX level, Nifty momentum, volatility regime) in a Random Forest classifier. This model powers the live **RBI Sentinel** dashboard.

---

## Backtesting

A simple **"Sentiment Shield"** strategy was tested:

- **Signal:** If `|Sentiment_Score| > 0.4` on an RBI day, exit the market for 2 trading days.
- **Otherwise:** Stay invested in Nifty 50 (buy & hold).

The strategy was evaluated against a passive buy-and-hold benchmark over the full 2020–2026 period (1,519 trading days) using total return, max drawdown, and Sharpe ratio.

---

## Live Dashboard

**RBI Sentinel** is a Bloomberg Terminal-inspired Streamlit dashboard that provides real-time volatility shock prediction.

### Features

- **Paste any RBI communication** (speech or MPC minutes) and get an instant shock probability.
- **Live market sidebar** with India VIX, Nifty 50 5-day return, and CBOE VIX pulled from Yahoo Finance.
- **FinBERT inference** using the same sliding-window pipeline as training.
- **Transparent methodology** section explaining every step of the prediction.
- **Risk gauge** with colour-coded probability bands (Low / Moderate / High).

### Running the Dashboard

```bash
cd streamlit-app
streamlit run app.py
```

---

## Project Structure

```
rbi-sentiment-volatility-forecasting/
│
├── data/
│   ├── raw/
│   │   ├── speeches/                  # 102 Governor speech PDFs (2020–2026)
│   │   ├── minutes/                   # 38 MPC minutes PDFs (2020–2026)
│   │   ├── rbi_communications_raw.csv # Extracted text from all 140 PDFs
│   │   └── market_data_raw.csv        # Nifty 50 + India VIX daily closes
│   └── processed/
│       ├── rbi_sentiment_scores.csv   # FinBERT sentiment scores (140 rows)
│       ├── master_dataset.csv         # Merged sentiment + market data (139 rows)
│       └── backtest_dataset.csv       # Full trading-day dataset (1,519 rows)
│
├── src/
│   ├── 01a_extract_pdfs.py            # PDF → text extraction (pdfplumber)
│   ├── 01b_get_market_data.py         # Yahoo Finance data download
│   ├── 02_sentiment_analysis.py       # FinBERT sliding-window sentiment scoring
│   ├── 03_master_merge.py             # Merge sentiment + market with merge_asof
│   ├── analysis/
│   │   ├── 01_baseline_ols.py         # OLS regression (Stage 1 — failed)
│   │   ├── 02_volatility_shocks.py    # Abs-value intensity analysis (Stage 2)
│   │   └── 03_results_synthesis.py    # Final odds ratio & probability table
│   ├── backtest/
│   │   ├── 01_prepare_backtest_data.py # Create full trading-day dataset
│   │   └── 02_strategy_backtest.py     # Sentiment Shield strategy backtest
│   └── models/
│       └── 01_train_predictor.py       # Random Forest training & evaluation
│
├── models/
│   ├── rbi_vix_rf_model.pkl           # Trained Random Forest model
│   ├── scaler.pkl                     # Fitted StandardScaler
│   ├── feature_names.pkl              # Ordered feature list
│   └── shock_threshold.pkl            # 80th-percentile VIX return threshold
│
├── streamlit-app/
│   ├── app.py                         # RBI Sentinel dashboard
│   ├── style.css                      # Bloomberg Lite dark theme
│   └── logo.png                       # Dashboard logo
│
├── reports/
│   └── figures/                       # Generated plots and visualisations
│
├── notebooks/                         # Exploratory Jupyter notebooks
├── requirements.txt                   # Python dependencies
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- ~2 GB disk space for FinBERT model weights (downloaded automatically on first run)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rbi-sentiment-volatility-forecasting.git
cd rbi-sentiment-volatility-forecasting

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the Full Pipeline

Run scripts **from the project root directory** in order:

```bash
# Step 1: Extract text from PDFs (requires PDFs in data/raw/)
python src/01a_extract_pdfs.py

# Step 2: Download market data
python src/01b_get_market_data.py

# Step 3: Run FinBERT sentiment analysis (~5-10 min on CPU)
python src/02_sentiment_analysis.py

# Step 4: Merge sentiment + market data
python src/03_master_merge.py

# Step 5: Analysis (optional — reproduces research findings)
python src/analysis/01_baseline_ols.py
python src/analysis/02_volatility_shocks.py
python src/analysis/03_results_synthesis.py

# Step 6: Prepare backtest data & train model
python src/backtest/01_prepare_backtest_data.py
python src/models/01_train_predictor.py

# Step 7: Run backtest (optional)
python src/backtest/02_strategy_backtest.py

# Step 8: Launch dashboard
cd streamlit-app
streamlit run app.py
```

### Data Note

The `data/` directory is excluded from Git (files are too large). To reproduce from scratch, you will need to:

1. Download the 140 RBI PDFs manually from the [RBI website](https://www.rbi.org.in) and place them in `data/raw/speeches/` and `data/raw/minutes/` with date-prefixed filenames (e.g., `2024-02-08_minutes.pdf`).
2. Run the pipeline from Step 1 above.

Alternatively, the trained model artifacts in `models/` are included in the repository and the Streamlit dashboard can be used directly.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **NLP** | [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) (Hugging Face Transformers) |
| **ML** | scikit-learn (Random Forest, StandardScaler) |
| **Data** | pandas, NumPy, pdfplumber, yfinance |
| **Visualisation** | Plotly, Matplotlib, Seaborn |
| **Statistical Analysis** | statsmodels (OLS, Logistic Regression), SciPy |
| **Dashboard** | Streamlit |
| **Deep Learning** | PyTorch (FinBERT inference) |

---

## Limitations

- **Small sample size:** ~130 RBI event-days in training. Results should be interpreted with caution.
- **No causal claims:** The model captures statistical *association*, not causation. Other macro events on RBI days are not isolated.
- **English-language model:** FinBERT was pre-trained on English financial text; nuances of RBI's specific phrasing may not be fully captured.
- **Regime dependence:** Model performance may degrade if market structure shifts significantly from the 2020–2024 training period.

---

## License

This project is for educational and research purposes.

---

<p align="center">
  <em>Built as a quantitative research project exploring the intersection of central bank communication and market volatility</em>
</p>
