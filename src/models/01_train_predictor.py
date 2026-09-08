import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/processed/backtest_dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_model():
    # ─────────────────────────────────────────────────────────
    # 1. LOAD ALL 1519 ROWS
    # ─────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['Sentiment_Score'] = df['Sentiment_Score'].fillna(0.0)
    df['Document_Type'] = df['Document_Type'].fillna('')

    print(f"Total rows loaded: {len(df)}")

    # ─────────────────────────────────────────────────────────
    # 2. FIX THE VIX GAP
    #    VIX only exists on 130 speech days
    #    Forward-fill it so every day carries the last known VIX
    #    (like saying "fear level stays until next reading")
    # ─────────────────────────────────────────────────────────
    df['VIX_Close_filled'] = df['VIX_Close'].ffill()

    # ─────────────────────────────────────────────────────────
    # 3. COMPUTE FEATURES ON ALL 1519 ROWS
    #    (Nifty is available daily, VIX is now filled daily)
    # ─────────────────────────────────────────────────────────
    df['Sentiment_Intensity'] = df['Sentiment_Score'].abs()

    # Yesterday's VIX (using filled values so no gaps)
    df['VIX_Lag1'] = df['VIX_Close_filled'].shift(1)

    # Nifty 5-day return (Nifty_Close is available every day)
    df['Nifty_5d_Ret'] = np.log(df['Nifty_Close'] / df['Nifty_Close'].shift(5))

    # Nifty 5-day volatility (replaces VIX_5d_Std which was broken)
    df['Nifty_Daily_Ret'] = np.log(
        df['Nifty_Close'] / df['Nifty_Close'].shift(1))
    df['Nifty_5d_Std'] = df['Nifty_Daily_Ret'].rolling(5).std()

    # Is MPC flag
    df['Is_MPC'] = df['Document_Type'].str.contains(
        'MPC', case=False).astype(int)

    # ─────────────────────────────────────────────────────────
    # 4. NOW FILTER TO SPEECH DAYS ONLY (features are ready)
    # ─────────────────────────────────────────────────────────
    df_speech = df[df['Document_Type'] != ''].copy()

    # Drop rows where features couldn't be computed (first few days)
    df_speech.dropna(subset=['VIX_Returns', 'VIX_Lag1',
                     'Nifty_5d_Ret', 'Nifty_5d_Std'], inplace=True)
    df_speech.reset_index(drop=True, inplace=True)

    print(f"Speech/MPC days after cleanup: {len(df_speech)}")
    print(f"  Speeches:    {(df_speech['Document_Type'] == 'Speech').sum()}")
    print(
        f"  MPC Minutes: {(df_speech['Document_Type'] == 'MPC_Minutes').sum()}")

    if len(df_speech) == 0:
        print("[ERROR] Still 0 rows. Something is wrong with the data.")
        return

    # ─────────────────────────────────────────────────────────
    # 5. DATA-DRIVEN THRESHOLD (from training period only)
    # ─────────────────────────────────────────────────────────
    train_slice = df_speech[df_speech['Date'] <= '2024-12-31']
    threshold = train_slice['VIX_Returns'].quantile(0.80)
    print(f"\nShock threshold (80th pctl of train): {threshold:.4f}")

    df_speech['Target'] = (df_speech['VIX_Returns'] > threshold).astype(int)
    print(
        f"Target distribution:\n{df_speech['Target'].value_counts().to_string()}")

    # ─────────────────────────────────────────────────────────
    # 6. CHRONOLOGICAL SPLIT
    # ─────────────────────────────────────────────────────────
    features = ['Sentiment_Intensity', 'VIX_Lag1',
                'Nifty_5d_Ret', 'Nifty_5d_Std', 'Is_MPC']

    train_mask = df_speech['Date'] <= '2024-12-31'
    test_mask = df_speech['Date'] >= '2025-01-01'

    X_train = df_speech.loc[train_mask, features]
    y_train = df_speech.loc[train_mask, 'Target']
    X_test = df_speech.loc[test_mask, features]
    y_test = df_speech.loc[test_mask, 'Target']

    print(f"\nTrain: {len(X_train)} rows ({y_train.sum()} shocks)")
    print(f"Test:  {len(X_test)} rows ({y_test.sum()} shocks)")

    # Safety checks
    if len(X_test) == 0:
        print("\n[WARNING] No test data after 2025-01-01.")
        print("Falling back to 80/20 chronological split...")
        split_idx = int(len(df_speech) * 0.8)
        X_train = df_speech.iloc[:split_idx][features]
        y_train = df_speech.iloc[:split_idx]['Target']
        X_test = df_speech.iloc[split_idx:][features]
        y_test = df_speech.iloc[split_idx:]['Target']
        print(f"Train: {len(X_train)} rows ({y_train.sum()} shocks)")
        print(f"Test:  {len(X_test)} rows ({y_test.sum()} shocks)")

    if y_train.nunique() < 2:
        print("[ERROR] Only one class in training data.")
        return

    # ─────────────────────────────────────────────────────────
    # 7. SCALE
    # ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ─────────────────────────────────────────────────────────
    # 8. TRAIN
    # ─────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # ─────────────────────────────────────────────────────────
    # 9. EVALUATE
    # ─────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_scaled)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['No Shock', 'Shock']))

    # Feature importance
    print("Feature Importances:")
    for name, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {name:25s} {imp:.3f} {bar}")

    # ─────────────────────────────────────────────────────────
    # 10. SAVE
    # ─────────────────────────────────────────────────────────
    joblib.dump(model, os.path.join(MODEL_DIR, "rbi_vix_rf_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump(threshold, os.path.join(MODEL_DIR, "shock_threshold.pkl"))

    print(f"\nAll artifacts saved in {MODEL_DIR}/")


if __name__ == "__main__":
    train_model()
