# generate_and_train.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta

np.random.seed(42)

def generate_transactions(n=20000, fraud_rate=0.01):
    """
    Generate synthetic credit card transaction data with injected fraud patterns.
    """
    txn_id = np.arange(1, n + 1)
    start_time = datetime(2025, 1, 1)
    
    # Random timestamps
    timestamps = [
        start_time + timedelta(seconds=int(x))
        for x in np.cumsum(np.random.exponential(scale=30, size=n).astype(int))
    ]
    
    # Transaction amounts
    amounts = np.random.exponential(scale=80, size=n) + 1
    
    # Merchant IDs
    merchant_id = np.random.randint(1, 501, size=n)
    
    # Card age in days
    card_age_days = np.random.exponential(scale=600, size=n)
    
    # Card country
    card_country = np.random.choice(['IN','US','GB','DE','CN','NG'], size=n, p=[0.7,0.12,0.05,0.04,0.06,0.03])
    
    # IP country (simulate foreign IPs)
    ip_country = [
        c if np.random.rand() < 0.95 else np.random.choice(['IN','US','GB','DE','CN','NG'])
        for c in card_country
    ]
    
    # Device IDs
    device_id = np.random.randint(1, 5000, size=n)
    
    # Hour of transaction
    hour = np.array([t.hour for t in timestamps])
    
    # Device transaction frequency
    device_txn_counts = pd.Series(device_id).value_counts().to_dict()
    device_rate = np.array([device_txn_counts[d] for d in device_id])
    
    # Inject fraud patterns
    is_fraud = np.zeros(n, dtype=int)
    cond_a = (amounts > 300) & (card_country == 'IN') & (np.array(ip_country) != 'IN') & (np.random.rand(n) < 0.6)
    cond_b = (device_rate > 8) & (amounts > 50) & (np.random.rand(n) < 0.5)
    bad_merchants = set(np.random.choice(np.arange(1, 501), size=8, replace=False))
    cond_c = np.isin(merchant_id, list(bad_merchants)) & (np.random.rand(n) < 0.4)
    cond_d = (card_age_days < 2) & (amounts > 100) & (np.random.rand(n) < 0.5)
    
    # Fraud probability
    prob = (
        cond_a.astype(float)*0.9 +
        cond_b.astype(float)*0.8 +
        cond_c.astype(float)*0.7 +
        cond_d.astype(float)*0.85 +
        np.random.beta(0.5, 20, size=n)
    )
    kth = np.quantile(prob, 1 - fraud_rate)
    is_fraud = (prob >= kth).astype(int)
    
    df = pd.DataFrame({
        "txn_id": txn_id,
        "timestamp": timestamps,
        "amount": np.round(amounts, 2),
        "merchant_id": merchant_id,
        "card_age_days": np.round(card_age_days, 1),
        "card_country": card_country,
        "ip_country": ip_country,
        "device_id": device_id,
        "hour": hour,
        "device_rate": device_rate,
        "is_fraud": is_fraud
    })
    
    return df, bad_merchants

def featurize(df, txdf=None):
    """
    Feature engineering compatible with app.py.
    If txdf (full history) is provided, compute merchant stats.
    """
    X = pd.DataFrame()
    X['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    X['merchant_id'] = pd.to_numeric(df['merchant_id'], errors='coerce')
    X['card_age_days'] = pd.to_numeric(df['card_age_days'], errors='coerce')
    X['is_foreign_ip'] = (df['card_country'] != df['ip_country']).astype(int)
    X['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    X['device_rate'] = pd.to_numeric(df['device_rate'], errors='coerce')

    if txdf is not None and not txdf.empty:
        merchant_stats = txdf.groupby('merchant_id')['amount'].agg(['count','mean']).rename(columns={'count':'m_count','mean':'m_mean'})
        X = X.join(df['merchant_id'].map(merchant_stats['m_count']).rename('m_count').fillna(0))
        X = X.join(df['merchant_id'].map(merchant_stats['m_mean']).rename('m_mean').fillna(0))
    else:
        X['m_count'] = 0
        X['m_mean'] = 0
    
    return X.fillna(0)

if __name__ == "__main__":
    print("Generating data...")
    df, bad_merchants = generate_transactions(n=20000, fraud_rate=0.01)
    
    print("Saving transactions.csv ...")
    df.to_csv("transactions.csv", index=False)
    
    print("Featurizing and training model...")
    X = featurize(df, txdf=df)
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    joblib.dump(model, "model.pkl")
    
    print("Trained model saved to model.pkl")
    print("Sample bad merchants (injected):", sorted(list(bad_merchants)))
    print("Done.")
