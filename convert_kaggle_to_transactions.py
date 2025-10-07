# convert_kaggle_to_transactions.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load Kaggle dataset
df = pd.read_csv("creditcard.csv")  # Make sure 'creditcard.csv' is in the same folder

# Rename columns if needed (original Kaggle dataset has 'Time', 'Amount', 'Class')
df.rename(columns={'Time': 'time_sec', 'Amount': 'amount', 'Class': 'is_fraud'}, inplace=True)

# Generate transaction IDs
df['txn_id'] = range(1, len(df)+1)

# Generate timestamp from 'time_sec' (Kaggle dataset time is seconds from first transaction)
start_time = datetime(2025, 1, 1)  # arbitrary start
df['timestamp'] = df['time_sec'].apply(lambda x: start_time + timedelta(seconds=int(x)))

# Generate card_age_days randomly (1 to 2000 days)
df['card_age_days'] = np.random.exponential(scale=600, size=len(df)).round(1)

# Assign random card_country
countries = ['IN','US','GB','DE','CN','NG']
df['card_country'] = np.random.choice(countries, size=len(df), p=[0.7,0.12,0.05,0.04,0.06,0.03])

# Assign IP country (mostly same as card_country, 95%)
df['ip_country'] = [
    c if np.random.rand() < 0.95 else np.random.choice(countries)
    for c in df['card_country']
]

# Generate device_id (random int)
df['device_id'] = np.random.randint(1, 5000, size=len(df))

# Extract hour from timestamp
df['hour'] = df['timestamp'].dt.hour

# Generate device_rate: how many transactions per device
device_counts = pd.Series(df['device_id']).value_counts().to_dict()
df['device_rate'] = df['device_id'].map(device_counts)

# Select only columns your app expects
transactions = df[[
    'txn_id', 'timestamp', 'amount', 'merchant_id' if 'merchant_id' in df.columns else 'txn_id',
    'card_age_days','card_country','ip_country','device_id','hour','device_rate','is_fraud'
]].copy()

# If merchant_id is missing, just assign random merchant IDs
if 'merchant_id' not in df.columns:
    transactions['merchant_id'] = np.random.randint(1, 501, size=len(transactions))

# Save to transactions.csv
transactions.to_csv("transactions.csv", index=False)
print("transactions.csv generated successfully!")
