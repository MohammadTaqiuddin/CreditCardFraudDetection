# app.py
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from datetime import datetime

app = FastAPI()

# Allow all origins (dev only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("model.pkl")

# Load transaction history (for demo)
try:
    txdf = pd.read_csv("transactions.csv", parse_dates=["timestamp"])
except Exception:
    txdf = pd.DataFrame(columns=[
        "txn_id","timestamp","amount","merchant_id","card_age_days",
        "card_country","ip_country","device_id","hour","device_rate","is_fraud"
    ])

# In-memory store for supervisor actions
actions = []

# Pydantic model for transaction input
class Txn(BaseModel):
    txn_id: int
    timestamp: str
    amount: float
    merchant_id: int
    card_age_days: float
    card_country: str
    ip_country: str
    device_id: int
    hour: int
    device_rate: float

def featurize_df(df):
    """
    Feature engineering compatible with training data.
    """
    X = pd.DataFrame()
    X['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    X['merchant_id'] = pd.to_numeric(df['merchant_id'], errors='coerce')
    X['card_age_days'] = pd.to_numeric(df['card_age_days'], errors='coerce')
    X['is_foreign_ip'] = (df['card_country'] != df['ip_country']).astype(int)
    X['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    X['device_rate'] = pd.to_numeric(df['device_rate'], errors='coerce')

    if not txdf.empty:
        merchant_stats = txdf.groupby('merchant_id')['amount'].agg(['count','mean']).rename(columns={'count':'m_count','mean':'m_mean'})
        X = X.join(df['merchant_id'].map(merchant_stats['m_count']).rename('m_count').fillna(0))
        X = X.join(df['merchant_id'].map(merchant_stats['m_mean']).rename('m_mean').fillna(0))
    else:
        X['m_count'] = 0
        X['m_mean'] = 0

    return X.fillna(0)

@app.get("/transactions")
def list_transactions(limit: int = 50):
    """
    Return recent transactions as JSON.
    """
    df = txdf.sort_values("timestamp", ascending=False).head(limit)
    return df.to_dict(orient="records")

@app.post("/score")
async def score_txn(txn: Txn):
    """
    Score a transaction for fraud probability.
    """
    try:
        d = txn.model_dump()
        d_df = pd.DataFrame([d])
        X = featurize_df(d_df)

        proba = float(model.predict_proba(X)[:, 1][0])

        important = {
            "amount": d['amount'],
            "is_foreign_ip": int(d['card_country'] != d['ip_country']),
            "device_rate": d['device_rate']
        }

        return {"score": proba, "explain": important}

    except Exception as e:
        # Return error so frontend sees what went wrong (prevents fake CORS errors)
        return {"error": str(e)}

@app.post("/action")
async def action(txn_id: int = Form(...), action: str = Form(...), reason: str = Form(None)):
    """
    Record supervisor action on a transaction.
    """
    actions.append({
        "txn_id": txn_id,
        "action": action,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat()
    })
    return {"ok": True, "actions_count": len(actions)}

@app.get("/actions")
def get_actions():
    """
    Retrieve all supervisor actions.
    """
    return actions

if __name__ == "__main__":
    # Run with: uvicorn app:app --reload
    uvicorn.run(app, host="127.0.0.1", port=8000)
