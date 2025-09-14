
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_customers(n_customers=2000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    cust_ids = [f"C{100000+i}" for i in range(n_customers)]
    tenure = np.random.exponential(scale=24, size=n_customers).astype(int) + 1
    monthly_charges = np.round(np.random.normal(loc=70, scale=25, size=n_customers),2)
    monthly_charges = np.clip(monthly_charges, 15, 200)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0,50,size=n_customers),2)
    contract = np.random.choice(["Month-to-month","One year","Two year"], size=n_customers, p=[0.55,0.25,0.20])
    payment_method = np.random.choice(["Electronic check","Mailed check","Bank transfer","Credit card"], size=n_customers, p=[0.4,0.2,0.2,0.2])
    has_internet = np.random.choice([0,1], size=n_customers, p=[0.1,0.9])
    # churn probability influenced by tenure, monthly_charges, contract
    churn_prob = (0.35 - (tenure/200)) + (monthly_charges-50)/500 + (contract=="Month-to-month")*0.12 + np.random.normal(0,0.05,size=n_customers)
    churn = (churn_prob > 0.25).astype(int)
    df = pd.DataFrame({
        "customer_id": cust_ids,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "payment_method": payment_method,
        "has_internet": has_internet,
        "churn": churn
    })
    return df

def generate_transactions(customers_df, days=365, seed=42):
    np.random.seed(seed)
    tx_rows = []
    start_date = datetime.now() - timedelta(days=days)
    for _, row in customers_df.iterrows():
        cid = row["customer_id"]
        # number of transactions roughly proportional to tenure and usage
        n_tx = int(max(1, min(200, np.random.poisson(lam=12 + row['tenure']/2))))
        for i in range(n_tx):
            ts = start_date + timedelta(days=np.random.uniform(0, days))
            amount = max(1.0, np.random.normal(loc=row['monthly_charges']/4, scale=10.0))
            # inject some anomalies randomly (fraud or abnormal spikes)
            is_anom = 0
            if np.random.rand() < 0.005:
                amount *= np.random.uniform(5, 20)
                is_anom = 1
            tx_rows.append({
                "tx_id": f"T{len(tx_rows)+1:07d}",
                "customer_id": cid,
                "timestamp": ts,
                "amount": round(amount,2),
                "is_anom": is_anom
            })
    tx = pd.DataFrame(tx_rows)
    tx.sort_values("timestamp", inplace=True)
    tx.reset_index(drop=True, inplace=True)
    return tx

if __name__ == '__main__':
    customers = generate_customers(2000)
    transactions = generate_transactions(customers, days=365)
    customers.to_csv("customers.csv", index=False)
    transactions.to_csv("transactions.csv", index=False)
