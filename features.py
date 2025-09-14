
import pandas as pd
import numpy as np

def create_features(customers_path="customers.csv", transactions_path="transactions.csv", out_path="features.csv"):
    cust = pd.read_csv(customers_path)
    tx = pd.read_csv(transactions_path, parse_dates=["timestamp"])
    # RFM per customer (recency in days, frequency, monetary)
    last_date = tx["timestamp"].max()
    rfm = tx.groupby("customer_id").agg(recency_days = ("timestamp", lambda x: (last_date - x.max()).days),
                                        frequency = ("tx_id", "count"),
                                        monetary = ("amount", "sum"))
    # rolling features: avg_last_30, avg_last_90 by joining last transactions
    tx["days_from_last"] = (last_date - tx["timestamp"]).dt.days
    avg30 = tx[tx["days_from_last"]<=30].groupby("customer_id")["amount"].mean().rename("avg_30d")
    avg90 = tx[tx["days_from_last"]<=90].groupby("customer_id")["amount"].mean().rename("avg_90d")
    df = cust.merge(rfm, left_on="customer_id", right_index=True, how="left").merge(avg30, left_on="customer_id", right_index=True, how="left").merge(avg90, left_on="customer_id", right_index=True, how="left")
    df["avg_30d"].fillna(0, inplace=True)
    df["avg_90d"].fillna(0, inplace=True)
    # features: tenure bucket, monthly_charges normalized, contract encoded simple
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=[0,6,12,24,48,999], labels=["0-6","7-12","13-24","25-48","48+"])
    df["monthly_norm"] = (df["monthly_charges"] - df["monthly_charges"].mean())/df["monthly_charges"].std()
    # interaction feature
    df["rfm_score"] = ( (1/(1+df["recency_days"])) * 0.4 + (df["frequency"]/ (1+df["frequency"].max()))*0.3 + (df["monetary"]/ (1+df["monetary"].max()))*0.3 )
    # fill na
    df.fillna(0, inplace=True)
    df.to_csv(out_path, index=False)
    return df

if __name__ == '__main__':
    df = create_features("customers.csv","transactions.csv")
    print(df.head())
