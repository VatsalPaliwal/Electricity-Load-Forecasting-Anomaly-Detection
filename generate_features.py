# isolation_forest/generate_features.py
import sqlite3
import pandas as pd
import joblib

feature_cols = ['PJME_MW', 'lag_1', 'lag_24', 'lag_168',
                 'hour', 'dayofweek', 'is_weekend', 'month', 'is_holiday']

MODEL_PATH = 'isolation_forest_model.pkl'
DB_PATH = 'pjme_load.db'

def generate_anomaly_features(model_path: str = MODEL_PATH, db_path: str = DB_PATH):

    iso_forest = joblib.load(model_path)

    conn = sqlite3.connect(db_path)
    train_data = pd.read_sql("SELECT * FROM train_data", conn)
    val_data = pd.read_sql("SELECT * FROM val_data", conn)
    conn.close()

    train_data["Datetime"] = pd.to_datetime(train_data["Datetime"])
    val_data["Datetime"] = pd.to_datetime(val_data["Datetime"])

    combined = pd.concat([train_data, val_data], ignore_index=True)
    combined = combined.sort_values("Datetime").reset_index(drop=True)

    X = combined[feature_cols]

    combined["anomaly_score"] = -iso_forest.decision_function(X)
    combined["is_anomaly"] = (iso_forest.predict(X) == -1).astype(int)
    combined["anomaly_score_24h_mean"] = (
        combined["anomaly_score"].rolling(window=24, min_periods=1).mean().shift(1)
    )
    combined["anomaly_score_24h_max"] = (
        combined["anomaly_score"].rolling(window=24, min_periods=1).max().shift(1)
    )

    # Split back apart using the original row counts (order preserved by sort)
    n_train = len(train_data)
    updated_train = combined.iloc[:n_train].reset_index(drop=True)
    updated_val = combined.iloc[n_train:].reset_index(drop=True)

    conn = sqlite3.connect(db_path)
    updated_train.to_sql("train_data", conn, if_exists="replace", index=False)
    updated_val.to_sql("val_data", conn, if_exists="replace", index=False)
    conn.close()

    print(f"train_data updated: {len(updated_train)} rows")
    print(f"val_data updated: {len(updated_val)} rows")


if __name__ == "__main__":
    generate_anomaly_features()