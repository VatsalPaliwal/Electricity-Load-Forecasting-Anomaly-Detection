import sqlite3
import pandas as pd
import joblib
import mlflow
from sklearn.ensemble import IsolationForest
from preprocessing import preprocess

feature_cols = ['PJME_MW', 'lag_1', 'lag_24', 'lag_168',
                 'hour', 'dayofweek', 'is_weekend', 'month', 'is_holiday']

N_ESTIMATORS = 100
CONTAMINATION = 0.01

def train():
    conn = sqlite3.connect('pjme.db')
    train_data = pd.read_sql('SELECT * FROM train_data', conn)
    conn.close()


    with mlflow.start_run():
        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=42
        )
        model.fit(train_data[feature_cols])

        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("contamination", CONTAMINATION)

        mlflow.log_param("num_rows", len(train_data))
        mlflow.log_param("date_start", str(train_data['Datetime'].min()))
        mlflow.log_param("date_end", str(train_data['Datetime'].max()))

        scores = model.predict(train_data[feature_cols])
        anomaly_rate = (scores == -1).mean()
        mlflow.log_metric("anomaly_rate", anomaly_rate)

        mlflow.sklearn.log_model(model, "isolation_forest_model")
        joblib.dump(model, 'isolation_forest_model.pkl')

    print(f"Model trained on {len(train_data)} rows. Anomaly rate: {anomaly_rate:.4f}")

if __name__ == "__main__":
    train()