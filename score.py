import sqlite3
import pandas as pd
import joblib
from preprocessing import preprocess

feature_cols = ['PJME_MW', 'lag_1', 'lag_24', 'lag_168',
                 'hour', 'dayofweek', 'is_weekend', 'month', 'is_holiday']

def score():
    model = joblib.load('isolation_forest_model.pkl')

    conn = sqlite3.connect('pjme_load.db')
    raw = pd.read_sql('SELECT * FROM load_data', conn)
    processed = preprocess(raw)

    processed['anomaly_score'] = model.decision_function(processed[feature_cols])
    processed['is_anomaly'] = model.predict(processed[feature_cols])

    results = processed[['Datetime', 'PJME_MW', 'anomaly_score', 'is_anomaly']]
    results.to_sql('anomaly_scores', conn, if_exists='replace', index=False)
    conn.close()

    print(f"Scored {len(results)} rows. {sum(results['is_anomaly']==-1)} flagged as anomalies.")
    return results

if __name__ == "__main__":
    score()