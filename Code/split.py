import sqlite3
import pandas as pd
from preprocessing import preprocess

def create_splits(df: pd.DataFrame, train_end: str, val_end: str, datetime_col: str = "Datetime"):
 
    ts = pd.to_datetime(df[datetime_col])
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)

    train = df[ts < train_end].copy()
    val = df[(ts >= train_end) & (ts < val_end)].copy()
    test = df[ts >= val_end].copy()

    return train, val, test


def write_splits_to_db(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, db_path: str,type:str='normal'):
    conn = sqlite3.connect(db_path)
    try:
        if type=='normal':
            train.to_sql("train_data", conn, if_exists="replace", index=False)
            val.to_sql("val_data", conn, if_exists="replace", index=False)
            test.to_sql("test_data", conn, if_exists="replace", index=False)
        if type=='anomaly':
            train.to_sql("train_anomaly_data", conn, if_exists="replace", index=False)
            val.to_sql("val_anomaly_data", conn, if_exists="replace", index=False)
            test.to_sql("test_anomaly_data", conn, if_exists="replace", index=False)
    finally:
        conn.close()


if __name__ == "__main__":
    # Example usage — adjust boundaries/paths to your actual setup
    conn = sqlite3.connect("pjme_load.db")
    raw = pd.read_sql("SELECT * FROM load_data", conn)
    raw_anamoly= pd.read_sql("SELECT * FROM anomaly_data", conn)
    conn.close()
    processed = preprocess(raw)  
    processed_anamoly = preprocess(raw_anamoly)
    train_df, val_df, test_df = create_splits(
        processed, train_end="2015-01-01", val_end="2017-01-01"
    )
    train_df_anamoly, val_df_anamoly, test_df_anamoly = create_splits(
        processed_anamoly, train_end="2015-01-01", val_end="2017-01-01"
    )
    write_splits_to_db(train_df, val_df, test_df, "pjme_load.db")
    write_splits_to_db(train_df_anamoly, val_df_anamoly, test_df_anamoly, "pjme_load.db",type='anomaly')
    print(f"train: {len(train_df)} rows, val: {len(val_df)} rows, test: {len(test_df)} rows")
