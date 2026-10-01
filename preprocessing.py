import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def preprocess(df):
    df = df.copy()

    # Convert Datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Remove duplicate timestamps
    df = df.drop_duplicates(
        subset="Datetime",
        keep="first"
    )

    # Remove rows with missing target values
    df = df.dropna(
        subset=["PJME_MW"]
    )

    # Sort chronologically
    df = df.sort_values(
        "Datetime"
    ).reset_index(drop=True)

    # Lag features
    df["lag_1"] = df["PJME_MW"].shift(1)
    df["lag_24"] = df["PJME_MW"].shift(24)
    df["lag_168"] = df["PJME_MW"].shift(168)

    # Calendar features
    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["Month"] = df["Datetime"].dt.month

    # US federal holidays
    cal = USFederalHolidayCalendar()

    holidays = cal.holidays(
        start=df["Datetime"].min(),
        end=df["Datetime"].max()
    )

    df["is_holiday"] = (
        df["Datetime"]
        .dt.normalize()
        .isin(holidays)
        .astype(int)
    )

    # Remove rows where lag features cannot be calculated
    df = df.dropna(
        subset=["lag_1", "lag_24", "lag_168"]
    )

    return df.reset_index(drop=True)