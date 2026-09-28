import numpy as np
import pandas as pd


def create_features(data, volatility_window=20):
    """
    Create features and a future volatility target.

    The target is the realized volatility over the next
    `volatility_window` trading days.
    """

    df = data.copy()

    # Daily log returns
    df["log_return"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    # Historical volatility
    df["volatility_5d"] = (
        df["log_return"]
        .rolling(5)
        .std()
        * np.sqrt(252)
    )

    df["volatility_20d"] = (
        df["log_return"]
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    df["volatility_60d"] = (
        df["log_return"]
        .rolling(60)
        .std()
        * np.sqrt(252)
    )

    # Trading-volume feature
    df["volume_change"] = df["Volume"].pct_change()

    # Future 20-day realized volatility.
    # Shift(-20) ensures we are predicting future volatility,
    # rather than accidentally using information from the future.
    future_returns = df["log_return"].shift(-1)

    df["target_volatility"] = (
        future_returns
        .rolling(volatility_window)
        .std()
        .shift(-(volatility_window - 1))
        * np.sqrt(252)
    )

    df = df.dropna()

    return df
