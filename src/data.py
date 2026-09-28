import yfinance as yf
import pandas as pd


def download_data(
    ticker="SPY",
    start="2015-01-01",
    end="2025-01-01"
):
    """
    Download historical daily market data.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Historical market data.
    """

    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True
    )

    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # yfinance can return a MultiIndex depending on the version/output.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna()

    return data
