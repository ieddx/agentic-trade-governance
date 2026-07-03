"""
data_loader.py — fetch AAPL hourly bars from Alpaca and cache them locally.

Alpaca provides free paper-trading API keys that give access to historical
market data.  Keys are read from environment variables (never hardcoded).
Set them in a .env file (see .env.example) before running.

The first call fetches from the network and writes data/AAPL_1h.csv.
Subsequent calls return the cached CSV so you don't burn rate-limit quota.
"""

import os
import pathlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Alpaca SDK imports.  alpaca-py >= 0.8 ships a clean REST client under
# alpaca.data.historical.
# ---------------------------------------------------------------------------
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Load .env from the project root regardless of the current working directory.
# __file__ is finance_core/data_loader.py, so .parent.parent is the project root.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Where to store cached bar data relative to the project root.
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CACHE_FILE = DATA_DIR / "AAPL_1h.csv"

# Default lookback: 90 calendar days of hourly bars.
DEFAULT_DAYS_BACK = 90


def _get_client() -> StockHistoricalDataClient:
    """
    Build an Alpaca data client using API keys from environment variables.
    Raises a clear error if the keys are missing so the developer knows
    exactly what to fix.
    """
    api_key    = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.\n"
            "Copy .env.example to .env and fill in your credentials."
        )

    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def load_bars(symbol: str = "AAPL", days_back: int = DEFAULT_DAYS_BACK) -> pd.DataFrame:
    """
    Return a DataFrame of 1-hour OHLCV bars for *symbol*.

    If a cache file already exists the function returns it immediately.
    Otherwise it fetches from Alpaca, saves the result, and returns it.

    Columns: timestamp (index), open, high, low, close, volume
    """
    if CACHE_FILE.exists():
        print(f"[data_loader] Using cached bars from {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE, index_col="timestamp", parse_dates=True)
        return df

    print(f"[data_loader] Fetching {days_back}-day hourly bars for {symbol} from Alpaca …")

    client = _get_client()

    # Define the request: symbol, bar size, time range.
    now   = datetime.now(tz=timezone.utc)
    start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    # Go back by the requested number of days.
    import datetime as dt
    start = start - dt.timedelta(days=days_back)

    # feed=DataFeed.IEX is required for free/paper Alpaca accounts.
    # SIP (the default consolidated feed) requires a paid subscription.
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=start,
        end=now,
        feed=DataFeed.IEX,
    )

    bars = client.get_stock_bars(request)

    # alpaca-py returns a BarSet; convert to a tidy DataFrame.
    df = bars.df

    # The multi-index has (symbol, timestamp); we only asked for one symbol so
    # we can drop the outer level.
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df.index.name = "timestamp"

    # Keep only the standard OHLCV columns to avoid version-specific extras.
    df = df[["open", "high", "low", "close", "volume"]].copy()

    # Persist so future runs skip the network call.
    df.to_csv(CACHE_FILE)
    print(f"[data_loader] Saved {len(df)} bars to {CACHE_FILE}")

    return df
