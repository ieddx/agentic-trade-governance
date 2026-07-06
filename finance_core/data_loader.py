"""
data_loader.py — fetch hourly bars from Alpaca for any ticker.

Feed priority on every call:
  1. SIP (consolidated tape) — free for historical queries when end ≥ 15 min
     in the past; most complete data.
  2. IEX — automatic fallback if SIP is restricted on this account.
  3. Local cache — fallback of last resort if both live feeds fail (current
     mode only; historical queries never fall back to a stale current cache).

Returns a (DataFrame, feed_name) tuple so callers can log which feed was used.

Modes
-----
Current mode  (as_of=None)   : end = now − 15 min.  Saves result to the
                                per-symbol cache so the next run can fall back.
Historical mode (as_of=<dt>) : end = as_of.  Always satisfies the SIP ≥15-min
                                rule.  Does NOT overwrite the current cache.
"""

import datetime as dt
import os
import pathlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# NYSE regular session: 9:30 AM – 4:00 PM Eastern.
# We filter to these hours so downstream callers (signal, market context) never
# see pre-market or after-hours bars whose volume is structurally different from
# regular-session volume and would distort ratio / volatility calculations.
_RTH_START = dt.time(9, 30)   # 09:30 ET inclusive
_RTH_END   = dt.time(16, 0)   # 16:00 ET exclusive

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

# Default lookback: 90 calendar days of hourly bars.
DEFAULT_DAYS_BACK = 90

# SIP historical is free when query end is at least this many minutes in the past.
SIP_LAG_MINUTES = 15


def _cache_file(symbol: str) -> Path:
    """Per-symbol cache path so AAPL and MSFT don't overwrite each other."""
    return DATA_DIR / f"{symbol}_1h.csv"


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


def _filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only bars whose timestamp falls within NYSE regular trading hours
    (9:30 AM – 4:00 PM Eastern, exclusive of 16:00).

    The index must be tz-aware.  Bars outside this window are pre-market or
    after-hours; their volume is structurally different from regular-session
    volume and must not be mixed into vol/volume-ratio calculations.
    """
    et = df.index.tz_convert("America/New_York")
    mask = (et.time >= _RTH_START) & (et.time < _RTH_END)
    return df.loc[mask]


def _fetch_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
    feed: DataFeed,
) -> pd.DataFrame:
    """Fetch bars from Alpaca for the given feed and return a clean DataFrame."""
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=start,
        end=end,
        feed=feed,
    )
    bars = client.get_stock_bars(request)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df.index.name = "timestamp"
    return df[["open", "high", "low", "close", "volume"]].copy()


def load_bars(
    symbol: str = "AAPL",
    days_back: int = DEFAULT_DAYS_BACK,
    as_of: Optional[datetime] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Return (DataFrame of 1-hour RTH OHLCV bars, feed_name) for *symbol*.

    Parameters
    ----------
    symbol   : equity ticker, e.g. "AAPL" or "MSFT".
    days_back: calendar days of history to fetch.
    as_of    : tz-aware datetime; when provided, bars end at this point
               (historical mode).  Must be at least SIP_LAG_MINUTES in the
               past — always true for any past date.  When None, end is
               set to now − SIP_LAG_MINUTES (current mode).

    Feed order: SIP → IEX → local cache (cache fallback only in current mode).
    """
    historical = as_of is not None

    if historical:
        end   = as_of
        start = end - dt.timedelta(days=days_back)
        mode_label = f"historical as-of {end.isoformat()}"
    else:
        now   = datetime.now(tz=timezone.utc)
        end   = now - timedelta(minutes=SIP_LAG_MINUTES)
        start = end - dt.timedelta(days=days_back)
        mode_label = "current"

    try:
        client = _get_client()
    except EnvironmentError:
        if historical:
            raise   # no point falling back to a current-data cache for a historical query
        return _load_cache(symbol)

    # --- Attempt 1: SIP feed ---
    try:
        print(f"[data_loader] Fetching {days_back}-day hourly bars for {symbol} "
              f"via SIP ({mode_label}) …")
        df = _filter_rth(_fetch_bars(client, symbol, start, end, DataFeed.SIP))
        if not historical:
            df.to_csv(_cache_file(symbol))
            print(f"[data_loader] SIP: {len(df)} regular-session bars. Cache updated.")
        else:
            print(f"[data_loader] SIP: {len(df)} regular-session bars (historical; no cache write).")
        return df, "SIP"
    except Exception as sip_err:
        print(f"[data_loader] SIP feed failed ({sip_err}); trying IEX …")

    # --- Attempt 2: IEX fallback ---
    try:
        df = _filter_rth(_fetch_bars(client, symbol, start, end, DataFeed.IEX))
        if not historical:
            df.to_csv(_cache_file(symbol))
            print(f"[data_loader] IEX: {len(df)} regular-session bars. Cache updated.")
        else:
            print(f"[data_loader] IEX: {len(df)} regular-session bars (historical; no cache write).")
        return df, "IEX"
    except Exception as iex_err:
        print(f"[data_loader] IEX feed failed ({iex_err}); falling back to local cache …")

    # --- Attempt 3: local cache (current mode only) ---
    if historical:
        raise RuntimeError(
            f"All live feeds failed for historical query ({symbol} as-of {as_of}). "
            "Cannot fall back to a current-data cache for a backdated query."
        )
    return _load_cache(symbol)


def _load_cache(symbol: str) -> tuple[pd.DataFrame, str]:
    """Load bars for *symbol* from the local CSV cache.  Raises if no cache exists."""
    path = _cache_file(symbol)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached bar data found at {path} and all live feeds failed.\n"
            "Run with valid Alpaca credentials to populate the cache."
        )
    print(f"[data_loader] Using cached bars from {path}")
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    # Cache may have been written before RTH filtering was introduced; re-apply.
    if df.index.tz is not None:
        df = _filter_rth(df)
    return df, "cache"
