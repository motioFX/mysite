
"""Rebuild merged hourly datasets for top-volume Bitget symbols.
Generates selection scores, normalized charts, and merged CSVs.
"""
import argparse
import asyncio
import math
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import subprocess

import matplotlib
import numpy as np
import pandas as pd
import pybotters
import requests
from config_loader import get_webhook_url

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

REST_API_URL = {
    "bitget": "https://api.bitget.com",
    "bybit": "https://api.bybit.com",
}

JST = timezone(timedelta(hours=9))
_now_jst = datetime.now(JST).replace(minute=0, second=0, microsecond=0)
DEFAULT_START_JST = _now_jst - timedelta(days=30)
DEFAULT_START_STR = DEFAULT_START_JST.isoformat()

GRANULARITY_MS = {
    "1H": 60 * 60 * 1000,
    "2H": 2 * 60 * 60 * 1000,
    "4H": 4 * 60 * 60 * 1000,
    "6H": 6 * 60 * 60 * 1000,
    "12H": 12 * 60 * 60 * 1000,
    "1D": 24 * 60 * 60 * 1000,
}

NormalizedWindow = Tuple[str, Optional[timedelta], Optional[int]]

NORMALIZED_WINDOWS: Sequence[NormalizedWindow] = (
    ("30d", timedelta(days=30), 30 * 24),
    ("10d", timedelta(days=10), 10 * 24),
    ("5d", timedelta(days=5), 5 * 24),
)

DEFAULT_PRODUCT_TYPE = "USDT-FUTURES"
PRODUCT_TYPE_MAP = {
    "UMCBL": "USDT-FUTURES",
    "CMCBL": "COIN-FUTURES",
    "DMCBL": "USDC-FUTURES",
}
PRODUCT_TYPE_SUFFIX_MAP = {value: key for key, value in PRODUCT_TYPE_MAP.items()}


class send_discord:
    def __init__(self) -> None:
        self.webhook_url = get_webhook_url()
        self.bitget_webhook = get_webhook_url("bitget")
        self.bybit_webhook = get_webhook_url("bybit")

    def _get_target_webhooks(self, text: str) -> list[str]:
        if sys.platform == "win32" and self.webhook_url:
            return [self.webhook_url]

        if self.bitget_webhook:
            return [self.bitget_webhook]
        elif self.webhook_url:
            return [self.webhook_url]
        return []

    def send_message(self, content: str) -> None:
        webhooks = self._get_target_webhooks(content)
        for url in webhooks:
            try:
                requests.post(url, json={"content": content}, timeout=10).raise_for_status()
            except requests.RequestException:
                pass

    def send_file(self, file_path: Path, description: str) -> None:
        webhooks = self._get_target_webhooks(description)
        for url in webhooks:
            try:
                if not file_path.exists():
                    continue
                with file_path.open("rb") as fh:
                    files = {"file": (file_path.name, fh)}
                    data = {"content": description}
                    requests.post(url, data=data, files=files, timeout=30).raise_for_status()
            except requests.RequestException:
                pass


def to_ms(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def from_ms_jst(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(JST)


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}")


def log_fetch_result(label: str, rows: Sequence[Any]) -> None:
    count = len(rows) if hasattr(rows, "__len__") else 0
    if count == 0:
        log(f"{label}: no data returned (skip)")
    else:
        log(f"{label}: {count} rows")


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def normalize_product_type(product_type: str) -> str:
    if not product_type:
        return DEFAULT_PRODUCT_TYPE
    key = str(product_type).upper()
    if key in PRODUCT_TYPE_MAP:
        return PRODUCT_TYPE_MAP[key]
    if key in PRODUCT_TYPE_SUFFIX_MAP:
        return key
    return key


def normalize_symbol(bitget_symbol: str) -> str:
    sym = str(bitget_symbol).upper()
    if "_" in sym:
        base, suffix = sym.rsplit("_", 1)
        if suffix in PRODUCT_TYPE_MAP or suffix in ("SUMCBL",):
            return base
    return sym


def to_bybit_symbol(symbol: str) -> Optional[str]:
    base = normalize_symbol(symbol)
    if base.endswith("USDT"):
        return base
    return None


def fetch_bitget_tickers(product_type: str) -> List[dict]:
    normalized_product_type = normalize_product_type(product_type)
    url = f"{REST_API_URL['bitget']}/api/v2/mix/market/tickers"
    params = {"productType": normalized_product_type}
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()
    if not isinstance(data, dict) or data.get("code") != "00000":
        raise RuntimeError(f"Ticker fetch failed: {data}")
    return data.get("data") or []


def fetch_demo_available_symbols() -> set:
    """Fetch symbols available in Bitget demo (paper trading) mode."""
    url = f"{REST_API_URL['bitget']}/api/v2/mix/market/tickers"
    params = {"productType": "usdt-futures"}
    headers = {"paptrading": "1"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log(f"Demo symbols fetch error: {exc}")
        return {"BTCUSDT", "ETHUSDT"}  # Fallback
    
    if not isinstance(data, dict) or data.get("code") != "00000":
        log(f"Demo symbols fetch failed: {data}")
        return {"BTCUSDT", "ETHUSDT"}
    
    symbols = set()
    for ticker in data.get("data", []):
        symbol = normalize_symbol(ticker.get("symbol", ""))
        if symbol:
            symbols.add(symbol)
    
    log(f"Demo mode available symbols: {len(symbols)} (e.g. {list(symbols)[:5]})")
    return symbols


def fetch_bitget_top_usdt_symbols(
    limit: int,
    *,
    product_type: str,
    quote: str = "USDT",
) -> List[Tuple[str, float]]:
    tickers = fetch_bitget_tickers(product_type)
    quote = quote.upper()
    scored: List[Tuple[float, str]] = []
    for t in tickers:
        raw_symbol = str(t.get("symbol", "")).upper()
        base_symbol = normalize_symbol(raw_symbol)
        if quote and not base_symbol.endswith(quote):
            continue
        try:
            turnover = float(
                t.get("usdtVolume")
                or t.get("quoteVolume")
                or t.get("baseVolume")
                or 0.0
            )
        except (TypeError, ValueError):
            turnover = 0.0
        scored.append((turnover, base_symbol))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(limit, 0)]
    top_filtered = [(sym, turnover) for turnover, sym in top if turnover > 0]
    log(f"Bitget auto-selected symbols (top {limit} by quoted volume): {[sym for sym, _ in top_filtered]}")
    return top_filtered


def fetch_bybit_top_usdt_symbols(
    limit: int,
    *,
    quote: str = "USDT",
    category: str = "linear",
) -> List[Tuple[str, float]]:
    url = REST_API_URL["bybit"] + "/v5/market/tickers"
    params = {"category": category}
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()
    if not isinstance(data, dict) or data.get("retCode") not in (0, None):
        raise RuntimeError(f"Bybit ticker fetch failed: {data}")
    items = ((data.get("result") or {}).get("list") or [])
    quote = quote.upper()
    scored: List[Tuple[float, str]] = []
    for item in items:
        sym = str(item.get("symbol") or "").upper()
        if not sym.endswith(quote):
            continue
        try:
            turnover = float(item.get("turnover24h") or item.get("volume24h") or 0.0)
        except (TypeError, ValueError):
            turnover = 0.0
        scored.append((turnover, sym))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(limit, 0)]
    top_filtered = [(sym, turnover) for turnover, sym in top if turnover > 0]
    log(f"Bybit auto-selected symbols (top {limit} by quoted volume): {[sym for sym, _ in top_filtered]}")
    return top_filtered


def fetch_bybit_active_symbols(category: str = "linear") -> set:
    """Fetch all active symbols from Bybit to filter candidates."""
    url = REST_API_URL["bybit"] + "/v5/market/tickers"
    params = {"category": category}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or data.get("retCode") not in (0, None):
            log(f"Bybit active symbols fetch failed: {data}")
            return set()
        items = ((data.get("result") or {}).get("list") or [])
        symbols = {item.get("symbol").upper() for item in items if item.get("symbol")}
        log(f"Fetched {len(symbols)} active Bybit symbols.")
        return symbols
    except Exception as exc:
        log(f"Bybit active symbols fetch error: {exc}")
        return set()

async def _fetch_bitget_ohlcv_endpoint(
    symbol: str,
    product_type: str,
    granularity: str,
    start_ms: int,
    end_ms: int,
    endpoint: str,
) -> List[List[Any]]:
    url = endpoint
    limit = 200
    step_ms = GRANULARITY_MS.get(granularity, 60 * 60 * 1000)
    rows: List[List[Any]] = []
    req = 0
    async with pybotters.Client(base_url=REST_API_URL["bitget"]) as client:
        cur = start_ms
        while cur < end_ms:
            batch_end = min(cur + step_ms * limit - 1, end_ms)
            params = {
                "symbol": symbol,
                "productType": product_type,
                "granularity": granularity,
                "startTime": cur,
                "endTime": batch_end,
                "limit": limit,
            }
            # Retry loop for API errors
            max_retries = 5
            success = False
            for attempt in range(max_retries):
                try:
                    resp = await client.get(url, params=params)
                    data = await resp.json()
                except Exception as e:
                    log(f"BITGET OHLCV request failed (attempt {attempt+1}/{max_retries}): {e}")
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue

                if not isinstance(data, dict) or data.get("code") != "00000":
                    log(f"BITGET OHLCV error (attempt {attempt+1}/{max_retries}): {data}")
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                
                # Check if data is empty but technically success
                # Note: valid empty response means no data in range, which is fine to move on, 
                # but sometimes it's a transient issue. However, "00000" usually means success.
                # If data is empty list, we break and accept it.
                success = True
                break
            
            req += 1
            if req == 1 or req % 20 == 0:
                log(
                    f"BITGET OHLCV {symbol} req#{req} window "
                    f"{datetime.fromtimestamp(cur/1000, tz=timezone.utc)} -> "
                    f"{datetime.fromtimestamp(batch_end/1000, tz=timezone.utc)}"
                )

            if not success:
                log(f"BITGET OHLCV failed after {max_retries} retries. Skipping window.")
                cur = batch_end + 1
                await asyncio.sleep(0.05)
                continue

            items = data.get("data") or []
            if not items:
                # No data in this window, move to next
                cur = batch_end + 1
                await asyncio.sleep(0.05)
                continue
            rows.extend(items)
            last_ts = max(int(it[0]) for it in items)
            cur = max(batch_end + 1, last_ts + step_ms)
            await asyncio.sleep(0.02)
    return rows


async def fetch_bitget_ohlcv(
    symbol: str,
    product_type: str,
    granularity: str,
    start_ms: int,
    end_ms: int,
) -> List[List[Any]]:
    rows = await _fetch_bitget_ohlcv_endpoint(
        symbol,
        product_type,
        granularity,
        start_ms,
        end_ms,
        "/api/v2/mix/market/candles",
    )
    if rows:
        return rows
    log(f"BITGET OHLCV empty for {symbol} via candles; trying history-candles.")
    return await _fetch_bitget_ohlcv_endpoint(
        symbol,
        product_type,
        granularity,
        start_ms,
        end_ms,
        "/api/v2/mix/market/history-candles",
    )


async def fetch_bitget_funding_history_range(
    symbol: str,
    product_type: str,
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    url = "/api/v2/mix/market/history-fund-rate"
    rows: List[Dict[str, Any]] = []
    cursor = end_ms
    req = 0
    max_requests = 500
    async with pybotters.Client(base_url=REST_API_URL["bitget"]) as client:
        while cursor >= start_ms:
            params = {
                "symbol": symbol,
                "productType": product_type,
                "endTime": cursor,
                "startTime": start_ms,
                "limit": 200,
            }
            # Retry loop
            max_retries = 5
            success = False
            for attempt in range(max_retries):
                try:
                    resp = await client.get(url, params=params)
                    data = await resp.json()
                except Exception as e:
                    log(f"BITGET FUNDING request failed (attempt {attempt+1}/{max_retries}): {e}")
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue

                if not isinstance(data, dict) or data.get("code") != "00000":
                    log(f"BITGET FUNDING error (attempt {attempt+1}/{max_retries}): {data}")
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                
                success = True
                break

            req += 1
            if req == 1 or req % 10 == 0:
                log(
                    f"BITGET FUNDING {symbol} req#{req} scanning <= "
                    f"{datetime.fromtimestamp(cursor/1000, tz=timezone.utc)}"
                )

            if not success:
                log(f"BITGET FUNDING failed after {max_retries} retries. Stopping pagination.")
                break
            items = data.get("data") or []
            if not items:
                break
            min_ts = None
            for it in items:
                ts = int(it.get("fundingTime") or 0)
                if not ts or ts < start_ms or ts > end_ms:
                    continue
                rows.append(
                    {
                        "timestamp": from_ms_jst(ts),
                        "fundingRate": float(it.get("fundingRate", 0.0)),
                    }
                )
                if min_ts is None or ts < min_ts:
                    min_ts = ts
            if min_ts is None or min_ts <= start_ms:
                break
            next_cursor = min_ts - 1
            if next_cursor >= cursor:
                log("BITGET FUNDING pagination stalled; stopping to avoid infinite loop.")
                break
            cursor = next_cursor
            if req >= max_requests:
                log("BITGET FUNDING reached max requests; stopping pagination.")
                break
            await asyncio.sleep(0.05)
    rows.sort(key=lambda d: d["timestamp"])
    return rows


async def fetch_bitget_open_interest_snapshot(
    symbol: str,
    product_type: str,
) -> List[Dict[str, Any]]:
    url = "/api/v2/mix/market/open-interest"
    async with pybotters.Client(base_url=REST_API_URL["bitget"]) as client:
        # Retry loop
        for attempt in range(5):
            try:
                resp = await client.get(url, params={"symbol": symbol, "productType": product_type})
                data = await resp.json()
                if isinstance(data, dict) and data.get("code") == "00000":
                    break
            except Exception:
                pass
            await asyncio.sleep(1.0)
        else:
            # Last attempt failed or data invalid
            if 'data' not in locals() or not isinstance(data, dict):
                data = {}
            if data.get("code") != "00000":
                 log(f"BITGET OI error: {data}")
                 return []
            
    if not isinstance(data, dict) or data.get("code") != "00000":
        log(f"BITGET OI error: {data}")
        return []
    payload = data.get("data") or {}
    ts = int(payload.get("ts") or 0)
    if not ts:
        return []
    oi_list = payload.get("openInterestList") or []
    size = None
    for item in oi_list:
        if str(item.get("symbol", "")).upper() == symbol.upper():
            size = item.get("size")
            break
    if size is None and oi_list:
        size = oi_list[0].get("size")
    try:
        oi_val = float(size) if size is not None else 0.0
    except (TypeError, ValueError):
        oi_val = 0.0
    return [{"timestamp": from_ms_jst(ts), "openInterest": oi_val}]


def determine_market_state(mean_norm: float, mean_cb_premium: float) -> str:
    if mean_norm >= 1.0 or mean_cb_premium > 0.0:
        return "long_only"
    else:
        return "short_only"


async def fetch_coinbase_candles_1h(start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    rows: List[Dict[str, Any]] = []
    step_ms = 300 * 60 * 60 * 1000
    cur_end = end_ms
    headers = {"User-Agent": "Mozilla/5.0"}
    async with pybotters.Client(headers=headers) as client:
        while cur_end > start_ms:
            cur_start = max(cur_end - step_ms, start_ms)
            start_iso = datetime.fromtimestamp(cur_start / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            end_iso = datetime.fromtimestamp(cur_end / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            params = {
                "granularity": "3600",
                "start": start_iso,
                "end": end_iso
            }
            max_retries = 5
            success = False
            for attempt in range(max_retries):
                try:
                    resp = await client.get(url, params=params)
                    data = await resp.json()
                    if isinstance(data, list):
                        success = True
                        break
                    await asyncio.sleep(1.5 * (attempt + 1))
                except Exception:
                    await asyncio.sleep(1.5 * (attempt + 1))
            
            if not success:
                cur_end = cur_start - 1
                continue
                
            for it in data:
                ts_sec = int(it[0])
                rows.append({
                    "timestamp": from_ms_jst(ts_sec * 1000),
                    "coinbase_close": float(it[4])
                })
            cur_end = cur_start - 1
            await asyncio.sleep(0.15)
    rows.sort(key=lambda d: d["timestamp"])
    return rows



async def fetch_bybit_ohlcv_1h(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    category: str = "linear",
) -> List[Dict[str, Any]]:
    url = REST_API_URL["bybit"] + "/v5/market/kline"
    limit = 1000
    step_ms = 60 * 60 * 1000
    rows: List[Dict[str, Any]] = []
    req = 0
    async with pybotters.Client() as client:
        cur = start_ms
        while cur < end_ms:
            batch_end = min(cur + step_ms * limit - 1, end_ms)
            params = {
                "category": category,
                "symbol": symbol,
                "interval": "60",
                "start": cur,
                "end": batch_end,
                "limit": limit,
            }
            
            # Retry loop
            max_retries = 5
            success = False
            for attempt in range(max_retries):
                try:
                    resp = await client.get(url, params=params)
                    data = await resp.json()
                except Exception as e:
                    log(f"BYBIT OHLCV request failed (attempt {attempt+1}/{max_retries}): {e}")
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue

                if data.get("retCode") != 0:
                    msg = str(data.get("retMsg", ""))
                    if data.get("retCode") == 10001 or "invalid" in msg.lower() or "not supported" in msg.lower():
                        log(f"BYBIT OHLCV skip: {symbol} not found or invalid ({msg})")
                        success = True # Treat as "handled" so we don't retry or log failure
                        # But we return empty for this batch
                        data["result"] = {} 
                        break

                    log(f"BYBIT OHLCV error (attempt {attempt+1}/{max_retries}): {data.get('retMsg')}")
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                
                success = True
                break

            req += 1
            if req == 1 or req % 20 == 0:
                log(
                    f"BYBIT OHLCV {symbol} req#{req} window "
                    f"{datetime.fromtimestamp(cur/1000, tz=timezone.utc)} -> "
                    f"{datetime.fromtimestamp(batch_end/1000, tz=timezone.utc)}"
                )

            if not success:
                log(f"BYBIT OHLCV failed after {max_retries} retries. Skipping window.")
                cur = batch_end + 1
                await asyncio.sleep(0.05)
                continue


            items = ((data.get("result") or {}).get("list") or [])
            if not items:
                cur = batch_end + 1
                await asyncio.sleep(0.05)
                continue
            for it in items:
                ts = int(it[0])
                rows.append(
                    {
                        "timestamp": from_ms_jst(ts),
                        "bybit_open": float(it[1]),
                        "bybit_high": float(it[2]),
                        "bybit_low": float(it[3]),
                        "bybit_close": float(it[4]),
                        "bybit_volume": float(it[5]) if len(it) > 5 else 0.0,
                    }
                )
            last_ts = max(int(it[0]) for it in items)
            cur = max(batch_end + 1, last_ts + step_ms)
            await asyncio.sleep(0.02)
    return rows


async def fetch_bybit_funding_history(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    category: str = "linear",
) -> List[Dict[str, Any]]:
    url = REST_API_URL["bybit"] + "/v5/market/funding/history"
    rows: List[Dict[str, Any]] = []
    cursor = end_ms
    req = 0
    async with pybotters.Client() as client:
        while True:
            params = {
                "category": category,
                "symbol": symbol,
                "endTime": str(cursor),
                "limit": str(200),
            }
            # Retry loop
            max_retries = 5
            success = False
            for attempt in range(max_retries):
                try:
                    resp = await client.get(url, params=params)
                    data = await resp.json()
                except Exception as e:
                    log(f"BYBIT FUNDING request failed (attempt {attempt+1}/{max_retries}): {e}")
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue

                if data.get("retCode") != 0:
                    msg = str(data.get("retMsg", ""))
                    if data.get("retCode") == 10001 or "invalid" in msg.lower() or "not supported" in msg.lower():
                        log(f"BYBIT FUNDING skip: {symbol} not found or invalid ({msg})")
                        success = True
                        data["result"] = {} 
                        break
                    
                    log(f"BYBIT FUNDING error (attempt {attempt+1}/{max_retries}): {data.get('retMsg')}")
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                
                success = True
                break

            req += 1
            if req == 1 or req % 10 == 0:
                log(
                    f"BYBIT FUNDING {symbol} req#{req} scanning <= "
                    f"{datetime.fromtimestamp(cursor/1000, tz=timezone.utc)}"
                )

            if not success:
                log(f"BYBIT FUNDING failed after {max_retries} retries. Stopping pagination.")
                break
            items = ((data.get("result") or {}).get("list") or [])
            if not items:
                break
            for it in items:
                ts = int(it.get("fundingRateTimestamp") or it.get("fundingTime") or it.get("timestamp") or 0)
                if not ts or ts < start_ms:
                    continue
                rows.append(
                    {
                        "timestamp": from_ms_jst(ts),
                        "bybit_fundingRate": float(it.get("fundingRate", 0.0)),
                    }
                )
            last_ts = int(items[-1].get("fundingRateTimestamp") or items[-1].get("fundingTime") or items[-1].get("timestamp") or 0)
            if last_ts <= start_ms:
                break
            cursor = last_ts - 1
            await asyncio.sleep(0.1)
    rows.sort(key=lambda d: d["timestamp"])
    return rows


async def fetch_bybit_open_interest_1h(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    category: str = "linear",
) -> List[Dict[str, Any]]:
    url = REST_API_URL["bybit"] + "/v5/market/open-interest"
    limit = 200
    step_ms = 60 * 60 * 1000
    rows: List[Dict[str, Any]] = []
    req = 0
    async with pybotters.Client() as client:
        cur = start_ms
        while cur < end_ms:
            batch_end = min(cur + step_ms * limit - 1, end_ms)
            params = {
                "category": category,
                "symbol": symbol,
                "intervalTime": "1h",
                "startTime": cur,
                "endTime": batch_end,
                "limit": limit,
            }
            # Retry loop
            max_retries = 5
            success = False
            for attempt in range(max_retries):
                try:
                    resp = await client.get(url, params=params)
                    data = await resp.json()
                except Exception as e:
                    log(f"BYBIT OI request failed (attempt {attempt+1}/{max_retries}): {e}")
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue

                if data.get("retCode") != 0:
                    msg = str(data.get("retMsg", ""))
                    if data.get("retCode") == 10001 or "invalid" in msg.lower() or "not supported" in msg.lower():
                        log(f"BYBIT OI skip: {symbol} not found or invalid ({msg})")
                        success = True
                        data["result"] = {} 
                        break

                    log(f"BYBIT OI error (attempt {attempt+1}/{max_retries}): {data.get('retMsg')}")
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                
                success = True
                break

            req += 1
            if req == 1 or req % 10 == 0:
                log(
                    f"BYBIT OI {symbol} req#{req} window "
                    f"{datetime.fromtimestamp(cur/1000, tz=timezone.utc)} -> "
                    f"{datetime.fromtimestamp(batch_end/1000, tz=timezone.utc)}"
                )
            
            if not success:
                log(f"BYBIT OI failed after {max_retries} retries. Skipping window.")
                cur = batch_end + 1
                await asyncio.sleep(0.1)
                continue
            items = ((data.get("result") or {}).get("list") or [])
            if not items:
                cur = batch_end + 1
                await asyncio.sleep(0.05)
                continue
            for it in items:
                ts = int(it.get("timestamp") or 0)
                if not ts:
                    continue
                rows.append(
                    {
                        "timestamp": from_ms_jst(ts),
                        "bybit_openInterest": float(it.get("openInterest", 0.0)),
                    }
                )
            last_ts = max(int(it.get("timestamp", 0)) for it in items if it.get("timestamp"))
            cur = max(batch_end + 1, last_ts + step_ms)
            await asyncio.sleep(0.05)
    return rows

def compute_hv_from_close(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df_hv = df[["timestamp", "close"]].copy()
    df_hv = df_hv.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df_hv.empty:
        return df_hv
    df_hv["log_ret"] = np.log(df_hv["close"]).diff()
    df_hv.loc[df_hv.index[0], "log_ret"] = 0.0
    sqrt_annual = math.sqrt(24 * 365)
    df_hv["hv_7d"] = df_hv["log_ret"].rolling(24 * 7).std(ddof=0) * sqrt_annual
    df_hv["hv_30d"] = df_hv["log_ret"].rolling(24 * 30).std(ddof=0) * sqrt_annual
    return df_hv


async def build_merged_dataset(
    symbol: str,
    start_utc: datetime,
    end_utc: datetime,
    *,
    out_dir: Path,
    product_type: str,
    granularity: str,
    rank: Optional[int] = None,
    premium_dict: Optional[Dict[datetime, float]] = None,
) -> Tuple[pd.DataFrame, Path]:
    start_ms = to_ms(start_utc)
    end_ms = to_ms(end_utc)
    log(f"Start unified fetch for {symbol} {start_utc.isoformat()} -> {end_utc.isoformat()}")

    ohlcv_task = asyncio.create_task(
        fetch_bitget_ohlcv(symbol, product_type, granularity, start_ms, end_ms)
    )
    funding_task = asyncio.create_task(
        fetch_bitget_funding_history_range(symbol, product_type, start_ms, end_ms)
    )
    oi_task = asyncio.create_task(fetch_bitget_open_interest_snapshot(symbol, product_type))

    bybit_symbol = to_bybit_symbol(symbol)
    bybit_ohlcv_task = None
    bybit_funding_task = None
    bybit_oi_task = None

    if bybit_symbol is not None:
        bybit_ohlcv_task = asyncio.create_task(
            fetch_bybit_ohlcv_1h(bybit_symbol, start_ms, end_ms)
        )
        bybit_funding_task = asyncio.create_task(
            fetch_bybit_funding_history(bybit_symbol, start_ms, end_ms)
        )
        bybit_oi_task = asyncio.create_task(
            fetch_bybit_open_interest_1h(bybit_symbol, start_ms, end_ms)
        )

    tasks = [ohlcv_task, funding_task, oi_task]
    if bybit_symbol is not None:
        tasks.extend([bybit_ohlcv_task, bybit_funding_task, bybit_oi_task])

    results = await asyncio.gather(*tasks)

    ohlcv_rows = results[0]
    fund_rows = results[1]
    oi_rows = results[2]

    bybit_ohlcv_rows = []
    bybit_fund_rows = []
    bybit_oi_rows = []

    if bybit_symbol is not None:
        bybit_ohlcv_rows = results[3]
        bybit_fund_rows = results[4]
        bybit_oi_rows = results[5]

    log_fetch_result("Bitget OHLCV", ohlcv_rows)
    log_fetch_result("Bitget Funding", fund_rows)
    log_fetch_result("Bitget OI", oi_rows)
    log_fetch_result("Bybit OHLCV", bybit_ohlcv_rows)
    log_fetch_result("Bybit OI", bybit_oi_rows)
    log_fetch_result("Bybit Funding", bybit_fund_rows)

    if not ohlcv_rows:
        raise RuntimeError(f"No OHLCV data returned from Bitget for {symbol}.")

    price_records = []
    for it in ohlcv_rows:
        ts = int(it[0])
        price_records.append(
            {
                "timestamp": from_ms_jst(ts),
                "open": float(it[1]),
                "high": float(it[2]),
                "low": float(it[3]),
                "close": float(it[4]),
                "volume": float(it[5]) if len(it) > 5 else 0.0,
                "qv": float(it[6]) if len(it) > 6 else float(it[5]) if len(it) > 5 else 0.0,
            }
        )
    df_price = (
        pd.DataFrame.from_records(price_records)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    log(f"OHLCV data prepared ({symbol}): {len(df_price)} rows")

    df = df_price.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if fund_rows:
        df_funding = (
            pd.DataFrame(fund_rows)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
        )
        df_funding["timestamp"] = pd.to_datetime(df_funding["timestamp"])
        df_funding = (
            df_funding.set_index("timestamp")
            .resample("1h")
            .last()
            .ffill()
            .reset_index()
        )
        df = pd.merge(df, df_funding, on="timestamp", how="left")
        log("Merged funding data")

    if oi_rows:
        df_oi = (
            pd.DataFrame(oi_rows)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
        )
        df_oi["timestamp"] = pd.to_datetime(df_oi["timestamp"])
        df_oi = (
            df_oi.set_index("timestamp")
            .resample("1h")
            .last()
            .ffill()
            .reset_index()
        )
        df = pd.merge(df, df_oi, on="timestamp", how="left")
        log("Merged open interest data")

    if bybit_ohlcv_rows:
        df_bybit_ohlcv = (
            pd.DataFrame(bybit_ohlcv_rows)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
        )
        df_bybit_ohlcv["timestamp"] = pd.to_datetime(df_bybit_ohlcv["timestamp"])
        df_bybit_ohlcv = (
            df_bybit_ohlcv.set_index("timestamp")
            .resample("1h")
            .last()
            .ffill()
            .reset_index()
        )
        df = pd.merge(df, df_bybit_ohlcv, on="timestamp", how="left")
        log("Merged Bybit OHLCV data")

    if bybit_fund_rows:
        df_bybit_funding = (
            pd.DataFrame(bybit_fund_rows)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
        )
        df_bybit_funding["timestamp"] = pd.to_datetime(df_bybit_funding["timestamp"])
        df_bybit_funding = (
            df_bybit_funding.set_index("timestamp")
            .resample("1h")
            .last()
            .ffill()
            .reset_index()
        )
        df = pd.merge(df, df_bybit_funding, on="timestamp", how="left")
        log("Merged Bybit funding data")

    if bybit_oi_rows:
        df_bybit_oi = (
            pd.DataFrame(bybit_oi_rows)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
        )
        df_bybit_oi["timestamp"] = pd.to_datetime(df_bybit_oi["timestamp"])
        df_bybit_oi = (
            df_bybit_oi.set_index("timestamp")
            .resample("1h")
            .last()
            .ffill()
            .reset_index()
        )
        df = pd.merge(df, df_bybit_oi, on="timestamp", how="left")
        log("Merged Bybit open interest data")

    df_hv = compute_hv_from_close(df)
    if not df_hv.empty:
        df_hv = df_hv[["timestamp", "log_ret", "hv_7d", "hv_30d"]]
        df_hv["timestamp"] = pd.to_datetime(df_hv["timestamp"])
        df = pd.merge(df, df_hv, on="timestamp", how="left")
        log("Merged HV data")

    # Fallback mechanism: populate Bybit-prefixed columns from the fetched Bybit data.
    # If the Bybit data is missing or returns NaN, fall back to the corresponding Bitget column values.
    for bybit_col, bitget_col in [
        ("bybit_open", "open"),
        ("bybit_high", "high"),
        ("bybit_low", "low"),
        ("bybit_close", "close"),
        ("bybit_volume", "volume"),
        ("bybit_openInterest", "openInterest"),
        ("bybit_fundingRate", "fundingRate"),
    ]:
        if bitget_col not in df.columns:
            df[bitget_col] = 0.0
        
        if bybit_col in df.columns:
            df[bybit_col] = df[bybit_col].fillna(df[bitget_col])
        else:
            df[bybit_col] = df[bitget_col]
    log("Populated Bybit columns with fallback to Bitget data")

    df = ensure_columns(
        df,
        [
            "fundingRate",
            "openInterest",
            "log_ret",
            "hv_7d",
            "hv_30d",
            "bybit_open",
            "bybit_high",
            "bybit_low",
            "bybit_close",
            "bybit_volume",
            "bybit_openInterest",
            "bybit_fundingRate",
        ],
    )

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["fundingRate"] = df["fundingRate"].ffill()
    df["openInterest"] = df["openInterest"].ffill()
    df["bybit_fundingRate"] = df["bybit_fundingRate"].ffill()

    # コインベースプレミアムをマージ
    df["coinbase_premium"] = 0.0
    if premium_dict is not None:
        df["coinbase_premium"] = df["timestamp"].map(premium_dict).fillna(0.0)

    start_jst = start_utc.astimezone(JST)
    end_jst = end_utc.astimezone(JST)
    effective_start_jst = max(start_jst, DEFAULT_START_JST)
    mask = (df["timestamp"] >= effective_start_jst) & (df["timestamp"] <= end_jst)
    df = df.loc[mask].reset_index(drop=True)

    df["symbol"] = symbol
    df = df[
        [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "qv",
            "bybit_open",
            "bybit_high",
            "bybit_low",
            "bybit_close",
            "bybit_volume",
            "fundingRate",
            "bybit_fundingRate",
            "openInterest",
            "bybit_openInterest",
            "log_ret",
            "hv_7d",
            "hv_30d",
            "coinbase_premium",
        ]
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    start_tag = start_utc.strftime("%Y%m%d")
    end_tag = end_utc.strftime("%Y%m%d")
    rank_prefix = f"{rank:02d}_" if rank is not None else ""
    file_name = f"{start_tag}_{end_tag}_{rank_prefix}{symbol}_Merged_All_Data_Bitget.csv"
    out_path = out_dir / file_name
    df.to_csv(out_path, index=False)
    log(f"Saved merged data: {len(df)} rows -> {out_path}")
    return df, out_path


async def build_all_symbols(
    symbols: Sequence[str],
    start_utc: datetime,
    end_utc: datetime,
    *,
    out_dir: Path,
    product_type: str,
    granularity: str,
    ranks: Optional[Mapping[str, int]] = None,
) -> List[Tuple[str, Path, int]]:
    # Coinbase プレミアムデータを事前に一括取得する
    log("=== Downloading Coinbase Premium Data ===")
    start_ms = to_ms(start_utc)
    end_ms = to_ms(end_utc)
    premium_dict = {}
    try:
        # Coinbase BTC-USD
        cb_rows = await fetch_coinbase_candles_1h(start_ms, end_ms)
        df_cb = pd.DataFrame(cb_rows).drop_duplicates(subset=["timestamp"])
        
        # Bybit BTCUSDT
        bybit_btc_ohlcv = await fetch_bybit_ohlcv_1h("BTCUSDT", start_ms, end_ms)
        df_bb_btc = pd.DataFrame(bybit_btc_ohlcv).drop_duplicates(subset=["timestamp"])
        
        df_premium_calc = pd.merge(
            df_cb,
            df_bb_btc[["timestamp", "bybit_close"]].rename(columns={"bybit_close": "bybit_btc_close"}),
            on="timestamp",
            how="inner"
        )
        df_premium_calc["coinbase_premium"] = (df_premium_calc["coinbase_close"] - df_premium_calc["bybit_btc_close"]) / df_premium_calc["bybit_btc_close"] * 100
        premium_dict = df_premium_calc.set_index("timestamp")["coinbase_premium"].to_dict()
        log(f"Coinbase Premium data prepared ({len(premium_dict)} data points)")
    except Exception as e:
        log(f"Failed to fetch Coinbase Premium reference data: {e}. Defaulting to 0.0")

    results: List[Tuple[str, Path, int]] = []
    for symbol in symbols:
        rank = (ranks or {}).get(symbol)
        try:
            if rank is not None:
                log(f"=== Building dataset for #{rank:02d} {symbol} ===")
            else:
                log(f"=== Building dataset for {symbol} ===")
            df, out_path = await build_merged_dataset(
                symbol,
                start_utc,
                end_utc,
                out_dir=out_dir,
                product_type=product_type,
                granularity=granularity,
                rank=rank,
                premium_dict=premium_dict,
            )
        except Exception as exc:
            log(f"Build failed for {symbol}: {exc}")
            continue
        results.append((symbol, out_path, len(df)))
    return results


def load_close_series(csv_path: Path) -> pd.Series:
    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    except Exception as exc:
        log(f"Failed to read {csv_path}: {exc}")
        return pd.Series(dtype=float)
    if df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)
    series = df.set_index("timestamp")["close"].astype(float)
    return series.dropna()


def generate_normalized_datasets(
    ranked_records: Sequence[Tuple[int, str, Path]],
    end_utc: datetime,
    *,
    out_dir: Path,
) -> List[Tuple[str, Path, pd.DataFrame]]:
    results: List[Tuple[str, Path, pd.DataFrame]] = []
    normalized_dir = out_dir / "Normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    end_jst = end_utc.astimezone(JST)

    series_entries: List[Tuple[int, str, pd.Series]] = []
    for rank, symbol, csv_path in ranked_records:
        series = load_close_series(csv_path)
        if series.empty:
            continue
        series_entries.append((rank, symbol, series))

    if not series_entries:
        return results

    for window_label, delta, expected_points in NORMALIZED_WINDOWS:
        if delta is None:
            start_jst = DEFAULT_START_JST
        else:
            start_jst = end_jst - delta
        series_list: List[Tuple[str, pd.Series]] = []
        for _rank, symbol, series in series_entries:
            window_series = series.loc[
                (series.index >= start_jst) & (series.index <= end_jst)
            ]
            if window_series.empty:
                continue
            if delta is None and window_series.index[0] > start_jst:
                continue
            if expected_points is not None and len(window_series) < expected_points:
                continue
            if expected_points is not None:
                window_series = window_series.iloc[-expected_points:]
            base = float(window_series.iloc[0])
            if not math.isfinite(base) or base == 0.0:
                continue
            normalized_series = window_series / base
            normalized_series.name = symbol
            series_list.append((symbol, normalized_series))
        if not series_list:
            continue
        combined = pd.concat([series for _, series in series_list], axis=1, join="inner")
        combined.columns = [symbol for symbol, _ in series_list]
        combined = combined.sort_index()
        combined = combined.dropna(how="all")
        if combined.empty:
            continue
        output_df = combined.reset_index().rename(columns={"index": "timestamp"})
        start_tag = output_df["timestamp"].iloc[0].strftime("%Y%m%d")
        end_tag = output_df["timestamp"].iloc[-1].strftime("%Y%m%d")
        csv_name = f"top_close_normalized_{window_label}_{start_tag}_{end_tag}.csv"
        csv_path = normalized_dir / csv_name
        output_df.to_csv(csv_path, index=False)
        log(f"Saved normalized dataset ({window_label}, {len(output_df)} rows) -> {csv_path}")
        results.append((window_label, csv_path, output_df))
    return results


def plot_normalized_dataset(
    window_label: str,
    df: pd.DataFrame,
    *,
    out_dir: Path,
    title_prefix: str = "",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamps = pd.to_datetime(df["timestamp"])
    value_columns = [col for col in df.columns if col != "timestamp"]
    if not value_columns:
        raise ValueError("No value columns available for plotting.")
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in value_columns:
        ax.plot(timestamps, df[column], label=column, linewidth=1.2)
    title = f"Top 20 Normalized Close ({window_label})"
    if title_prefix:
        title = f"[{title_prefix}] {title}"
    ax.set_title(title)
    ax.set_ylabel("Normalized Close (base = 1)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.autofmt_xdate()

    time_span = timestamps.iloc[-1] - timestamps.iloc[0]
    if isinstance(time_span, pd.Timedelta):
        offset = max(pd.Timedelta(hours=1), time_span / 30 if time_span > pd.Timedelta(0) else pd.Timedelta(hours=1))
    else:
        offset = pd.Timedelta(hours=1)
    x_text = timestamps.iloc[-1] + offset
    ax.set_xlim(timestamps.iloc[0], x_text + offset)

    for column in value_columns:
        y_val = df[column].iloc[-1]
        if not pd.isna(y_val) and math.isfinite(float(y_val)):
            ax.text(
                x_text,
                float(y_val),
                column,
                fontsize=8,
                va="center",
                ha="left",
            )

    fig.tight_layout()
    start_tag = timestamps.iloc[0].strftime("%Y%m%d")
    end_tag = timestamps.iloc[-1].strftime("%Y%m%d")
    png_name = f"top20_close_normalized_{window_label}_{start_tag}_{end_tag}.png"
    png_path = out_dir / png_name
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log(f"Saved normalized plot ({window_label}) -> {png_path}")
    return png_path

def compute_symbol_metrics(
    symbol: str,
    csv_path: Path,
    lookback_days: int = 5,
    btc_df: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, float]]:
    """各シンボルの CSV から指定期間ベースの指標をまとめて計算。"""
    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=["timestamp"],
        )
    except FileNotFoundError:
        log(f"Metrics skip: file not found for {symbol} -> {csv_path}")
        return None
    except Exception as exc:
        log(f"Metrics skip: failed to read {csv_path}: {exc}")
        return None

    if df.empty:
        return None

    if "timestamp" not in df.columns or "close" not in df.columns:
        return None

    df = df.sort_values("timestamp").reset_index(drop=True)

    # 必須列がなければ空で追加（NaN）
    for col in ("high", "low", "volume", "qv", "openInterest", "fundingRate"):
        if col not in df.columns:
            df[col] = np.nan

    end_ts = df["timestamp"].max()
    start_w = end_ts - timedelta(days=lookback_days)
    df_w = df[df["timestamp"] >= start_w].copy()

    if df_w.empty or len(df_w) < 2:
        return None

    def calc_metrics(prefix: str, col_close: str, col_vol: str, col_oi: str, col_fund: str) -> Dict[str, float]:
        res = {
            f"norm_perf_5d{prefix}": float("nan"),
            f"norm_perf_5d_btc{prefix}": float("nan"),
            f"momentum_dir_5d{prefix}": float("nan"),
            f"max_drawdown_5d{prefix}": float("nan"),
            f"volume_change_5d{prefix}": float("nan"),
            f"oi_change_5d{prefix}": float("nan"),
            f"funding_avg_5d{prefix}": float("nan"),
        }
        if col_close not in df_w.columns:
            return res
            
        close_series = df_w[col_close].astype(float).dropna()
        if close_series.empty or len(close_series) < 2:
            return res
            
        p5_first = float(close_series.iloc[0])
        p5_last = float(close_series.iloc[-1])
        price_dir = float(np.sign(p5_last - p5_first))
        if p5_first > 0:
            res[f"norm_perf_5d{prefix}"] = p5_last / p5_first
            # モメンタム: 前時間比で上昇した割合 (バックテストと同一方式)
            diffs = close_series.diff().dropna()
            res[f"momentum_dir_5d{prefix}"] = float((diffs > 0).mean()) if len(diffs) > 0 else 0.5
            
            rolling_max_5 = close_series.cummax()
            drawdowns = (close_series / rolling_max_5 - 1.0) * 100.0
            res[f"max_drawdown_5d{prefix}"] = float(drawdowns.min()) if not drawdowns.empty else float("nan")

        # BTC-relative performance if btc_df is provided
        if btc_df is not None and col_close in btc_df.columns:
            btc_ts = pd.to_datetime(btc_df["timestamp"])
            btc_w = btc_df[(btc_ts >= start_w) & (btc_ts <= end_ts)].sort_values("timestamp")
            if col_close in btc_w.columns:
                btc_close_series = btc_w[col_close].astype(float).dropna()
                if len(btc_close_series) >= 2:
                    btc_close_first = float(btc_close_series.iloc[0])
                    btc_close_last = float(btc_close_series.iloc[-1])
                    if btc_close_first > 0 and btc_close_last > 0 and p5_first > 0:
                        res[f"norm_perf_5d_btc{prefix}"] = (p5_last / p5_first) / (btc_close_last / btc_close_first)

        if col_vol in df_w.columns and not df_w[col_vol].isna().all():
            vol_series = df_w[col_vol].astype(float).replace(0.0, np.nan).dropna()
            if len(vol_series) >= 4:
                mid = len(vol_series) // 2
                vol_first_half = float(vol_series.iloc[:mid].mean())
                vol_second_half = float(vol_series.iloc[mid:].mean())
                if vol_first_half > 0:
                    res[f"volume_change_5d{prefix}"] = (vol_second_half / vol_first_half - 1.0) * 100.0 * price_dir

        if col_oi in df_w.columns and not df_w[col_oi].isna().all():
            oi_series = df_w[col_oi].astype(float).replace(0.0, np.nan).dropna()
            if len(oi_series) >= 2:
                oi_first = float(oi_series.iloc[0])
                oi_last = float(oi_series.iloc[-1])
                if oi_first > 0:
                    res[f"oi_change_5d{prefix}"] = (oi_last / oi_first - 1.0) * 100.0 * price_dir

        if col_fund in df_w.columns and not df_w[col_fund].isna().all():
            fund_series = df_w[col_fund].astype(float).replace(0.0, np.nan).dropna()
            res[f"funding_avg_5d{prefix}"] = float(fund_series.mean()) if not fund_series.empty else float("nan")

        return res

    bg_vol_col = "qv" if ("qv" in df_w.columns and not df_w["qv"].isna().all()) else "volume"
    bg_metrics = calc_metrics("_bg", "close", bg_vol_col, "openInterest", "fundingRate")
    
    bb_vol_col = "bybit_volume"
    bb_metrics = calc_metrics("_bb", "bybit_close", bb_vol_col, "bybit_openInterest", "bybit_fundingRate")

    # Calculate daily turnover for Bitget
    if bg_vol_col == "qv" and not df_w["qv"].isna().all():
        bg_daily_turnover = float(df_w["qv"].dropna().mean() * 24)
    else:
        bg_daily_turnover = float((df_w["volume"] * df_w["close"]).dropna().mean() * 24)

    # Calculate daily turnover for Bybit
    if "bybit_volume" in df_w.columns and "bybit_close" in df_w.columns:
        bb_daily_turnover = float((df_w["bybit_volume"] * df_w["bybit_close"]).dropna().mean() * 24)
    else:
        bb_daily_turnover = 0.0

    cb_premium_val = float(df_w["coinbase_premium"].iloc[-1]) if "coinbase_premium" in df_w.columns and not df_w["coinbase_premium"].empty else 0.0

    return {
        "symbol": symbol,
        "bg_daily_turnover": bg_daily_turnover,
        "bb_daily_turnover": bb_daily_turnover,
        "coinbase_premium_bg": cb_premium_val,
        "coinbase_premium_bb": cb_premium_val,
        **bg_metrics,
        **bb_metrics,
    }


def compute_rolling_scores(valid_results: Sequence[Tuple[str, Path, int]]) -> None:
    log("=== Computing Rolling Scores ===")
    symbol_dfs = {}
    for symbol, path, rows in valid_results:
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            symbol_dfs[symbol] = df
        except Exception as exc:
            log(f"Failed to load {symbol} for rolling scores: {exc}")
            
    if not symbol_dfs:
        return

    all_timestamps = sorted(list(set().union(*(df["timestamp"] for df in symbol_dfs.values()))))
    
    btc_df_aligned = None
    if "BTCUSDT" in symbol_dfs:
        btc_df_aligned = symbol_dfs["BTCUSDT"].set_index("timestamp").reindex(all_timestamps)
        btc_df_aligned["close"] = btc_df_aligned["close"].ffill().bfill()
        btc_df_aligned["bybit_close"] = btc_df_aligned["bybit_close"].ffill().bfill()

    metrics = {
        "perf_bg": {}, "perf_btc_bg": {}, "momentum_bg": {}, "dd_bg": {}, "vol_bg": {}, "oi_bg": {}, "funding_bg": {}, "cb_premium_bg": {},
        "perf_bb": {}, "perf_btc_bb": {}, "momentum_bb": {}, "dd_bb": {}, "vol_bb": {}, "oi_bb": {}, "funding_bb": {}, "cb_premium_bb": {}
    }
    
    for symbol, df in symbol_dfs.items():
        df_aligned = df.set_index("timestamp").reindex(all_timestamps)
        
        df_aligned["close"] = df_aligned["close"].ffill().bfill()
        df_aligned["bybit_close"] = df_aligned["bybit_close"].ffill().bfill()
        df_aligned["volume"] = df_aligned["volume"].fillna(0.0)
        df_aligned["qv"] = df_aligned["qv"].fillna(0.0)
        df_aligned["bybit_volume"] = df_aligned["bybit_volume"].fillna(0.0)
        df_aligned["openInterest"] = df_aligned["openInterest"].ffill().bfill()
        df_aligned["bybit_openInterest"] = df_aligned["bybit_openInterest"].ffill().bfill()
        df_aligned["fundingRate"] = df_aligned["fundingRate"].fillna(0.0)
        df_aligned["bybit_fundingRate"] = df_aligned["bybit_fundingRate"].fillna(0.0)
        df_aligned["coinbase_premium"] = df_aligned["coinbase_premium"].ffill().bfill().fillna(0.0)
        
        bg_vol_col = "qv" if (not df_aligned["qv"].isna().all() and (df_aligned["qv"] > 0).any()) else "volume"
        
        price_dir_bg = np.sign(df_aligned["close"] - df_aligned["close"].shift(120)).fillna(0.0)
        
        metrics["perf_bg"][symbol] = df_aligned["close"] / df_aligned["close"].shift(120)
        
        if btc_df_aligned is not None:
            btc_close_bg = btc_df_aligned["close"]
            btc_close_bb = btc_df_aligned["bybit_close"]
        else:
            btc_close_bg = pd.Series(1.0, index=all_timestamps)
            btc_close_bb = pd.Series(1.0, index=all_timestamps)
            
        alt_btc_bg = df_aligned["close"] / btc_close_bg
        metrics["perf_btc_bg"][symbol] = alt_btc_bg / alt_btc_bg.shift(120)

        metrics["momentum_bg"][symbol] = df_aligned["close"] / df_aligned["close"].rolling(window=121, min_periods=1).max()
        
        dd_bg = (df_aligned["close"] / df_aligned["close"].rolling(window=121, min_periods=1).max() - 1.0) * 100.0
        metrics["dd_bg"][symbol] = dd_bg.rolling(window=121, min_periods=1).min().abs()
        
        vol_first_bg = df_aligned[bg_vol_col].rolling(window=60).mean().shift(60)
        vol_second_bg = df_aligned[bg_vol_col].rolling(window=60).mean()
        metrics["vol_bg"][symbol] = ((vol_second_bg / vol_first_bg.replace(0.0, np.nan) - 1.0) * 100.0) * price_dir_bg
        
        metrics["oi_bg"][symbol] = ((df_aligned["openInterest"] / df_aligned["openInterest"].shift(120).replace(0.0, np.nan) - 1.0) * 100.0) * price_dir_bg
        metrics["funding_bg"][symbol] = df_aligned["fundingRate"].rolling(window=121, min_periods=1).mean().abs()
        metrics["cb_premium_bg"][symbol] = df_aligned["coinbase_premium"]
        
        price_dir_bb = np.sign(df_aligned["bybit_close"] - df_aligned["bybit_close"].shift(120)).fillna(0.0)
        
        metrics["perf_bb"][symbol] = df_aligned["bybit_close"] / df_aligned["bybit_close"].shift(120)
        
        alt_btc_bb = df_aligned["bybit_close"] / btc_close_bb
        metrics["perf_btc_bb"][symbol] = alt_btc_bb / alt_btc_bb.shift(120)

        metrics["momentum_bb"][symbol] = df_aligned["bybit_close"] / df_aligned["bybit_close"].rolling(window=121, min_periods=1).max()
        
        dd_bb = (df_aligned["bybit_close"] / df_aligned["bybit_close"].rolling(window=121, min_periods=1).max() - 1.0) * 100.0
        metrics["dd_bb"][symbol] = dd_bb.rolling(window=121, min_periods=1).min().abs()
        
        vol_first_bb = df_aligned["bybit_volume"].rolling(window=60).mean().shift(60)
        vol_second_bb = df_aligned["bybit_volume"].rolling(window=60).mean()
        metrics["vol_bb"][symbol] = ((vol_second_bb / vol_first_bb.replace(0.0, np.nan) - 1.0) * 100.0) * price_dir_bb
        
        metrics["oi_bb"][symbol] = ((df_aligned["bybit_openInterest"] / df_aligned["bybit_openInterest"].shift(120).replace(0.0, np.nan) - 1.0) * 100.0) * price_dir_bb
        metrics["funding_bb"][symbol] = df_aligned["bybit_fundingRate"].rolling(window=121, min_periods=1).mean().abs()
        metrics["cb_premium_bb"][symbol] = df_aligned["coinbase_premium"]

    dfs_metrics = {}
    for name, sym_dict in metrics.items():
        dfs_metrics[name] = pd.DataFrame(sym_dict, index=all_timestamps)

    def get_zscore_df(df_metric, invert=False):
        mean = df_metric.mean(axis=1)
        std = df_metric.std(axis=1, ddof=0).replace(0.0, np.nan)
        z = df_metric.sub(mean, axis=0).div(std, axis=0)
        z = z.fillna(0.0)
        return -z if invert else z

    z_perf_bg = get_zscore_df(dfs_metrics["perf_bg"])
    z_perf_btc_bg = get_zscore_df(dfs_metrics["perf_btc_bg"])
    z_momentum_bg = get_zscore_df(dfs_metrics["momentum_bg"])
    z_dd_bg = get_zscore_df(dfs_metrics["dd_bg"], invert=True)
    z_vol_bg = get_zscore_df(dfs_metrics["vol_bg"])
    z_oi_bg = get_zscore_df(dfs_metrics["oi_bg"])
    z_funding_bg = get_zscore_df(dfs_metrics["funding_bg"], invert=True)
    z_cb_premium_bg = get_zscore_df(dfs_metrics["cb_premium_bg"])

    z_perf_bb = get_zscore_df(dfs_metrics["perf_bb"])
    z_perf_btc_bb = get_zscore_df(dfs_metrics["perf_btc_bb"])
    z_momentum_bb = get_zscore_df(dfs_metrics["momentum_bb"])
    z_dd_bb = get_zscore_df(dfs_metrics["dd_bb"], invert=True)
    z_vol_bb = get_zscore_df(dfs_metrics["vol_bb"])
    z_oi_bb = get_zscore_df(dfs_metrics["oi_bb"])
    z_funding_bb = get_zscore_df(dfs_metrics["funding_bb"], invert=True)
    z_cb_premium_bb = get_zscore_df(dfs_metrics["cb_premium_bb"])

    df_score_bg = (
        1.0 * z_perf_bg
        + 0.8 * z_perf_btc_bg
        + 0.5 * z_momentum_bg
        + 0.5 * z_dd_bg
        + 0.5 * z_vol_bg
        + 0.5 * z_oi_bg
        + 0.3 * z_funding_bg
        + 0.5 * z_cb_premium_bg
    )
    df_score_bb = (
        1.0 * z_perf_bb
        + 0.8 * z_perf_btc_bb
        + 0.5 * z_momentum_bb
        + 0.5 * z_dd_bb
        + 0.5 * z_vol_bb
        + 0.5 * z_oi_bb
        + 0.3 * z_funding_bb
        + 0.5 * z_cb_premium_bb
    )
    df_score = (df_score_bg + df_score_bb) / 2.0

    for symbol, path, rows in valid_results:
        df = symbol_dfs[symbol]
        df["score_bitget"] = df["timestamp"].map(df_score_bg[symbol])
        df["score_bybit"] = df["timestamp"].map(df_score_bb[symbol])
        df["score"] = df["timestamp"].map(df_score[symbol])
        df.to_csv(path, index=False)
        log(f"Added rolling scores to: {path}")


def score_and_plot_symbols(
    results: Sequence[Tuple[str, Path, int]],
    *,
    out_dir: Path,
    top_n: int = 20,
    file_prefix: str = "",
    title_prefix: str = "",
) -> Tuple[
    pd.DataFrame,
    Optional[Path],
    Optional[Path],
    pd.DataFrame,
    Optional[Path],
    Optional[Path],
]:
    # Locate the BTCUSDT CSV in results, read it as btc_df
    btc_df = None
    for symbol, csv_path, rows in results:
        if symbol == "BTCUSDT" and rows > 0:
            try:
                btc_df = pd.read_csv(csv_path, parse_dates=["timestamp"])
                log(f"Loaded BTCUSDT reference data from {csv_path}")
            except Exception as e:
                log(f"Failed to load BTCUSDT reference data: {e}")
            break

    # 1. 地合い判定フェーズ (仮の5日ルックバックで計算)
    temp_metrics = []
    for symbol, csv_path, rows in results:
        if rows <= 0:
            continue
        m = compute_symbol_metrics(symbol, csv_path, lookback_days=5, btc_df=btc_df)
        if m is not None:
            temp_metrics.append(m)

    market_state = "long_only"
    if temp_metrics:
        temp_df = pd.DataFrame(temp_metrics)
        perf_bg = temp_df["norm_perf_5d_bg"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        perf_bb = temp_df["norm_perf_5d_bb"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        mean_norm = float(((perf_bg + perf_bb) / 2.0).mean())
        
        cb_prem_bg = temp_df["coinbase_premium_bg"].astype(float).fillna(0.0) if "coinbase_premium_bg" in temp_df.columns else pd.Series(0.0)
        cb_prem_bb = temp_df["coinbase_premium_bb"].astype(float).fillna(0.0) if "coinbase_premium_bb" in temp_df.columns else pd.Series(0.0)
        mean_cb_premium = float(((cb_prem_bg + cb_prem_bb) / 2.0).mean())
        
        market_state = determine_market_state(mean_norm, mean_cb_premium)
        log(f"[market_state] Temp 5d mean_norm = {mean_norm:.4f}, mean_cb_premium = {mean_cb_premium:.4f} -> market_state = {market_state}")
    else:
        log("[market_state] No temp metrics available for 5d -> market_state = long_only (default)")

    # 2. 本番計算フェーズ
    window_days = 30
    log(f"[scoring] Selected window_days = {window_days} based on market_state = {market_state}")

    metrics_list: List[Dict[str, float]] = []
    for symbol, csv_path, rows in results:
        if rows <= 0:
            continue
        metrics = compute_symbol_metrics(symbol, csv_path, lookback_days=window_days, btc_df=btc_df)
        if metrics is not None:
            metrics_list.append(metrics)

    if not metrics_list:
        log("No metrics available for scoring.")
        return pd.DataFrame(), None, None, pd.DataFrame(), None, None

    df = pd.DataFrame(metrics_list)

    # --------- Zスコア計算 ---------

    def add_zscore(df: pd.DataFrame, src_col: str, dst_col: str, invert: bool = False) -> None:
        """単純なZスコア（平均0, 分散1）。stdが0/NaNなら0固定。"""
        if src_col not in df.columns:
            df[dst_col] = 0.0
            return
        s = df[src_col].astype(float)
        mean = float(s.mean())
        std = float(s.std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            df[dst_col] = 0.0
        else:
            val = (s - mean) / std
            df[dst_col] = -val if invert else val

    def compute_exchange_score(prefix: str, score_col_name: str) -> None:
        add_zscore(df, f"norm_perf_5d{prefix}", f"z_perf_5d{prefix}")
        add_zscore(df, f"norm_perf_5d_btc{prefix}", f"z_perf_5d_btc{prefix}")
        add_zscore(df, f"momentum_dir_5d{prefix}", f"z_momentum_5d{prefix}")
        
        # DDの計算
        if f"max_drawdown_5d{prefix}" in df.columns:
            df[f"dd_abs{prefix}"] = df[f"max_drawdown_5d{prefix}"].astype(float).abs()
            if market_state == "long_only":
                # 安定銘柄狙い：ドローダウンが小さいほど高評価 (バックテストと同一)
                add_zscore(df, f"dd_abs{prefix}", f"z_dd_5d{prefix}", invert=True)
            else:
                # 通常：ドローダウンが小さいほど高評価（ショートでは使わないが一応計算）
                add_zscore(df, f"dd_abs{prefix}", f"z_dd_5d{prefix}", invert=True)
        else:
            df[f"z_dd_5d{prefix}"] = 0.0

        add_zscore(df, f"volume_change_5d{prefix}", f"z_vol_5d{prefix}")
        add_zscore(df, f"oi_change_5d{prefix}", f"z_oi_5d{prefix}")
        
        # Funding -> 0に近いほどプラス (invert=True)
        if f"funding_avg_5d{prefix}" in df.columns:
            df[f"fund_abs{prefix}"] = df[f"funding_avg_5d{prefix}"].astype(float).abs()
            add_zscore(df, f"fund_abs{prefix}", f"z_funding_5d{prefix}", invert=True)
        else:
            df[f"z_funding_5d{prefix}"] = 0.0

        # コインベースプレミアムZスコア
        add_zscore(df, f"coinbase_premium{prefix}", f"z_cb_premium{prefix}")

        for col in [f"z_perf_5d{prefix}", f"z_perf_5d_btc{prefix}", f"z_momentum_5d{prefix}", f"z_dd_5d{prefix}",
                    f"z_vol_5d{prefix}", f"z_oi_5d{prefix}", f"z_funding_5d{prefix}", f"z_cb_premium{prefix}"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 地合いに応じたスコアリング式の適用
        if market_state == "long_only":
            # 強気リバウンド戦略 (15日)
            df[score_col_name] = (
                1.0 * df[f"z_perf_5d{prefix}"]
                + 0.8 * df[f"z_perf_5d_btc{prefix}"]
                + 0.5 * df[f"z_momentum_5d{prefix}"]
                + 2.0 * df[f"z_dd_5d{prefix}"]
                + 0.5 * df[f"z_vol_5d{prefix}"]
                + 0.5 * df[f"z_oi_5d{prefix}"]
                + 0.5 * df[f"z_cb_premium{prefix}"]
            )
        else:
            # 弱気ブレイクアウト追撃売り戦略 (20日)
            df[score_col_name] = (
                1.5 * df[f"z_perf_5d{prefix}"]
                + 0.8 * df[f"z_perf_5d_btc{prefix}"]
                + 0.5 * df[f"z_momentum_5d{prefix}"]
                - 1.0 * df[f"z_vol_5d{prefix}"]
                + 0.5 * df[f"z_cb_premium{prefix}"]
            )
        df[score_col_name] = df[score_col_name].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Compute for both Bitget and Bybit
    compute_exchange_score("_bg", "score_bitget")
    compute_exchange_score("_bb", "score_bybit")

    # --------- 最終スコア（合算/平均） ---------
    df["score"] = (df["score_bitget"] + df["score_bybit"]) / 2.0

    # Determine if symbol is oversold (本番期間の performance <= 0.8)
    is_oversold_bg = (df["norm_perf_5d_bg"] <= 0.8) if "norm_perf_5d_bg" in df.columns else pd.Series(False, index=df.index)
    is_oversold_bb = (df["norm_perf_5d_bb"] <= 0.8) if "norm_perf_5d_bb" in df.columns else pd.Series(False, index=df.index)
    is_oversold = is_oversold_bg | is_oversold_bb

    # Determine if symbol is low liquidity (daily turnover < 5M USDT on EITHER exchange)
    turnover_bg = df["bg_daily_turnover"].fillna(0.0) if "bg_daily_turnover" in df.columns else pd.Series(0.0, index=df.index)
    turnover_bb = df["bb_daily_turnover"].fillna(0.0) if "bb_daily_turnover" in df.columns else pd.Series(0.0, index=df.index)
    is_low_liq = (turnover_bg < 5000000.0) | (turnover_bb < 5000000.0)

    # Ban shorts if oversold or low liquidity
    df["is_oversold"] = is_oversold.astype(int)
    df["is_low_liq"] = is_low_liq.astype(int)
    df["ban_short"] = (is_oversold | is_low_liq).astype(int)

    log(
        f"[scoring] Market state: {market_state}"
    )

    prefix = file_prefix if (not file_prefix) or file_prefix.endswith("_") else f"{file_prefix}_"

    # それぞれのスコアでチャートとCSVを出力するヘルパー
    def export_score(score_col: str, title_str: str, file_suffix: str):
        df_rank = df.sort_values(by=[score_col], ascending=[False]).reset_index(drop=True)
        top_df_sub = df_rank.head(top_n).copy()
        
        png_path = out_dir / f"{prefix}symbol_selection_{file_suffix}.png"
        csv_path = out_dir / f"{prefix}symbol_selection_{file_suffix}.csv"
        
        if not top_df_sub.empty:
            top_df_sub.to_csv(csv_path, index=False)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(top_df_sub["symbol"], top_df_sub[score_col])
            
            title = title_str
            if title_prefix:
                title = f"[{title_prefix}] {title}"
            ax.set_title(title)
            ax.set_ylabel("Score")
            ax.grid(True, axis="y", alpha=0.2)
            plt.xticks(rotation=45, ha="right")
            fig.tight_layout()
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            log(f"Saved score chart -> {png_path}")
            log(f"Saved score table -> {csv_path}")

        return top_df_sub, csv_path, png_path

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bybit のスコア出力
    df_bybit, csv_bybit, png_bybit = export_score("score_bybit", "Altcoin Selection Score(Bybit-Only)", "scores_bybit")
    
    # 2. Bitget のスコア出力
    df_bitget, csv_bitget, png_bitget = export_score("score_bitget", "Altcoin Selection Score(Bitget-Only)", "scores_bitget")

    # 3. Dual-Exchange (合算/平均) のスコア出力 -> 今後のメインランキングとして扱う
    top_df_all, scores_all_csv, scores_all_png = export_score(
        "score", "Altcoin Selection Score(Dual-Exchange)", "scores_all"
    )

    top_df = top_df_all
    scores_csv = scores_all_csv
    scores_png = scores_all_png

    print("\n=== Selection Ranking (Dual-Exchange Multi-Window-Based) ===")
    display_cols = [c for c in [
        "symbol", "score", "score_bitget", "score_bybit",
        "norm_perf_5d_bg", "norm_perf_5d_bb",
        "momentum_dir_5d_bg", "max_drawdown_5d_bg",
        "z_perf_5d_bg", "z_perf_5d_bb"
    ] if c in top_df.columns]
    
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(top_df[display_cols].round(4))

    # ---- Market State 保存 ----
    # すでに判定済みの market_state をそのまま使用・保存します。
    log(f"Final Market State selected: {market_state}")

    # 状態を保存 (Data直下)
    state_file = Path(__file__).resolve().parent / "Data" / "market_state.json"
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(
                {
                    "market_state": market_state,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        log(f"Saved market state to {state_file}")
    except Exception as e:
        log(f"Failed to save market state: {e}")

    # discord 通知用に配列にして返す
    return top_df, [csv_bybit, csv_bitget, scores_all_csv], [png_bybit, png_bitget, scores_all_png], top_df_all, scores_all_csv, scores_all_png

def parse_symbol_list(raw: str) -> List[str]:
    return [token.strip().upper() for token in raw.split(",") if token.strip()]


def dedup_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for sym in items:
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def parse_dt(value: Optional[str], fallback: datetime) -> datetime:
    if value:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=JST)
    else:
        dt = fallback
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.replace(minute=0, second=0, microsecond=0)


def default_out_dir() -> Path:
    return Path(__file__).resolve().parent / "Data"


def clean_output_dir(path: Path) -> None:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor):
        log(f"Skip cleanup for unsafe path: {resolved}")
        return
    if not resolved.exists():
        return
    log(f"Cleaning output directory before saving new files: {resolved}")
    for entry in resolved.iterdir():
        try:
            if entry.is_file() or entry.is_symlink():
                if entry.name == "historical_all_symbols_merged.csv":
                    continue
                entry.unlink()
            elif entry.is_dir():
                # raw_cache 自体を削除してしまうのを防ぐため、raw_cache の中身をクリーンアップするか、
                # または raw_cache フォルダ自体を削除対象から除外する
                if entry.name == "raw_cache":
                    continue
                shutil.rmtree(entry)
        except OSError as exc:
            log(f"Cleanup skipped for {entry}: {exc}")


async def generate_historical_scores(limit_symbols: int = 100, days: int = 30) -> None:
    log("=== Generating 30-Day Historical Selection Scores ===")
    data_dir = default_out_dir()
    cache_dir = data_dir / "raw_cache"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch Candidates (top 100 USDT symbols today by turnover)
    try:
        top_100_symbols = [sym for sym, _ in fetch_bybit_top_usdt_symbols(limit_symbols)]
    except Exception as e:
        log(f"Failed to fetch Bybit top symbols for history: {e}")
        return

    if not top_100_symbols:
        log("No symbols fetched for history. Skipping.")
        return

    # 2. Determine trade days (last 30 days daily at 11:00 JST)
    now_jst = datetime.now(JST)
    anchor_today_1100 = now_jst.replace(hour=11, minute=0, second=0, microsecond=0)
    anchor = anchor_today_1100 if now_jst >= anchor_today_1100 else anchor_today_1100 - timedelta(days=1)
    
    trade_days = [anchor - timedelta(days=i) for i in range(days)]
    trade_days.sort()  # Chronological order
    
    start_download = trade_days[0] - timedelta(days=5, hours=2)  # Extra padding for rolling calculation
    end_download = now_jst
    
    start_ms = to_ms(start_download)
    end_ms = to_ms(end_download)
    
    log(f"Historical calculation window: {trade_days[0].strftime('%Y-%m-%d %H:%M JST')} to {trade_days[-1].strftime('%Y-%m-%d %H:%M JST')}")
    
    # 3. Download data for top symbols with semaphore
    sem = asyncio.Semaphore(5)
    symbol_dfs = {}
    
    async with pybotters.Client() as client:
        # Helper to align single symbol
        async def process_symbol(symbol: str):
            cache_path = cache_dir / f"{symbol}_bybit_raw.csv"
            # Try cache first
            if cache_path.exists():
                try:
                    df = pd.read_csv(cache_path)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    min_ts = df["timestamp"].min()
                    max_ts = df["timestamp"].max()
                    req_start = start_download
                    req_end = end_download
                    if min_ts <= req_start and max_ts >= req_end - timedelta(hours=2):
                        log(f"Loaded {symbol} from cache ({len(df)} rows)")
                        return symbol, df
                except Exception as e:
                    log(f"Failed to read cache for {symbol}: {e}. Re-downloading...")
            
            log(f"Downloading historical Bybit data for {symbol}...")
            try:
                async with sem:
                    ohlcv = await fetch_bybit_ohlcv_1h(symbol, start_ms, end_ms)
                    oi = await fetch_bybit_open_interest_1h(symbol, start_ms, end_ms)
                    fund = await fetch_bybit_funding_history(symbol, start_ms, end_ms)
            except Exception as e:
                log(f"Failed to fetch API data for {symbol}: {e}")
                return symbol, pd.DataFrame()
            
            if not ohlcv:
                return symbol, pd.DataFrame()
                
            df = pd.DataFrame.from_records(ohlcv).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            if oi:
                df_oi = pd.DataFrame.from_records(oi).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                df_oi["timestamp"] = pd.to_datetime(df_oi["timestamp"])
                df_oi = df_oi.set_index("timestamp").resample("1h").last().ffill().reset_index()
                df = pd.merge(df, df_oi, on="timestamp", how="left")
                df["bybit_openInterest"] = df["bybit_openInterest"].ffill().bfill()
            else:
                df["bybit_openInterest"] = 0.0
                
            if fund:
                df_fund = pd.DataFrame.from_records(fund).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                df_fund["timestamp"] = pd.to_datetime(df_fund["timestamp"])
                df_fund = df_fund.set_index("timestamp").resample("1h").last().ffill().reset_index()
                df = pd.merge(df, df_fund, on="timestamp", how="left")
                df["bybit_fundingRate"] = df["bybit_fundingRate"].ffill().bfill()
            else:
                df["bybit_fundingRate"] = 0.0
                
            for col in ["bybit_open", "bybit_high", "bybit_low", "bybit_close", "bybit_volume"]:
                if col in df.columns:
                    df[col] = df[col].ffill().bfill()
            
            # Save to cache
            df.to_csv(cache_path, index=False)
            log(f"Saved {symbol} data to cache ({len(df)} rows)")
            return symbol, df

        tasks = [process_symbol(sym) for sym in top_100_symbols]
        results = await asyncio.gather(*tasks)
        for sym, df_sym in results:
            if df_sym is not None and not df_sym.empty:
                symbol_dfs[sym] = df_sym

    if not symbol_dfs:
        log("No historical symbol data available. Skipping.")
        return

    # 4. Daily selection and scoring loop
    all_daily_records = []
    
    def get_zscore(series: pd.Series, invert: bool = False) -> pd.Series:
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            return pd.Series(0.0, index=series.index)
        val = (series - mean) / std
        return -val if invert else val

    for day_dt in trade_days:
        day_str = day_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # A. Find top 20 symbols based on 24h turnover (estimated as bybit_volume * bybit_close) before day_dt
        daily_turnovers = []
        for symbol, df in symbol_dfs.items():
            # 24h window before day_dt
            window_24h = df[(df["timestamp"] > day_dt - timedelta(days=1)) & (df["timestamp"] <= day_dt)]
            if not window_24h.empty:
                turnover_sum = (window_24h["bybit_volume"] * window_24h["bybit_close"]).sum()
                daily_turnovers.append((turnover_sum, symbol))
            else:
                daily_turnovers.append((0.0, symbol))
                
        daily_turnovers.sort(key=lambda x: x[0], reverse=True)
        top_20_daily = [sym for _, sym in daily_turnovers[:20]]
        
        # B. Determine market state using temporary 5d metrics
        temp_metrics = []
        for symbol in top_20_daily:
            df = symbol_dfs[symbol]
            df_5d = df[(df["timestamp"] > day_dt - timedelta(days=5)) & (df["timestamp"] <= day_dt)].copy()
            if len(df_5d) < 24 * 4:
                continue
            close_now = float(df_5d["bybit_close"].iloc[-1])
            close_start = float(df_5d["bybit_close"].iloc[0])
            norm_perf_5d = close_now / close_start if close_start > 0 else 1.0
            cb_premium_val = float(df_5d["coinbase_premium"].iloc[-1]) if "coinbase_premium" in df_5d.columns and not df_5d["coinbase_premium"].empty else 0.0
            
            temp_metrics.append({
                "symbol": symbol,
                "norm_perf_5d": norm_perf_5d,
                "coinbase_premium": cb_premium_val
            })
            
        market_state = "long_only"
        if temp_metrics:
            df_temp = pd.DataFrame(temp_metrics)
            mean_norm = float(df_temp["norm_perf_5d"].mean())
            mean_cb_premium = float(df_temp["coinbase_premium"].mean())
            market_state = determine_market_state(mean_norm, mean_cb_premium)
            
        window_days = 30
        
        # BTCデータの取得
        btc_df_window = None
        if "BTCUSDT" in symbol_dfs:
            df_btc = symbol_dfs["BTCUSDT"]
            btc_df_window = df_btc[(df_btc["timestamp"] > day_dt - timedelta(days=window_days)) & (df_btc["timestamp"] <= day_dt)].copy()

        # C. Compute metrics for the selected top 20 symbols
        metrics_list = []
        for symbol in top_20_daily:
            df = symbol_dfs[symbol]
            df_w = df[(df["timestamp"] > day_dt - timedelta(days=window_days)) & (df["timestamp"] <= day_dt)].copy()
            if len(df_w) < 24 * (window_days - 1):
                continue
                
            close_series = df_w["bybit_close"].astype(float).dropna()
            if close_series.empty or len(close_series) < 2:
                continue
                
            p5_first = float(close_series.iloc[0])
            p5_last = float(close_series.iloc[-1])
            price_dir = float(np.sign(p5_last - p5_first))
            
            norm_perf_5d = p5_last / p5_first if p5_first > 0 else 1.0
            
            # BTC相対
            norm_perf_5d_btc = 1.0
            if btc_df_window is not None and not btc_df_window.empty:
                df_sym_btc = pd.merge(
                    df_w[['timestamp', 'bybit_close']],
                    btc_df_window[['timestamp', 'bybit_close']].rename(columns={'bybit_close': 'btc_close'}),
                    on='timestamp',
                    how='left'
                )
                if 'btc_close' in df_sym_btc.columns:
                    alt_btc = df_sym_btc['bybit_close'] / df_sym_btc['btc_close']
                    alt_btc = alt_btc.dropna()
                    if len(alt_btc) >= 2:
                        norm_perf_5d_btc = float(alt_btc.iloc[-1] / alt_btc.iloc[0]) if alt_btc.iloc[0] > 0 else 1.0
            
            # モメンタム: 前時間比で上昇した割合 (バックテストと同一方式)
            diffs = close_series.diff().dropna()
            momentum_dir_5d = float((diffs > 0).mean()) if len(diffs) > 0 else 0.5
            
            rolling_max_5 = close_series.cummax()
            drawdowns = (close_series / rolling_max_5 - 1.0) * 100.0
            max_drawdown_5d = float(drawdowns.min()) if not drawdowns.empty else 0.0
            
            vol_series = df_w["bybit_volume"].astype(float).replace(0.0, np.nan).dropna()
            volume_change_5d = 0.0
            if len(vol_series) >= 4:
                mid = len(vol_series) // 2
                vol_first_half = float(vol_series.iloc[:mid].mean())
                vol_second_half = float(vol_series.iloc[mid:].mean())
                if vol_first_half > 0:
                    volume_change_5d = (vol_second_half / vol_first_half - 1.0) * 100.0 * price_dir
                    
            oi_series = df_w["bybit_openInterest"].astype(float).replace(0.0, np.nan).dropna()
            oi_change_5d = 0.0
            if len(oi_series) >= 2:
                oi_first = float(oi_series.iloc[0])
                oi_last = float(oi_series.iloc[-1])
                if oi_first > 0:
                    oi_change_5d = (oi_last / oi_first - 1.0) * 100.0 * price_dir
                    
            fund_series = df_w["bybit_fundingRate"].astype(float).replace(0.0, np.nan).dropna()
            funding_avg_5d = float(fund_series.mean()) if not fund_series.empty else 0.0
            
            cb_premium_val = float(df_w["coinbase_premium"].iloc[-1]) if "coinbase_premium" in df_w.columns and not df_w["coinbase_premium"].empty else 0.0
            
            # Low liquidity check (daily turnover < 5M USDT)
            bb_daily_turnover = float((df_w["bybit_volume"] * df_w["bybit_close"]).dropna().mean() * 24)
            is_low_liq = 1 if bb_daily_turnover < 5000000.0 else 0
            is_oversold = 1 if norm_perf_5d <= 0.8 else 0
            ban_short = 1 if (is_low_liq or is_oversold) else 0
            
            metrics_list.append({
                "symbol": symbol,
                "norm_perf_5d_bb": norm_perf_5d,
                "norm_perf_5d_btc_bb": norm_perf_5d_btc,
                "momentum_dir_5d_bb": momentum_dir_5d,
                "max_drawdown_5d_bb": max_drawdown_5d,
                "volume_change_5d_bb": volume_change_5d,
                "oi_change_5d_bb": oi_change_5d,
                "funding_avg_5d_bb": funding_avg_5d,
                "coinbase_premium_bb": cb_premium_val,
                "bb_daily_turnover": bb_daily_turnover,
                "is_low_liq": is_low_liq,
                "is_oversold": is_oversold,
                "ban_short": ban_short
            })

        if not metrics_list:
            continue
            
        df_daily = pd.DataFrame(metrics_list)
        
        # Z-scores
        df_daily["z_perf_5d_bb"] = get_zscore(df_daily["norm_perf_5d_bb"])
        df_daily["z_perf_5d_btc_bb"] = get_zscore(df_daily["norm_perf_5d_btc_bb"])
        df_daily["z_momentum_5d_bb"] = get_zscore(df_daily["momentum_dir_5d_bb"])
        df_daily["dd_abs_bb"] = df_daily["max_drawdown_5d_bb"].abs()
        df_daily["z_dd_5d_bb"] = get_zscore(df_daily["dd_abs_bb"], invert=True)
        df_daily["z_vol_5d_bb"] = get_zscore(df_daily["volume_change_5d_bb"])
        df_daily["z_oi_5d_bb"] = get_zscore(df_daily["oi_change_5d_bb"])
        df_daily["fund_abs_bb"] = df_daily["funding_avg_5d_bb"].abs()
        df_daily["z_funding_5d_bb"] = get_zscore(df_daily["fund_abs_bb"], invert=True)
        df_daily["z_cb_premium_bb"] = get_zscore(df_daily["coinbase_premium_bb"])
        
        z_cols_bb = ["z_perf_5d_bb", "z_perf_5d_btc_bb", "z_momentum_5d_bb", "z_dd_5d_bb", "z_vol_5d_bb", "z_oi_5d_bb", "z_funding_5d_bb", "z_cb_premium_bb"]
        for col in z_cols_bb:
            df_daily[col] = df_daily[col].fillna(0.0)
            
        # 地合いに応じたスコア算出
        if market_state == "long_only":
            df_daily["score_bybit"] = (
                1.0 * df_daily["z_perf_5d_bb"]
                + 0.8 * df_daily["z_perf_5d_btc_bb"]
                + 0.5 * df_daily["z_momentum_5d_bb"]
                + 2.0 * df_daily["z_dd_5d_bb"]
                + 0.5 * df_daily["z_vol_5d_bb"]
                + 0.5 * df_daily["z_oi_5d_bb"]
                + 0.5 * df_daily["z_cb_premium_bb"]
            )
        else:
            df_daily["score_bybit"] = (
                1.5 * df_daily["z_perf_5d_bb"]
                + 0.8 * df_daily["z_perf_5d_btc_bb"]
                + 0.5 * df_daily["z_momentum_5d_bb"]
                - 1.0 * df_daily["z_vol_5d_bb"]
                + 0.5 * df_daily["z_cb_premium_bb"]
            )
        
        # Bitget互換フィールドを設定
        df_daily["norm_perf_5d_bg"] = df_daily["norm_perf_5d_bb"]
        df_daily["norm_perf_5d_btc_bg"] = df_daily["norm_perf_5d_btc_bb"]
        df_daily["momentum_dir_5d_bg"] = df_daily["momentum_dir_5d_bb"]
        df_daily["max_drawdown_5d_bg"] = df_daily["max_drawdown_5d_bb"]
        df_daily["volume_change_5d_bg"] = df_daily["volume_change_5d_bb"]
        df_daily["oi_change_5d_bg"] = df_daily["oi_change_5d_bb"]
        df_daily["funding_avg_5d_bg"] = df_daily["funding_avg_5d_bb"]
        
        df_daily["z_perf_5d_bg"] = df_daily["z_perf_5d_bb"]
        df_daily["z_perf_5d_btc_bg"] = df_daily["z_perf_5d_btc_bb"]
        df_daily["z_momentum_5d_bg"] = df_daily["z_momentum_5d_bb"]
        df_daily["dd_abs_bg"] = df_daily["dd_abs_bb"]
        df_daily["z_dd_5d_bg"] = df_daily["z_dd_5d_bb"]
        df_daily["z_vol_5d_bg"] = df_daily["z_vol_5d_bb"]
        df_daily["z_oi_5d_bg"] = df_daily["z_oi_5d_bb"]
        df_daily["fund_abs_bg"] = df_daily["fund_abs_bb"]
        df_daily["z_funding_5d_bg"] = df_daily["z_funding_5d_bb"]
        
        df_daily["score_bitget"] = df_daily["score_bybit"]
        df_daily["score"] = df_daily["score_bybit"]
        df_daily["date"] = day_str
        df_daily["market_state"] = market_state
        
        df_daily = df_daily.sort_values(by="score", ascending=False).reset_index(drop=True)
        all_daily_records.append(df_daily)

    if not all_daily_records:
        log("No historical scores computed.")
        return

    df_master = pd.concat(all_daily_records, ignore_index=True)
    cols = ["date", "symbol", "score", "score_bitget", "score_bybit", "ban_short", "is_oversold", "is_low_liq"]
    other_cols = [c for c in df_master.columns if c not in cols]
    df_master = df_master[cols + other_cols]
    
    output_path = data_dir / "historical_selection_scores.csv"
    df_master.to_csv(output_path, index=False)
    log(f"SUCCESS: Created historical selection scores -> {output_path}")


def main() -> None:
    default_end_jst = datetime.now(JST).replace(minute=0, second=0, microsecond=0)
    default_end_str = default_end_jst.isoformat()

    parser = argparse.ArgumentParser(
        description="Rebuild merged Bitget hourly datasets for multiple symbols."
    )
    parser.add_argument("--symbol", dest="symbols", help="Comma separated symbols to fetch (overrides auto-selection)")
    parser.add_argument("--start", help=f"Start datetime (ISO, default {DEFAULT_START_STR})")
    parser.add_argument("--end", help=f"End datetime (ISO, default {default_end_str})")
    parser.add_argument("--outdir", help="Directory to store merged CSV files (default: Data/ next to this script)")
    parser.add_argument("--product-type", default=DEFAULT_PRODUCT_TYPE, help="Bitget productType")
    parser.add_argument("--granularity", default="1H", help="Bitget candle granularity (default: 1H)")
    parser.add_argument("--limit", type=int, default=40, help="Number of top-volume symbols to auto-select")
    parser.add_argument("--quote", default="USDT", help="Quote currency filter for auto-selection")
    parser.add_argument("--demo-mode", default="live", help="Filter symbols compatible with demo trading (live, demo, paper)")
    parser.set_defaults(include_btc=True)
    parser.add_argument("--include-btc", dest="include_btc", action="store_true", help="Include BTCUSDT when auto-selecting")
    parser.add_argument("--exclude-btc", dest="include_btc", action="store_false", help="Exclude BTCUSDT when auto-selecting")

    args = parser.parse_args()

    product_type = normalize_product_type(args.product_type)
    granularity = str(args.granularity).upper()
    if granularity not in GRANULARITY_MS:
        parser.error(f"Unsupported granularity: {granularity} (supported: {', '.join(GRANULARITY_MS)})")

    fallback_start = DEFAULT_START_JST
    fallback_end = default_end_jst
    start_utc = parse_dt(args.start, fallback_start)
    end_utc = parse_dt(args.end, fallback_end)
    if end_utc <= start_utc:
        parser.error("End datetime must be greater than start datetime")

    out_dir = Path(args.outdir) if args.outdir else default_out_dir()
    clean_output_dir(out_dir)

    # 1. 2年分ヒストリカルデータ更新判定＆ダウンロード実行
    master_csv_path = out_dir / "historical_all_symbols_merged.csv"
    need_download = False

    if not master_csv_path.exists():
        log("historical_all_symbols_merged.csv not found. Triggering initial download...")
        need_download = True
    else:
        mtime = datetime.fromtimestamp(master_csv_path.stat().st_mtime, tz=JST)
        now = datetime.now(JST)
        days_since_update = (now - mtime).days
        
        # 最後に更新されてから7日以上経過しているか、または今日が日曜日でかつ今日まだ更新されていない場合
        if days_since_update >= 7:
            log(f"historical_all_symbols_merged.csv is old ({days_since_update} days old). Triggering update...")
            need_download = True
        elif now.weekday() == 6 and mtime.date() != now.date():
            log("Today is Sunday and data was not updated today. Triggering weekly update...")
            need_download = True

    if need_download:
        try:
            download_script = Path(__file__).resolve().parent / "download_historical_candles.py"
            log(f"Running download script: {download_script}")
            subprocess.run([sys.executable, str(download_script)], check=True)
            log("Download script completed successfully.")
        except Exception as e:
            log(f"Error running download script: {e}")

    # 2. パラメータ最適化＆バックテスト評価の自動実行
    try:
        backtest_script = Path(__file__).resolve().parent / "generate_selection_scores.py"
        log(f"Running backtest & optimization script: {backtest_script}")
        subprocess.run([sys.executable, str(backtest_script)], check=True)
        log("Backtest script completed successfully.")
    except Exception as e:
        log(f"Error running backtest script: {e}")

    quote = args.quote.upper()

    if args.symbols:
        manual_symbols = parse_symbol_list(args.symbols)
        bitget_top = [(sym, 0.0) for sym in manual_symbols]
    else:
        # Fetch top symbols directly from Bitget without Bybit overlap check
        bitget_top = fetch_bitget_top_usdt_symbols(
            args.limit,
            product_type=product_type,
            quote=quote,
        )
        
        if args.include_btc and all(sym != "BTCUSDT" for sym, _ in bitget_top):
            bitget_top.insert(0, ("BTCUSDT", bitget_top[0][1] if bitget_top else 0.0))

    # 選定されたシンボルリスト (Bitget優先)
    limit_n = args.limit if args.limit > 0 else len(bitget_top)
    selected_symbols = [sym for sym, _ in bitget_top][:limit_n]
    
    # 重複削除
    symbols = dedup_preserve_order(selected_symbols)
    if not symbols:
        raise RuntimeError("No symbols selected for processing.")

    if args.demo_mode in ("demo", "paper"):
        log("Filtering symbols available for demo trading...")
        try:
             # Bitget v2 contracts endpoint for demo
             base_url = "https://api.bitget.com"
             resp = requests.get(f"{base_url}/api/v2/mix/market/contracts", params={"productType": product_type}, headers={"paptrading": "1"})
             d = resp.json()
             if d.get("code") == "00000":
                 available_demo = {it.get("symbol").upper() for it in d.get("data", [])}
                 original_count = len(symbols)
                 symbols = [s for s in symbols if s in available_demo]
                 log(f"Demo filter applied: {original_count} -> {len(symbols)} symbols remaining.")
             else:
                 log(f"Failed to fetch demo symbols: {d.get('msg')}")
        except Exception as e:
             log(f"Error fetching demo symbols: {e}")

    log(f"Targets for unified fetch: {symbols}")

    ranks = {symbol: idx + 1 for idx, symbol in enumerate(symbols)}

    results_map: Dict[str, Tuple[Path, int]] = {}
    
    results = asyncio.run(
        build_all_symbols(
            symbols,
            start_utc,
            end_utc,
            out_dir=out_dir,
            product_type=product_type,
            granularity=granularity,
            ranks=ranks,
        )
    )

    for symbol, out_path, rows in results:
        results_map[symbol] = (out_path, rows)
        log(f"{symbol}: {rows} rows -> {out_path}")

    # --- Scoring & Analysis ---
    valid_results = [(sym, path, rows) for sym, path, rows in results if rows > 0]
    
    if valid_results:
        try:
            compute_rolling_scores(valid_results)
        except Exception as exc:
            log(f"Computing rolling scores failed: {exc}")

    top_df = pd.DataFrame()
    scores_csv = None
    scores_png = None

    if valid_results:
        try:
            log("=== Scoring Symbols ===")
            (
                top_df,
                scores_csv,
                scores_png,
                top_df_all,
                scores_all_csv,
                scores_all_png,
            ) = score_and_plot_symbols(
                valid_results[:20],
                out_dir=out_dir,
                top_n=20,
                file_prefix="",
                title_prefix="Bitget Top",
            )
        except Exception as exc:
            log(f"Scoring failed: {exc}")

    # --- Notification (Discord) ---
    discord_client: Optional[send_discord] = None
    
    def ensure_discord() -> send_discord:
        nonlocal discord_client
        if discord_client is None:
            discord_client = send_discord()
        return discord_client

    if scores_png or scores_csv:
        client = ensure_discord()
        msg_names = ["Bybit", "Bitget", "総合"]
        if isinstance(scores_png, list):
            for i in range(len(msg_names)):
                client.send_message(f"{msg_names[i]} 銘柄選定スコア ({len(top_df)}銘柄) 計算完了。")
                if scores_png and i < len(scores_png) and scores_png[i]:
                    client.send_file(scores_png[i], "銘柄選定スコアチャート。。。")
                if scores_csv and i < len(scores_csv) and scores_csv[i]:
                    client.send_file(scores_csv[i], "銘柄選定スコア詳細CSV")
        else:
            client.send_message(f"総合 銘柄選定スコア ({len(top_df)}銘柄) 計算完了。")
            if scores_png:
                client.send_file(scores_png, "銘柄選定スコアチャート。。。")
            if scores_csv:
                client.send_file(scores_csv, "銘柄選定スコア詳細CSV")

    if not top_df.empty:
        top_symbols_list = top_df["symbol"].tolist()
        ranked_for_norm = []
        for rank, sym in enumerate(top_symbols_list, start=1):
            if sym in results_map:
                path, rows = results_map[sym]
                ranked_for_norm.append((rank, sym, path))
        
        # Limit to top 20 for charts
        ranked_for_norm = ranked_for_norm[:20]

        normalized_outputs = generate_normalized_datasets(
            ranked_for_norm,
            end_utc,
            out_dir=out_dir,
        )

        normalized_plots = []
        for window_label, csv_path, df_norm in normalized_outputs:
             try:
                plot_path = plot_normalized_dataset(
                    window_label,
                    df_norm,
                    out_dir=out_dir / "Normalized",
                    title_prefix="Bitget Top",
                )
                normalized_plots.append((window_label, csv_path, plot_path))
             except Exception as exc:
                 log(f"Plot failed for {window_label}: {exc}")

        if normalized_plots:
            client = ensure_discord()
            client.send_message("ノーマライズチャート (30d/10d/5d) を送信します。")
            for window_label, csv_path, plot_path in normalized_plots:
                 client.send_file(plot_path, f"{window_label} Normalized Chart")

    if valid_results:
        import zipfile
        client = ensure_discord()
        dfs = []
        for sym, path, rows in valid_results:
            try:
                df_sym = pd.read_csv(path)
                dfs.append(df_sym)
            except Exception as e:
                log(f"Failed to read {path} for concatenation: {e}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            start_tag = start_utc.strftime("%Y%m%d")
            end_tag = end_utc.strftime("%Y%m%d")
            combined_csv_name = f"{start_tag}_{end_tag}_all_symbols_merged.csv"
            combined_csv_path = out_dir / combined_csv_name
            combined_df.to_csv(combined_csv_path, index=False)
            log(f"Saved combined CSV: {len(combined_df)} rows -> {combined_csv_path}")

            combined_zip_name = f"{start_tag}_{end_tag}_all_symbols_merged.zip"
            combined_zip_path = out_dir / combined_zip_name
            try:
                with zipfile.ZipFile(combined_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(combined_csv_path, arcname=combined_csv_name)
                log(f"Created ZIP file: {combined_zip_path}")
            except Exception as e:
                log(f"Failed to create ZIP: {e}")
                combined_zip_path = None

            client.send_message(f"統合データCSV (全{len(valid_results)}銘柄、計{len(combined_df)}行) を送信します。")
            if combined_zip_path and combined_zip_path.exists():
                client.send_file(combined_zip_path, "全銘柄統合データCSV (ZIP圧縮版)")
            else:
                client.send_file(combined_csv_path, "全銘柄統合データCSV")

    log("=== Running historical selection scores generation ===")
    try:
        asyncio.run(generate_historical_scores(100, 30))
    except Exception as exc:
        log(f"Failed to generate historical scores: {exc}")

    log("All tasks completed.")

if __name__ == "__main__":
    main()
