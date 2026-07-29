import os
import sys
import json
import asyncio
import time
import datetime
from datetime import timezone, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pybotters

# ロギングと定数定義
def log(message: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}")

JST = timezone(timedelta(hours=9))

def to_ms(dt: datetime.datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)

def from_ms_jst(ms: int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(JST)

REST_API_URL = {
    "bybit": "https://api.bybit.com",
    "coinbase": "https://api.exchange.coinbase.com"
}

script_dir = Path(__file__).resolve().parent
data_dir = script_dir / "Data"
data_dir.mkdir(parents=True, exist_ok=True)
raw_cache_dir = data_dir / "raw_cache"
raw_cache_dir.mkdir(parents=True, exist_ok=True)

scores_csv_path = data_dir / "historical_selection_scores_1year.csv"
if not scores_csv_path.exists():
    scores_csv_path = data_dir / "historical_selection_scores.csv"

# ダウンロード対象銘柄の決定
if scores_csv_path.exists():
    df_scores = pd.read_csv(scores_csv_path)
    symbols = sorted(df_scores['symbol'].unique())
    log(f"Loaded {len(symbols)} symbols from {scores_csv_path.name}")
else:
    # デフォルト銘柄
    symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", 
        "SUIUSDT", "WLDUSDT", "NEARUSDT", "LTCUSDT", "BCHUSDT", "AVAXUSDT", 
        "LINKUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "RENDERUSDT", "TAOUSDT"
    ]
    log(f"Scores CSV not found, using default {len(symbols)} symbols.")

# --- Bybit 非同期データフェッチ関数 ---

async def fetch_bybit_ohlcv_1h(symbol: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    url = REST_API_URL["bybit"] + "/v5/market/kline"
    limit = 1000
    step_ms = 60 * 60 * 1000
    rows: List[Dict[str, Any]] = []
    async with pybotters.Client() as client:
        cur = start_ms
        while cur < end_ms:
            batch_end = min(cur + step_ms * limit - 1, end_ms)
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": "60",
                "start": cur,
                "end": batch_end,
                "limit": limit,
            }
            max_retries = 5
            success = False
            for attempt in range(max_retries):
                try:
                    resp = await client.get(url, params=params)
                    data = await resp.json()
                    if data.get("retCode") == 0:
                        success = True
                        break
                    else:
                        msg = str(data.get("retMsg", ""))
                        if data.get("retCode") == 10001 or "invalid" in msg.lower():
                            success = True
                            data["result"] = {}
                            break
                        await asyncio.sleep(1.0 * (attempt + 1))
                except Exception:
                    await asyncio.sleep(1.0 * (attempt + 1))

            if not success:
                cur = batch_end + 1
                continue

            items = ((data.get("result") or {}).get("list") or [])
            if not items:
                cur = batch_end + 1
                continue
            for it in items:
                ts = int(it[0])
                rows.append({
                    "timestamp": from_ms_jst(ts),
                    "bybit_open": float(it[1]),
                    "bybit_high": float(it[2]),
                    "bybit_low": float(it[3]),
                    "bybit_close": float(it[4]),
                    "bybit_volume": float(it[5]) if len(it) > 5 else 0.0,
                })
            last_ts = max(int(it[0]) for it in items)
            cur = max(batch_end + 1, last_ts + step_ms)
            await asyncio.sleep(0.02)
    return rows

async def fetch_bybit_funding_history(symbol: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    url = REST_API_URL["bybit"] + "/v5/market/funding/history"
    rows: List[Dict[str, Any]] = []
    cursor = end_ms
    async with pybotters.Client() as client:
        while True:
            params = {
                "category": "linear",
                "symbol": symbol,
                "endTime": str(cursor),
                "limit": "200",
            }
            max_retries = 5
            success = False
            for attempt in range(max_retries):
                try:
                    resp = await client.get(url, params=params)
                    data = await resp.json()
                    if data.get("retCode") == 0:
                        success = True
                        break
                    else:
                        msg = str(data.get("retMsg", ""))
                        if data.get("retCode") == 10001 or "invalid" in msg.lower():
                            success = True
                            data["result"] = {}
                            break
                        await asyncio.sleep(1.0 * (attempt + 1))
                except Exception:
                    await asyncio.sleep(1.0 * (attempt + 1))

            if not success or not data.get("result"):
                break
            items = (data.get("result").get("list") or [])
            if not items:
                break
            for it in items:
                ts = int(it.get("fundingRateTimestamp") or it.get("fundingTime") or 0)
                if not ts or ts < start_ms:
                    continue
                rows.append({
                    "timestamp": from_ms_jst(ts),
                    "bybit_fundingRate": float(it.get("fundingRate", 0.0)),
                })
            last_ts = int(items[-1].get("fundingRateTimestamp") or items[-1].get("fundingTime") or 0)
            if last_ts <= start_ms:
                break
            cursor = last_ts - 1
            await asyncio.sleep(0.05)
    rows.sort(key=lambda d: d["timestamp"])
    return rows

async def fetch_bybit_open_interest_1h(symbol: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    url = REST_API_URL["bybit"] + "/v5/market/open-interest"
    limit = 200
    step_ms = 60 * 60 * 1000
    rows: List[Dict[str, Any]] = []
    async with pybotters.Client() as client:
        cur = start_ms
        while cur < end_ms:
            batch_end = min(cur + step_ms * limit - 1, end_ms)
            params = {
                "category": "linear",
                "symbol": symbol,
                "intervalTime": "1h",
                "startTime": cur,
                "endTime": batch_end,
                "limit": limit,
            }
            max_retries = 5
            success = False
            for attempt in range(max_retries):
                try:
                    resp = await client.get(url, params=params)
                    data = await resp.json()
                    if data.get("retCode") == 0:
                        success = True
                        break
                    else:
                        msg = str(data.get("retMsg", ""))
                        if data.get("retCode") == 10001 or "invalid" in msg.lower():
                            success = True
                            data["result"] = {}
                            break
                        await asyncio.sleep(1.0 * (attempt + 1))
                except Exception:
                    await asyncio.sleep(1.0 * (attempt + 1))

            if not success:
                cur = batch_end + 1
                continue
            items = ((data.get("result") or {}).get("list") or [])
            if not items:
                cur = batch_end + 1
                continue
            for it in items:
                ts = int(it.get("timestamp") or 0)
                rows.append({
                    "timestamp": from_ms_jst(ts),
                    "bybit_openInterest": float(it.get("openInterest", 0.0)),
                })
            last_ts = max(int(it.get("timestamp") or 0) for it in items)
            cur = max(batch_end + 1, last_ts + step_ms)
            await asyncio.sleep(0.02)
    return rows

# --- Coinbase BTC-USD 非同期データフェッチ関数 ---

async def fetch_coinbase_candles_1h(start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    url = REST_API_URL["coinbase"] + "/products/BTC-USD/candles"
    rows: List[Dict[str, Any]] = []
    step_ms = 300 * 60 * 60 * 1000  # Coinbase 制限: 1リクエスト最大300データ
    cur_end = end_ms
    headers = {"User-Agent": "Mozilla/5.0"}
    async with pybotters.Client(headers=headers) as client:
        while cur_end > start_ms:
            cur_start = max(cur_end - step_ms, start_ms)
            start_iso = datetime.datetime.fromtimestamp(cur_start / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            end_iso = datetime.datetime.fromtimestamp(cur_end / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
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
                # it: [time, low, high, open, close, volume]
                ts_sec = int(it[0])
                rows.append({
                    "timestamp": from_ms_jst(ts_sec * 1000),
                    "coinbase_close": float(it[4])
                })
            cur_end = cur_start - 1
            await asyncio.sleep(0.2)
    rows.sort(key=lambda d: d["timestamp"])
    return rows

# --- メイン非同期実行制御 ---

async def main():
    log("Starting historical candles and indicators download...")
    
    # 過去730日分の期間を設定
    end_dt = datetime.datetime.now(JST).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - datetime.timedelta(days=730)
    
    start_ms = to_ms(start_dt)
    end_ms = to_ms(end_dt)
    
    log(f"Target Period: {start_dt} to {end_dt}")
    
    # 1. Coinbase の BTC-USD 終値を取得
    log("Fetching Coinbase BTC-USD hourly candles...")
    cb_rows = await fetch_coinbase_candles_1h(start_ms, end_ms)
    df_cb = pd.DataFrame(cb_rows)
    if df_cb.empty:
        log("Error: Failed to fetch Coinbase BTC-USD reference prices.")
        sys.exit(1)
    df_cb = df_cb.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    
    # 2. Bybit の BTCUSDT データを取得（プレミアム算出用）
    log("Fetching Bybit BTCUSDT for premium calculation...")
    bybit_btc_ohlcv = await fetch_bybit_ohlcv_1h("BTCUSDT", start_ms, end_ms)
    df_bb_btc = pd.DataFrame(bybit_btc_ohlcv)
    if df_bb_btc.empty:
        log("Error: Failed to fetch Bybit BTCUSDT reference prices.")
        sys.exit(1)
    df_bb_btc = df_bb_btc.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    
    # コインベースプレミアムを算出
    df_premium_calc = pd.merge(
        df_cb,
        df_bb_btc[["timestamp", "bybit_close"]].rename(columns={"bybit_close": "bybit_btc_close"}),
        on="timestamp",
        how="inner"
    )
    df_premium_calc["coinbase_premium"] = (df_premium_calc["coinbase_close"] - df_premium_calc["bybit_btc_close"]) / df_premium_calc["bybit_btc_close"] * 100
    df_premium = df_premium_calc[["timestamp", "coinbase_premium"]].copy()
    log(f"Generated Coinbase Premium series (length: {len(df_premium)})")
    
    # 各銘柄の並列ダウンロード用のセマフォ
    sem = asyncio.Semaphore(3)
    
    async def download_symbol(symbol: str):
        async with sem:
            log(f"Downloading {symbol} data...")
            try:
                # 3種のデータを非同期フェッチ
                tasks = [
                    fetch_bybit_ohlcv_1h(symbol, start_ms, end_ms),
                    fetch_bybit_open_interest_1h(symbol, start_ms, end_ms),
                    fetch_bybit_funding_history(symbol, start_ms, end_ms)
                ]
                ohlcv, oi, fund = await asyncio.gather(*tasks)
                
                if not ohlcv:
                    log(f"  WARNING: No ohlcv for {symbol}. Skipping.")
                    return None
                
                # DataFrame 化してマージ
                df_ohlcv = pd.DataFrame(ohlcv).drop_duplicates(subset=["timestamp"])
                df_ohlcv = df_ohlcv.set_index("timestamp").resample("1h").last().ffill().reset_index()
                
                if oi:
                    df_oi = pd.DataFrame(oi).drop_duplicates(subset=["timestamp"])
                    df_oi = df_oi.set_index("timestamp").resample("1h").last().ffill().reset_index()
                    df_ohlcv = pd.merge(df_ohlcv, df_oi, on="timestamp", how="left")
                else:
                    df_ohlcv["bybit_openInterest"] = 0.0
                    
                if fund:
                    df_fund = pd.DataFrame(fund).drop_duplicates(subset=["timestamp"])
                    df_fund = df_fund.set_index("timestamp").resample("1h").last().ffill().reset_index()
                    df_ohlcv = pd.merge(df_ohlcv, df_fund, on="timestamp", how="left")
                else:
                    df_ohlcv["bybit_fundingRate"] = 0.0
                
                # 穴埋め
                df_ohlcv["bybit_openInterest"] = df_ohlcv["bybit_openInterest"].ffill().bfill().fillna(0.0)
                df_ohlcv["bybit_fundingRate"] = df_ohlcv["bybit_fundingRate"].ffill().bfill().fillna(0.0)
                for col in ["bybit_open", "bybit_high", "bybit_low", "bybit_close", "bybit_volume"]:
                    df_ohlcv[col] = df_ohlcv[col].ffill().bfill()
                
                # コインベースプレミアムをマージ
                df_ohlcv = pd.merge(df_ohlcv, df_premium, on="timestamp", how="left")
                df_ohlcv["coinbase_premium"] = df_ohlcv["coinbase_premium"].ffill().bfill().fillna(0.0)
                
                # 識別子としてシンボルを追加
                df_ohlcv["symbol"] = symbol
                
                # 各銘柄個別のキャッシュとしても保存しておく（後々のデバッグ用）
                out_path = raw_cache_dir / f"{symbol}_bybit_raw.csv"
                df_ohlcv.to_csv(out_path, index=False)
                log(f"  SUCCESS: Saved {len(df_ohlcv)} rows for {symbol} to cache")
                
                return df_ohlcv
            except Exception as e:
                log(f"  ERROR: Failed to process {symbol}: {e}")
                return None

    # 全銘柄一括ダウンロード実行
    tasks = [download_symbol(sym) for sym in symbols]
    results = await asyncio.gather(*tasks)
    
    # 有効なデータをマージ
    valid_dfs = [df for df in results if df is not None and not df.empty]
    if not valid_dfs:
        log("Error: No data successfully fetched.")
        sys.exit(1)
        
    df_merged_all = pd.concat(valid_dfs, ignore_index=True)
    
    # タイムスタンプは文字列型に変換して保存
    df_merged_all["timestamp"] = df_merged_all["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    
    # ソートして保存
    df_merged_all = df_merged_all.sort_values(by=["timestamp", "symbol"]).reset_index(drop=True)
    merged_output_path = data_dir / "historical_all_symbols_merged.csv"
    
    log(f"Saving combined master dataset ({len(df_merged_all)} rows)...")
    df_merged_all.to_csv(merged_output_path, index=False)
    log(f"SUCCESS: Master dataset saved to {merged_output_path}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
