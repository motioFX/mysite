# === Time Synchronization & Import Path Setup ===
import sys
from pathlib import Path
import time
import requests

# Prioritize the directory containing this script for imports
script_dir = str(Path(__file__).resolve().parent)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    # Measure clock offset against Bitget server time
    r = requests.get('https://api.bitget.com/api/v2/public/time', timeout=5).json()
    if r.get('code') == '00000' and 'data' in r:
        server_time = int(r['data']['serverTime'])
        local_time = int(time.time() * 1000)
        offset_seconds = (local_time - server_time) / 1000.0
        if abs(offset_seconds) > 0.5:
            print(f"[TIME SYNC] Applying clock offset correction: {offset_seconds:+.3f}s")
            original_time = time.time
            time.time = lambda: original_time() - offset_seconds
            
            if hasattr(time, 'time_ns'):
                original_time_ns = time.time_ns
                time.time_ns = lambda: original_time_ns() - int(offset_seconds * 1_000_000_000)
except Exception as e:
    print(f"[TIME SYNC] Failed to sync time with Bitget: {e}")
# ================================================

from bitget5_44_2api_dual import (api_bitget, apis, RestAPI_url, flatten_current_position, flatten_all_positions, fetch_all_position_symbols, bitget_mode, apis_bitget, fetch_instrument_spec_bitget, compute_bitget_lot_size, BITGET_TARGET_POSITION_VALUE_USDT)
from bitget5_44_3logic import send_discord, logicinstance, PnLCalculator, MPStrategy, backtester, run_interval_comparison, resample_candles

from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import asyncio
import json
import math
import numpy as np
import pandas as pd
import pybotters

import requests
import subprocess
import sys
import time
from rich import print
from typing import Any, Dict, List, Optional, Set, Tuple
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    asyncio.set_event_loop_policy(None)

# 動作確認用: Trueで銘柄選定分析をスキップ（既存CSVスコアを使用）
# （__main__ の is_air フラグによって自動的に上書きされます）
SKIP_MIX_ANALYSIS: bool = True
is_air: bool = False
current_strategy_type: str = "range"

import signal
def signal_handler(sig, frame):
    print('\nプログラムを終了します')
    try:
        loop = asyncio.get_running_loop()
        loop.stop()
    except RuntimeError:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print('実行中... Ctrl+Cで停止できます')


Bybittimescale=True
if(Bybittimescale):
    interval_list = ['1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D']
    interval_map ={'1':1,'3':3,'5':5,'15':15,'30':30,'60':60,'120':120,'240':240,'360':360,'720':720,'D':1440}
else:
    interval_list = ['1m','3m','5m','15m','30m','1h','4h','6h','12h','1d']
    interval_map ={'1m':1,'3m':3,'5m':5,'15m':15,'30m':30,'1h':60,'4h':240,'6h':360,'12h':720,'1d':1440}

JST = timezone(timedelta(hours=9))
MIX_SCRIPT_PATH = Path(__file__).resolve().parent / "bitget5_44_4mix_candle_Merged_Alt9.py"
SCORES_CSV_PATH = MIX_SCRIPT_PATH.parent / "Data" / "symbol_selection_scores_all.csv"

TARGET_POSITION_VALUE_USDT = 100.0
LEVERAGE_FACTOR = 25.0
LEVERAGE_USAGE_RATIO = 0.25  # use 25% of exchange max leverage
TARGET_LEVERAGE = 8.5       # レバレッジ10倍設定時に証拠金不足にならないよう、少し低めの実質レバレッジ（例: 8.5倍）に調整

DAILY_ANALYSIS_HOUR = 11
DAILY_ANALYSIS_MINUTE = 30
TAKE_PROFIT_PCT_FOR_FLATTEN = 0.0015  # +0.15% over entry when flattening for symbol switch
SYMBOL_SWITCH_FLATTEN_START_HOUR = 12
SYMBOL_SWITCH_FORCE_CLOSE_HOUR = 15
FLATTEN_REISSUE_MINUTES = 60
PAUSED_FORCE_MARKET_DELAY_HOURS = 2  # Hours after paused mode starts before force market close

DEFAULT_INSTRUMENT_SPEC = {
    "qty_step": 0.01,
    "min_qty": 0.01,
    "max_qty": float("inf"),
    "min_notional": 0.0,
    "max_leverage": LEVERAGE_FACTOR,
}

instrument_spec: Dict[str, float] = DEFAULT_INSTRUMENT_SPEC.copy()
pnl_symbols: List[str] = []
trade_paused: bool = False
entry_candle_time = None

discord = send_discord()

def format_state_message(symbol: str, position: Dict[str, Any], open_orders_count: int, pending_flatten: bool, trade_side: str = "long") -> str:
    buy_qty = position.get("buy", 0.0)
    sell_qty = position.get("sell", 0.0)
    pnl = position.get("profit", 0.0)
    raw_pnl = position.get("raw_pnl", pnl)
    total_fee = position.get("total_fee", 0.0)

    fee_info = f" (価格差: {raw_pnl:+.2f}, 手数料: -{total_fee:.2f})" if total_fee > 0 else ""

    if buy_qty > 0:
        pos_str = f"🟢 保有: LONG {buy_qty} | 実質含み損益: {pnl:+.2f} USDT{fee_info}"
    elif sell_qty > 0:
        pos_str = f"🔴 保有: SHORT {sell_qty} | 実質含み損益: {pnl:+.2f} USDT{fee_info}"
    else:
        pos_str = f"⚪ 保有: なし (ノーポジ) | 方向: {trade_side.upper()}"

    extra = f" | 注文数: {open_orders_count}"
    if pending_flatten:
        extra += " | 全決済処理中"

    return f"{symbol} | {pos_str}{extra}"


def format_decision_message(exit_reason: Optional[str], evaluated: bool, fade_signal: bool, time_window: str, pnl: float, trade_side: str = "long") -> str:
    return f"exit_reason={exit_reason} (evaluated={evaluated}, fade_signal={fade_signal}, time_window={time_window}, net_pnl={pnl:+.2f})"


def log_state(message: str) -> None:
    discord.print_log(f"[STATE] {message}")


def log_decision(message: str) -> None:
    discord.print_log(f"[DECISION] {message}")


def log_action(message: str) -> None:
    discord.print_log(f"[ACTION] {message}")


def decide_exit_reason(ctx: Dict[str, Any]) -> Optional[str]:
    """
    Decide a single exit_reason based on priority.
    Priority: force_flatten > fade_take_profit.
    Returns None when no exit condition is active.
    """
    if ctx.get("force_flatten"):
        return "force_flatten"
    if ctx.get("fade_take_profit"):
        return "fade_take_profit"
    return None


def extract_scoring_candidates(trade_plan: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if not trade_plan:
        return candidates
    scoring = trade_plan.get("scoring")
    if not isinstance(scoring, dict):
        return candidates
    raw_candidates = scoring.get("top_candidates")
    if not isinstance(raw_candidates, list):
        return candidates
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        symbol = raw.get("symbol")
        if not symbol:
            continue
        symbol_str = str(symbol).upper()
        score = raw.get("score", 0.0)
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        components_raw = raw.get("components", {})
        components: Dict[str, float] = {}
        if isinstance(components_raw, dict):
            for key, value in components_raw.items():
                try:
                    components[key] = float(value)
                except (TypeError, ValueError):
                    continue
        candidates.append(
            {
                "symbol": symbol_str,
                "score": score_value,
                "components": components,
            }
        )
    return candidates


def select_symbol_from_trade_plan(trade_plan: Optional[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]], str]:
    candidates = extract_scoring_candidates(trade_plan)
    side = str((trade_plan or {}).get("trade_side", "long")).lower()
    if candidates:
        return candidates[0]["symbol"], candidates, side
    fallback_symbol = (trade_plan or {}).get("trade_symbol")
    if fallback_symbol:
        return str(fallback_symbol).upper(), candidates, side
    return None, candidates, side


def log_scoring_candidates(candidates: List[Dict[str, Any]]) -> None:
    if not candidates:
        return
    discord.print_log("Weighted score top candidates:")
    for idx, candidate in enumerate(candidates, 1):
        components = candidate.get("components") or {}
        component_parts: List[str] = []
        for label in ("full", "30d", "10d"):
            if label in components:
                component_parts.append(f"{label}={components[label]:.4f}")
        if not component_parts:
            component_parts.append("components=NA")
        discord.print_log(
            f"  {idx}. {candidate['symbol']} score={candidate['score']:.4f} ({', '.join(component_parts)})"
        )


async def fetch_last_price(symbol: str, product_type: str, mode: str) -> Optional[float]:
    # Redirect to Bitget fetch to avoid Bybit dependency
    import bitget5_44_2api_dual
    bg_mode = getattr(bitget5_44_2api_dual, 'bitget_mode', 'live')
    return await fetch_last_price_bitget(symbol, product_type or 'USDT-FUTURES', bg_mode)


async def fetch_last_price_bitget(symbol: str, product_type: str, mode: str) -> Optional[float]:
    base_url = RestAPI_url["bitget"]
    params = {"symbol": symbol, "productType": product_type}
    try:
        # パブリックAPIのため、デモモードであっても本番環境の価格を取得します（デモ環境に銘柄が存在しないエラーを回避するため）
        async with pybotters.Client(base_url=base_url) as client:
            response = await client.get("/api/v2/mix/market/ticker", params=params)
            data = await response.json()
    except Exception as exc:
        discord.print_log(f"Bitget 価格取得エラー: {exc}")
        return None
    if not isinstance(data, dict) or data.get("code") != "00000":
        return None
    ticker_data = data.get("data") or []
    if isinstance(ticker_data, dict):
        ticker_data = [ticker_data]
    if not ticker_data:
        return None
    ticker = ticker_data[0]
    for key in ("lastPr", "markPr", "indexPr", "bidPr"):
        value = ticker.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


async def get_btc_performance(mode: str) -> Tuple[float, float]:
    """
    BTCの3時間前および24時間前の終値に対する、現在のマーク価格の比率を返します。
    戻り値: (btc_perf_3h, btc_perf_24h)
    """
    base_url = RestAPI_url["bybit_demo" if mode == "demo" else "bybit"]
    # 直近25本の1時間足を取得
    params = {"category": "linear", "symbol": "BTCUSDT", "interval": "60", "limit": 25}
    try:
        async with pybotters.Client(apis=apis, base_url=base_url) as client:
            response = await client.get("/v5/market/kline", params=params)
            data = await response.json()
            
            ticker_response = await client.get("/v5/market/tickers", params={"category": "linear", "symbol": "BTCUSDT"})
            ticker_data = await ticker_response.json()
    except Exception as exc:
        print(f"[BTC PERFORMANCE] Fetch error: {exc}")
        return 1.0, 1.0
        
    klines = ((data or {}).get("result") or {}).get("list") or []
    if len(klines) < 24:
        return 1.0, 1.0
        
    # klinesは最新順に並んでいる [最新, 1日前, 2日前...]
    try:
        ticker_list = ((ticker_data or {}).get("result") or {}).get("list") or []
        price_now = float(ticker_list[0].get("lastPrice")) if ticker_list else float(klines[0][4])
    except Exception:
        price_now = float(klines[0][4])
        
    try:
        price_3h = float(klines[3][4])
        perf_3h = price_now / price_3h
    except Exception:
        perf_3h = 1.0
        
    try:
        price_24h = float(klines[min(24, len(klines)-1)][4])
        perf_24h = price_now / price_24h
    except Exception:
        perf_24h = 1.0
        
    return perf_3h, perf_24h


async def get_btc_lot_multiplier(mode: str, current_side: str) -> Tuple[float, str]:
    """
    BTCの3時間急落（勝率100%ゾーン）や24時間下落トレンドを判定し、
    ショートエントリー時のロット倍率と判定理由を返します。
    """
    if current_side != "short":
        return 1.0, "LONG_MODE (NORMAL)"
        
    perf_3h, perf_24h = await get_btc_performance(mode)
    
    # 3時間で-0.5%以上の急落 ➔ 1.0倍ロット (即エントリー)
    if perf_3h <= 0.995:
        return 1.0, f"BTC 3H DROP (perf_3h={perf_3h:.4f}, perf_24h={perf_24h:.4f})"
    # 24時間でマイナス ➔ 1.0倍ロット
    elif perf_24h < 1.0:
        return 1.0, f"BTC 24H DOWN TREND (perf_3h={perf_3h:.4f}, perf_24h={perf_24h:.4f})"
    # BTCが下落していない ➔ ショート禁止 (0.0倍ロット ＝ 実質エントリーしない)
    else:
        return 0.0, f"BTC NO DOWN (perf_3h={perf_3h:.4f}, perf_24h={perf_24h:.4f})"


def wait_until_analysis_time(target_hour: int = 9, target_minute: int = 0) -> None:
    now_jst = datetime.now(JST)
    target = now_jst.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    if now_jst >= target:
        discord.print_log(f"Past {target_hour}:{target_minute:02d}, running analysis immediately.")
        return
    wait_seconds = (target - now_jst).total_seconds()
    minutes = wait_seconds / 60
    discord.print_log(f"Waiting about {minutes:.1f} minutes until {target_hour}:{target_minute:02d}.")
    time.sleep(wait_seconds)

def load_top_symbol_from_scores(trade_side: str = "long") -> Optional[str]:
    """CSVの1位シンボルを取得。"""
    try:
        df = pd.read_csv(SCORES_CSV_PATH)
    except FileNotFoundError:
        discord.print_log(f"スコアCSVが見つかりません: {SCORES_CSV_PATH}")
        return None
    except Exception as exc:
        discord.print_log(f"スコアCSV読込エラー: {exc}")
        return None
    if df.empty or "symbol" not in df.columns:
        discord.print_log("スコアCSVにシンボル列がありません、または空です")
        return None

    score_col = "score" if "score" in df.columns else None
    if not score_col:
        discord.print_log("Score列が見つからないため、トレードをスキップします。")
        return None

    # v9スコアはロング・ショートとも正の数が上位（降順ソート）
    df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    ban_col = "ban_short" if trade_side == "short" else "ban_long"

    target_row = None
    for idx, row in df_sorted.iterrows():
        is_banned = False
        if ban_col in df_sorted.columns:
            is_banned = bool(row[ban_col])
        if not is_banned:
            target_row = row
            break

    if target_row is None:
        side_label = "ショート" if trade_side == "short" else "ロング"
        discord.print_log(f"すべての{side_label}候補銘柄が禁止対象（売られすぎ/買われすぎ、または低流動性）のため、トレードなし。")
        return None
        
    target_symbol = str(target_row.get("symbol", "")).upper()
    try:
        target_score = float(target_row[score_col])
    except (TypeError, ValueError):
        target_score = float("nan")

    if not math.isfinite(target_score):
        discord.print_log("選定されたスコアがNaN/Infのため、トレードをスキップします。")
        return None
        
    if target_score <= 0.0:
        discord.print_log(f"選定シンボル {target_symbol or '[EMPTY]'} のスコアが0以下({target_score:.4f})のため、トレードなし。")
        return None

    side_label = "ショート用" if trade_side == "short" else "ロング用"
    discord.print_log(f"スコアCSVの1位を採用({side_label}): {target_symbol} (score={target_score:.4f}, col={score_col})")
    return target_symbol


def select_candidates_from_scores(trade_side: str = "long", limit: int = 15) -> List[str]:
    """CSVから上位のシンボルを指定件数取得。"""
    try:
        df = pd.read_csv(SCORES_CSV_PATH)
    except FileNotFoundError:
        discord.print_log(f"スコアCSVが見つかりません: {SCORES_CSV_PATH}")
        return []
    except Exception as exc:
        discord.print_log(f"スコアCSV読込エラー: {exc}")
        return []
    if df.empty or "symbol" not in df.columns:
        discord.print_log("スコアCSVにシンボル列がありません、または空です")
        return []

    score_col = "score" if "score" in df.columns else None
    if not score_col:
        discord.print_log("Score列が見つからないため、候補選定をスキップします。")
        return []

    df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    ban_col = "ban_short" if trade_side == "short" else "ban_long"

    targets = []
    for idx, row in df_sorted.iterrows():
        is_banned = False
        if ban_col in df_sorted.columns:
            is_banned = bool(row[ban_col])
        if not is_banned:
            symbol = str(row.get("symbol", "")).upper()
            try:
                score_val = float(row[score_col])
            except (TypeError, ValueError):
                score_val = float("nan")
            if math.isfinite(score_val) and score_val > 0.0:
                targets.append(symbol)
                if limit > 0 and len(targets) >= limit:
                    break

    return targets
                    
    return targets


def check_skip_mix_analysis() -> bool:
    """
    本日の銘柄選定データがすでに更新されているか確認します。
    ファイルが存在し、かつ最終更新日が本日（JST）であればTrue、そうでなければFalseを返します。
    """
    if not SCORES_CSV_PATH.exists():
        return False
    mtime = SCORES_CSV_PATH.stat().st_mtime
    mtime_dt = datetime.fromtimestamp(mtime, tz=JST)
    now_dt = datetime.now(JST)
    return mtime_dt.date() == now_dt.date()


def run_mix_analysis(mode: str) -> Optional[Dict[str, Any]]:
    global SKIP_MIX_ANALYSIS
    SKIP_MIX_ANALYSIS = check_skip_mix_analysis()

    # 最初に Market State を読み込んで trade_side を決定する
    market_state_file = MIX_SCRIPT_PATH.parent / "Data" / "market_state.json"
    trade_side = "long"
    if market_state_file.exists():
        try:
            with open(market_state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                state = data.get("market_state", "long_only")
                if state == "long_only":
                    trade_side = "long"
                elif state == "short_only":
                    trade_side = "short"
        except Exception as e:
            discord.print_log(f"Failed to read market_state.json: {e}")
    discord.print_log(f"Market state logic: Set trade_side to {trade_side}")

    if SKIP_MIX_ANALYSIS:
        discord.print_log("SKIP_MIX_ANALYSIS=True: 銘柄選定分析をスキップし、既存CSVスコアを使用します。")
        top_symbol = load_top_symbol_from_scores(trade_side=trade_side)
        return {"trade_symbol": top_symbol, "trade_side": trade_side}

    if not MIX_SCRIPT_PATH.exists():
        discord.print_log(f"ミックス分析スクリプトが見つかりません: {MIX_SCRIPT_PATH}")
        top_symbol = load_top_symbol_from_scores(trade_side=trade_side)
        return {"trade_symbol": top_symbol, "trade_side": trade_side}
    cmd = [sys.executable, str(MIX_SCRIPT_PATH)]
    cmd.extend(["--demo-mode", "demo" if bitget_mode == "demo" else "live"])
    try:
        discord.print_log("ミックス分析スクリプトを実行します。")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        discord.print_log(f"ミックス分析スクリプトの実行に失敗しました: {exc}")
    except Exception as exc:
        discord.print_log(f"ミックス分析スクリプトの実行中にエラー発生: {exc}")

    # 分析スクリプト実行後に再度 Market State を読み込む（最新状態を反映）
    if market_state_file.exists():
        try:
            with open(market_state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                state = data.get("market_state", "long_only")
                if state == "long_only":
                    trade_side = "long"
                elif state == "short_only":
                    trade_side = "short"
        except Exception as e:
            discord.print_log(f"Failed to read market_state.json: {e}")
            
    discord.print_log(f"Market state logic: Set trade_side to {trade_side}")

    # スコアCSVから上位10銘柄のスコアをDiscordへ出力
    try:
        df_scores = pd.read_csv(SCORES_CSV_PATH)
        if not df_scores.empty and "symbol" in df_scores.columns and "score" in df_scores.columns:
            df_sorted_scores = df_scores.sort_values("score", ascending=False)
            
            top_n = df_sorted_scores.head(10)
            score_lines = [f"【選定候補スコアランキング (方向: {trade_side.upper()})】"]
            ban_col = "ban_short" if trade_side == "short" else "ban_long"
            for idx, row in enumerate(top_n.itertuples(), 1):
                is_banned = getattr(row, ban_col, False) if hasattr(row, ban_col) else False
                ban_str = " (Banned)" if is_banned else ""
                score_lines.append(f"{idx}. {row.symbol}: {row.score:.4f}{ban_str}")
            
            discord.print_log("\n".join(score_lines))
    except Exception as exc:
        discord.print_log(f"選定候補スコアランキングの出力に失敗しました: {exc}")

    top_symbol = load_top_symbol_from_scores(trade_side=trade_side)
    return {"trade_symbol": top_symbol, "trade_side": trade_side}


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fetch_instrument_spec(symbol: str, category: str, mode: str) -> Optional[Dict[str, float]]:
    base_url = RestAPI_url["bybit_demo" if mode == "demo" else "bybit"]
    endpoint = f"{base_url}/v5/market/instruments-info"
    try:
        response = requests.get(
            endpoint,
            params={"category": category, "symbol": symbol},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        discord.print_log(f"銘柄仕様の取得に失敗しました: {exc}")
        return None
    if payload.get("retCode") != 0:
        ret_code = payload.get("retCode")
        if ret_code == 10002:
            # timestamp/recv_window error -> stop immediately
            discord.print_log(f"retCode 10002: timestamp/recv_window error, stopping: {payload}")
            sys.exit("retCode 10002 detected")
        # print(f"Instrument spec response contains error: {payload}")
        return None
    items = ((payload.get("result") or {}).get("list") or [])
    if not items:
        discord.print_log("銘柄仕様に対象シンボルが見つかりませんでした。")
        return None
    item = items[0]
    lot_filter = item.get("lotSizeFilter", {})
    leverage_filter = item.get("leverageFilter", {})
    return {
        "qty_step": _to_float(lot_filter.get("qtyStep"), 0.01),
        "min_qty": _to_float(lot_filter.get("minOrderQty"), 0.01),
        "max_qty": _to_float(lot_filter.get("maxOrderQty"), float("inf")),
        "min_notional": _to_float(lot_filter.get("minNotionalValue"), 0.0),
        "max_leverage": _to_float(leverage_filter.get("maxLeverage"), LEVERAGE_FACTOR),
    }


async def fetch_instrument_spec_async(symbol: str, category: str, mode: str) -> Optional[Dict[str, float]]:
    """Non-blocking wrapper for instrument spec lookup."""
    return await asyncio.to_thread(fetch_instrument_spec, symbol, category, mode)


def quantize_quantity(quantity: float, step: float) -> float:
    if step <= 0:
        return quantity
    # Normalize to drop trailing zeros (e.g. "1.0" -> "1") so integer steps quantize correctly.
    decimal_step = Decimal(str(step)).normalize()
    return float(Decimal(str(quantity)).quantize(decimal_step, rounding=ROUND_HALF_UP))

# Keep trade history outside the Data/ folder so mix script cleanup won't delete it.
TRADED_SYMBOLS_PATH = MIX_SCRIPT_PATH.parent / "traded_symbols.json"
TRADED_SYMBOL_RETENTION_DAYS = 30


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_added_at(value: Any, fallback: datetime) -> datetime:
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    return fallback


def _load_traded_entries() -> List[Dict[str, Any]]:
    try:
        with open(TRADED_SYMBOLS_PATH, "r") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        return []
    except Exception as exc:
        discord.print_log(f"traded symbols load error: {exc}")
        return []

    now = _now_utc()
    cutoff = now - timedelta(days=TRADED_SYMBOL_RETENTION_DAYS)
    entries: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            symbol: Optional[str] = None
            added_at = now
            if isinstance(item, dict):
                symbol = item.get("symbol") or item.get("sym")
                added_at = _parse_added_at(item.get("added_at") or item.get("added"), now)
            else:
                symbol = str(item)
            if not symbol:
                continue
            symbol = symbol.upper()
            if added_at < cutoff:
                continue
            entries.append({"symbol": symbol, "added_at": added_at})
    return entries


def load_traded_symbols() -> List[str]:
    entries = _load_traded_entries()
    # Deduplicate while preserving order
    seen: Set[str] = set()
    symbols: List[str] = []
    for entry in entries:
        sym = entry["symbol"]
        if sym in seen:
            continue
        seen.add(sym)
        symbols.append(sym)
    return symbols


def save_traded_symbols(entries: List[Dict[str, Any]]) -> None:
    try:
        TRADED_SYMBOLS_PATH.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for entry in entries:
            sym = entry.get("symbol")
            if not sym:
                continue
            added_at = entry.get("added_at") or _now_utc()
            if isinstance(added_at, datetime):
                added_at_str = added_at.isoformat()
            else:
                added_at_str = str(added_at)
            serializable.append({"symbol": str(sym).upper(), "added_at": added_at_str})
        with open(TRADED_SYMBOLS_PATH, "w") as fp:
            json.dump(serializable, fp, ensure_ascii=True, indent=2)
    except Exception as exc:
        discord.print_log(f"traded symbols save error: {exc}")


def remember_symbols(symbols: List[str]) -> List[str]:
    now = _now_utc()
    existing_entries = _load_traded_entries()
    existing_map: Dict[str, Dict[str, Any]] = {e["symbol"]: e for e in existing_entries}

    for sym in symbols:
        if not sym:
            continue
        s = str(sym).upper()
        existing_map[s] = {"symbol": s, "added_at": now}

    # Preserve original order, append new symbols at the end
    combined: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for entry in existing_entries:
        sym = entry["symbol"]
        if sym in seen:
            continue
        combined.append(existing_map[sym])
        seen.add(sym)
    for sym, entry in existing_map.items():
        if sym in seen:
            continue
        combined.append(entry)
        seen.add(sym)

    cutoff = now - timedelta(days=TRADED_SYMBOL_RETENTION_DAYS)
    pruned = [e for e in combined if e.get("added_at", now) >= cutoff]
    save_traded_symbols(pruned)
    return [e["symbol"] for e in pruned]


def build_pnl_symbol_pool(trade_plan: Optional[Dict[str, Any]], current_symbol: str) -> List[str]:
    candidate_syms = [c["symbol"] for c in extract_scoring_candidates(trade_plan)][:20]
    history = load_traded_symbols()
    merged = list(dict.fromkeys([current_symbol] + candidate_syms + history))
    return merged


def compute_lot_size(
    price: Optional[float],
    target_position_value_usdt: float,
    spec: Dict[str, float],
) -> float:
    qty_step = spec.get("qty_step", 0.01)
    min_qty = spec.get("min_qty", 0.01)
    max_qty = spec.get("max_qty", float("inf"))
    min_notional = spec.get("min_notional", 0.0)

    if price is None or price <= 0:
        base_qty = min_qty
    else:
        base_qty = target_position_value_usdt / price

    quantity = quantize_quantity(base_qty, qty_step)
    if quantity < min_qty:
        quantity = quantize_quantity(min_qty, qty_step)
    if price and min_notional > 0 and quantity * price < min_notional:
        required_qty = min_notional / price
        quantity = quantize_quantity(max(required_qty, quantity), qty_step)
    if math.isfinite(max_qty):
        quantity = min(quantity, max_qty)
    return quantity


def get_bitget_granularity(bybit_interval: str) -> str:
    mapping = {
        '1': '1m',
        '3': '3m',
        '5': '5m',
        '15': '15m',
        '30': '30m',
        '60': '1H',
        '120': '2H',
        '240': '4H',
        'D': '1D',
    }
    return mapping.get(str(bybit_interval), '1H')


async def fetch_bitget_candles(symbol: str, granularity: str, limit: int = 150) -> pd.DataFrame:
    """BitgetのKlinesを取得し、DataFrameとして整形"""
    import pybotters
    url = "https://api.bitget.com/api/v2/mix/market/candles"
    params = {
        "symbol": symbol,
        "productType": "USDT-FUTURES",
        "granularity": granularity,
        "limit": str(limit)
    }
    try:
        async with pybotters.Client() as client:
            resp = await asyncio.wait_for(client.get(url, params=params), timeout=30)
            data = await resp.json()
            if data and data.get("code") == "00000" and data.get("data"):
                rows = data["data"]
                df = pd.DataFrame(rows)
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
                return df
    except Exception as e:
        print(f"Failed to fetch Bitget candles: {e}")
    return None


async def plot_pnl():
    # Bitget の PNL 処理
    try:
        calculator = PnLCalculator(apis, RestAPI_url, symbol, mode)
        bg_trades = await calculator.get_bitget_trade_history(apis_bitget, days_back=30)
        
        if not bg_trades:
            discord.print_log(f"【Bitget】取引履歴が見つかりませんでした（新規口座、または最近の取引がない可能性があります）")
        else:
            df_bg = pd.DataFrame(bg_trades)
            df_bg_pnl = calculator.calculate_bitget_pnl_from_df(df_bg)
            if df_bg_pnl is not None and not df_bg_pnl.empty:
                final_pnl = df_bg_pnl['cumulative_pnl'].iloc[-1]
                # discord.print_log(f"【Bitget】取引履歴が {len(df_bg)} 件見つかりました。累積損益: {final_pnl:.2f} USDT")
                calculator.plot_pnl(df_bg_pnl, save_path="backtest_data/bitget_pnl_graph.jpg", label="Bitget")
            else:
                discord.print_log(f"【Bitget】取引履歴が見つかりませんでした（新規口座、または最近の取引がない可能性があります）")
    except Exception as e:
        discord.print_log(f"Bitget PNL計算中にエラーが発生しました: {e}")
        print(f"Bitget PNL Error: {e}")

def should_display_chart(dt_now, interval_int):
    if interval_int <= 3:
        return dt_now.minute % 10 == 0
    elif 4 <= interval_int < 60:
        return dt_now.minute % 20 == 0
    else:
        return dt_now.hour % 4 == 0


async def generate_bitget_backtest_chart(symbol: str, interval: str, df_loop: pd.DataFrame, best_mp: int, best_margin: float, best_er: float, trade_side: str, api, current_bg_target_value: float, bybit_exec_history=None, bybit_lot_size=None) -> None:
    """BitgetのKlinesを取得し、Bybitのシグナルをマッピングして約定シミュレーションを実行、グラフを出力します。"""
    try:
        granularity = get_bitget_granularity(interval)
        df_bg_candles = await fetch_bitget_candles(symbol, granularity, limit=150)
        if df_bg_candles is None or df_bg_candles.empty:
            discord.print_log("Warning: Failed to fetch Bitget candles or empty data, skipping Bitget backtest chart.")
            return

        df_bg_loop = df_loop.copy()
        bg_map = df_bg_candles.set_index('timestamp')
        for col in ['open', 'high', 'low', 'close']:
            if col in bg_map.columns:
                df_bg_loop[col] = df_bg_loop['timestamp'].map(bg_map[col]).fillna(df_bg_loop[col])
        
        try:
            # Bitget価格でインディケータとシグナルを再計算
            df_bg_loop = logic.make_logic(df_bg_loop, market_profile_period=best_mp, er_threshold=best_er, strategy_type=current_strategy_type)
        except Exception as e:
            print(f"Error recomputing indicators for Bitget backtest: {e}")

        # Bitgetの口座残高とロットサイズを取得 (例外対策・フォールバック付き)
        import bitget5_44_2api_dual
        
        # 1. 口座残高の取得とフォールバック
        try:
            bitget_onhand_amount = await api.bitget.get_account()
        except Exception as e:
            print(f"Warning: Failed to get Bitget account balance for chart: {e}")
            bitget_onhand_amount = float(df_loop['pnl'].iloc[-1]) * 0.5 if 'pnl' in df_loop.columns else 100.0
            if pd.isna(bitget_onhand_amount) or bitget_onhand_amount <= 0:
                bitget_onhand_amount = 100.0

        # 2. 最新価格の取得とフォールバック
        try:
            bg_latest_price = await fetch_last_price_bitget(symbol, 'USDT-FUTURES', bitget5_44_2api_dual.bitget_mode)
            if bg_latest_price is None:
                raise ValueError("fetch_last_price_bitget returned None")
        except Exception as e:
            print(f"Warning: Failed to get Bitget last price for chart: {e}")
            bg_latest_price = float(df_bg_candles['close'].iloc[-1])

        # 3. 銘柄仕様の取得とフォールバック
        try:
            bg_spec = fetch_instrument_spec_bitget(symbol, 'USDT-FUTURES', bitget5_44_2api_dual.bitget_mode)
            if bg_spec is None:
                raise ValueError("fetch_instrument_spec_bitget returned None")
        except Exception as e:
            print(f"Warning: Failed to get Bitget spec for chart: {e}")
            bg_spec = {"qty_step": 1.0, "min_qty": 1.0, "price_place": 4.0}

        # 4. ロットサイズの計算とフォールバック
        try:
            bitget_actual_lot_size = compute_bitget_lot_size(bg_latest_price, current_bg_target_value, bg_spec)
        except Exception as e:
            print(f"Warning: Failed to compute Bitget lot size for chart: {e}")
            bitget_actual_lot_size = bybit_lot_size if bybit_lot_size is not None else 1.0

        bt_bg = backtester()
        df_bg_loop = bt_bg.run_backtest(
            df=df_bg_loop, 
            lot=bitget_actual_lot_size, 
            data_equity=bitget_onhand_amount, 
            side_mode=trade_side,
            mp_period=best_mp, 
            er_threshold=best_er,
            bybit_exec_history=bybit_exec_history,
            bybit_lot_size=bybit_lot_size,
            strategy_type=current_strategy_type,
            sl_margin_pct=best_margin
        )
        df_bg_loop.to_csv("./backtest_data/klines100_bitget.csv")
        await asyncio.sleep(1)
        discord.plot_backtest(label="Bitget_MP", csv_file="backtest_data/klines100_bitget.csv", symbol=symbol)
    except Exception as e:
        discord.print_log(f"Failed to generate Bitget backtest: {e}")


async def validate_profitable_candidates(trade_side: str, mode: str, base_symbol: str, interval: str, interval_int: int) -> Tuple[List[str], str, int, int, float, float]:
    """1位の銘柄でパラメータを最適化し、そのパラメータを使って上位銘柄をバックテスト検証してプラスになる3銘柄を選定する"""
    discord.print_log(f"1位銘柄 {base_symbol} で自動最適化（バックテストMTF比較）を実行します...")
    
    # ベース銘柄のデータ取得
    base_spec = (await fetch_instrument_spec_async(base_symbol, category, mode)) or DEFAULT_INSTRUMENT_SPEC.copy()
    base_api = api_bitget(base_symbol, category, coin, mode, instrument_spec=base_spec)
    usdt_onhand = await base_api.get_account()
    
    base_price = await fetch_last_price(base_symbol, category, mode)
    base_target = max(100.0, usdt_onhand)
    base_lot = compute_lot_size(base_price, base_target, base_spec)
    
    df_base = pd.DataFrame()
    df_base = await base_api.get_candle(df_base, interval, interval_int)
    
    best_strategy = "range"
    if df_base is None or df_base.empty or len(df_base) <= 1:
        discord.print_log(f"ベース銘柄 {base_symbol} のデータ取得に失敗しました。デフォルトパラメータを使用します。")
        best_interval, best_mp, best_er, best_atr = 60, 20, 0.5, 1.0
    else:
        df_base = df_base.iloc[:-1].reset_index(drop=True)
        results, best_strategy, best_interval, best_mp, best_er, best_margin = run_interval_comparison(
            df_60m=df_base, lot=base_lot, data_equity=usdt_onhand, side_mode=trade_side
        )
        
    discord.print_log(f"★ 最適化パラメーター決定: 戦略={base_symbol} {best_strategy.upper()} {trade_side.upper()}, interval={best_interval}, MP={best_mp}, ER={best_er}, Margin={best_margin}%")
    
    # 候補銘柄の検証
    discord.print_log("候補銘柄のバックテスト検証を開始し、プラスになる3銘柄を選定します...")
    all_candidates = select_candidates_from_scores(trade_side, limit=15)
    profitable_cands = []
    
    for cand in all_candidates:
        if len(profitable_cands) >= 3:
            break
            
        discord.print_log(f"検証中: {cand}")
        cand_spec = (await fetch_instrument_spec_async(cand, category, mode)) or DEFAULT_INSTRUMENT_SPEC.copy()
        cand_api = api_bitget(cand, category, coin, mode, instrument_spec=cand_spec)
        cand_onhand = await cand_api.get_account()
        
        cand_df = pd.DataFrame()
        cand_df = await cand_api.get_candle(cand_df, interval, interval_int)
        
        if cand_df is None or cand_df.empty or len(cand_df) <= 1:
            discord.print_log(f"  -> データ不足のためスキップ")
            continue
            
        cand_df = cand_df.iloc[:-1].reset_index(drop=True)
        cand_df_loop = resample_candles(cand_df, best_interval)
        cand_df_loop = logic.make_logic(cand_df_loop, market_profile_period=best_mp, er_threshold=best_er, strategy_type=best_strategy)
        
        cand_price = await fetch_last_price(cand, category, mode)
        cand_target = max(100.0, cand_onhand)
        cand_lot = compute_lot_size(cand_price, cand_target, cand_spec)
        
        bt_cand = backtester()
        cand_df_best = bt_cand.run_backtest(
            df=cand_df_loop, lot=cand_lot, data_equity=cand_onhand,
            side_mode=trade_side, mp_period=best_mp,
            er_threshold=best_er, strategy_type=best_strategy, sl_margin_pct=best_margin
        )
        
        if not cand_df_best.empty and 'pnl' in cand_df_best.columns:
            final_pnl = cand_df_best['pnl'].iloc[-1]
            if final_pnl > cand_onhand:
                discord.print_log(f"  -> 合格! PnL: {final_pnl:.2f} USDT (初期: {cand_onhand:.2f})")
                profitable_cands.append(cand)
            else:
                discord.print_log(f"  -> 不合格 (マイナス/同値) PnL: {final_pnl:.2f} USDT")
        else:
            discord.print_log(f"  -> バックテスト失敗のためスキップ")
            
    return profitable_cands, best_strategy, best_interval, best_mp, best_er, best_margin

logic = logicinstance()
#ボット起動**********************************************************************************************
async def start(mode, max_lot, interval):
    global symbol, instrument_spec, trade_side, pnl_symbols, trade_paused, current_strategy_type
    startup_top_symbols = []
    
    interval_int = interval_map.get(interval)
    df = pd.DataFrame()
    best_mp = 20
    best_er = 0.5
    best_margin = 1.0
    best_atr = 1.0
    best_interval = 60
    
    trade_plan = run_mix_analysis(mode)
    trade_paused = not bool(trade_plan and trade_plan.get("trade_symbol"))
    trade_label = trade_plan.get("trade_label") if trade_plan else None
    if trade_paused:
        discord.print_log("スコア不足/Zスコアマイナスのため、今日は待機モードで開始します。")
        selected_symbol = None
        trade_side = trade_plan.get("trade_side", trade_side) if trade_plan else trade_side
        startup_top_symbols = select_candidates_from_scores(trade_side, limit=15)
        if startup_top_symbols:
            symbol = startup_top_symbols[0]
    else:
        trade_side = trade_plan.get("trade_side", trade_side) if trade_plan else trade_side
        all_candidates = select_candidates_from_scores(trade_side, limit=15)
        if all_candidates:
            base_symbol = all_candidates[0]
            discord.print_log(f"Selected base symbol from trade plan: {base_symbol} (side={trade_side})")
            
            # 1位の銘柄で最適化し、検証済み3銘柄を選定
            profitable_cands, best_strategy, best_interval, best_mp, best_er, best_margin = await validate_profitable_candidates(
                trade_side, mode, base_symbol, interval, interval_int
            )
            current_strategy_type = best_strategy
            
            startup_top_symbols = profitable_cands if profitable_cands else all_candidates
            if startup_top_symbols:
                selected_symbol = startup_top_symbols[0]
                symbol = selected_symbol
                discord.print_log(f"\n{symbol} {trade_side.upper()} {current_strategy_type.upper()}")
                discord.print_log(f"Active symbol set to: {symbol}")
                discord.print_log(f"Top profitable candidates: {', '.join(startup_top_symbols)}")
            else:
                discord.print_log("合格する銘柄が1つもなかったため、待機モードで開始します。")
                trade_paused = True
                selected_symbol = None
                symbol = base_symbol  # 念のためベースを設定
        else:
            discord.print_log("トレード対象がないため、待機モードで開始します。")
            trade_paused = True
            selected_symbol = None

    if trade_label:
        discord.print_log(trade_label)

    # Remember all startup top candidates
    remember_symbols(startup_top_symbols if startup_top_symbols else [symbol])
    pnl_symbols = build_pnl_symbol_pool(trade_plan, symbol)

    instrument_spec = fetch_instrument_spec(symbol, category, mode) or DEFAULT_INSTRUMENT_SPEC.copy()
    # discord.print_log(
    #     f"[BYBIT] {symbol} 仕様確認: qty_step={instrument_spec['qty_step']} min_qty={instrument_spec['min_qty']} "
    #     f"max_qty={instrument_spec['max_qty']} max_leverage={instrument_spec['max_leverage']}"
    # )

    leverage_cap = instrument_spec.get("max_leverage", LEVERAGE_FACTOR) or LEVERAGE_FACTOR
    requested_leverage = leverage_cap * LEVERAGE_USAGE_RATIO
    leverage_factor = max(1.0, min(LEVERAGE_FACTOR, requested_leverage))
    # discord.print_log(
    #     f"[BYBIT] Leverage cap {leverage_cap}x -> using {leverage_factor:.2f}x (ratio {LEVERAGE_USAGE_RATIO:.2f})"
    # )

    local_api = api_bitget(symbol, category, coin, mode, instrument_spec=instrument_spec)
    usdt_onhand_amount_init = await local_api.get_account()
    base_target_value = max(100.0, usdt_onhand_amount_init)

    latest_price = await fetch_last_price(symbol, category, mode)
    lot_size = compute_lot_size(latest_price, base_target_value, instrument_spec)
    position_value = lot_size * latest_price if latest_price else None
    effective_margin = (
        position_value / leverage_factor if (position_value is not None and leverage_factor > 0) else None
    )

    # if latest_price:
    #     discord.print_log(
    #         f"[BYBIT] {symbol} 最新価格 {latest_price:.4f} -> ロット {lot_size:.2f}, 想定ポジション {position_value:.2f} USDT"
    #     )
    #     if effective_margin is not None:
    #         discord.print_log(
    #             f"[BYBIT] レバレッジ {leverage_factor:.0f} 倍時の必要証拠金: {effective_margin:.2f} USDT"
    #         )
    # else:
    #     discord.print_log(
    #         f"[BYBIT] {symbol} の価格を取得できなかったため、最小仕様ロット {lot_size} を暫定使用します。"
    #     )

    # === Bitget側の仕様・価格・ロット計算 ===
    bg_instrument_spec = fetch_instrument_spec_bitget(symbol, 'USDT-FUTURES', 'live') or DEFAULT_INSTRUMENT_SPEC.copy()
    discord.print_log(
        f"[BITGET] {symbol} 仕様確認: qty_step={bg_instrument_spec['qty_step']} min_qty={bg_instrument_spec['min_qty']} "
        f"max_qty={bg_instrument_spec['max_qty']} max_leverage={bg_instrument_spec['max_leverage']}"
    )

    bg_leverage_cap = bg_instrument_spec.get("max_leverage", LEVERAGE_FACTOR) or LEVERAGE_FACTOR
    bg_requested_leverage = bg_leverage_cap * LEVERAGE_USAGE_RATIO
    bg_leverage_factor = max(1.0, min(LEVERAGE_FACTOR, bg_requested_leverage))
    discord.print_log(
        f"[BITGET] Leverage cap {bg_leverage_cap}x -> using {bg_leverage_factor:.2f}x (ratio {LEVERAGE_USAGE_RATIO:.2f})"
    )

    bg_latest_price = await fetch_last_price_bitget(symbol, 'USDT-FUTURES', 'live')
    base_bg_target_value = base_target_value * 0.5
    bg_lot_size = compute_bitget_lot_size(bg_latest_price, base_bg_target_value, bg_instrument_spec) if bg_latest_price else bg_instrument_spec.get("min_qty", 0.01)
    bg_position_value = bg_lot_size * bg_latest_price if bg_latest_price else None
    bg_effective_margin = (
        bg_position_value / bg_leverage_factor if (bg_position_value is not None and bg_leverage_factor > 0) else None
    )

    if bg_latest_price:
        discord.print_log(
            f"[BITGET] {symbol} 最新価格 {bg_latest_price:.4f} -> ロット {bg_lot_size:.2f}, 想定ポジション {bg_position_value:.2f} USDT"
        )
        if bg_effective_margin is not None:
            discord.print_log(
                f"[BITGET] レバレッジ {bg_leverage_factor:.0f} 倍時の必要証拠金: {bg_effective_margin:.2f} USDT"
            )
    else:
        discord.print_log(
            f"[BITGET] {symbol} の価格を取得できなかったため、最小仕様ロット {bg_lot_size} を暫定使用します。"
        )



    if interval not in interval_list:
        print("Invalid interval. Choose from {}".format(interval_list))
        sys.exit()
    interval_int = interval_map.get(interval)
    if not pnl_symbols:
        pnl_symbols = build_pnl_symbol_pool(None, symbol)
    # Flush setup logs buffer to separate initialization from trade start
    discord.flush_all()
    
    if mode == 'demo':
        discord.print_log("\n=============================================")
        discord.print_log("🚀【DEMO 自動トレードスタート】🚀")
        discord.print_log("demo---------demo---------demo---------demo--------demo")
        discord.print_log(f"bitget5_44_1 futures limit dual {symbol} demo started!")
    else:
        discord.print_log("\n=============================================")
        discord.print_log("⚡【REAL 自動トレードスタート】⚡")
        discord.print_log("$$$----------$$$----------$$$----------$$$----------$$$")
        discord.print_log(f"bitget5_44_1 futures limit dual {symbol} real started!")
    discord.print_log(f"symbol : {symbol}, max_lot : {max_lot}, lot_size : {lot_size}")
    discord.print_log(f"Target amount : {target_amount} interval : {interval} interval_int : {interval_int}")
    discord.flush_all()
    # await asyncio.sleep(1)
    usdt_onhand_amount = 0.0
    onhand_amount = 0.0
    bitget_onhand_amount = 0.0
    bitget_target_amount = 100.0  # デフォルト（50 USDT の 2倍）

    async def initialize_trading_state() -> None:
        nonlocal df, usdt_onhand_amount, onhand_amount, best_mp, best_er, best_atr, best_interval, bitget_onhand_amount, bitget_target_amount
        global trade_paused, target_amount
        local_api = api_bitget(symbol, category, coin, mode, instrument_spec=instrument_spec)
        usdt_onhand_amount = await local_api.get_account()
        await asyncio.sleep(1)
        df = pd.DataFrame()
        df = await local_api.get_candle(df, interval, interval_int)
        await asyncio.sleep(1)
        
        # ローソク足データ取得失敗チェック
        if df is None or df.empty or len(df) == 0:
            discord.print_log("初期化エラー: ローソク足データが取得できませんでした。APIキーとネットワークを確認してください。")
            return
        
        # 未確定の最新足を除外（確定した足のみでインジケータ等を計算）
        df = df.iloc[:-1].reset_index(drop=True)
        df = logic.make_logic(df, strategy_type=current_strategy_type)
        await asyncio.sleep(1)
        position = await local_api.get_positions(lot_size=lot_size)
        if position is None:
            discord.print_log("Failed to fetch position data. Retrying.")
            onhand_amount = 0
        else:
            onhand_amount = position['sell'] if trade_side == 'short' else position['buy']
        await asyncio.sleep(1)
        
        discord.print_log(f"★ 事前計算済みのパラメーターを採用します: 戦略={symbol} {current_strategy_type.upper()} {trade_side.upper()}, interval={best_interval}, MP={best_mp}, ER={best_er}, Margin={best_margin}% (trade_paused={trade_paused})")
        
        # ベスト設定でmake_logicを再計算（時間足 + MP期間を反映）
        df_best = resample_candles(df, best_interval)
        df_best = logic.make_logic(df_best, market_profile_period=best_mp, strategy_type=current_strategy_type)
        
        # ベスト設定でバックテスト実行（ライブトレード用データ生成）
        bt = backtester()
        df_best = bt.run_backtest(
            df=df_best, lot=lot_size, data_equity=usdt_onhand_amount,
            side_mode=trade_side, mp_period=best_mp,
            er_threshold=best_er, strategy_type=current_strategy_type, sl_margin_pct=best_margin
        )

        # ★ バックテストPnLゲート: プラスにならない場合はエントリー無効化
        if not df_best.empty and 'pnl' in df_best.columns:
            final_pnl = df_best['pnl'].iloc[-1]
            discord.print_log(f"★ バックテストPnL: {final_pnl:.2f} USDT (初期: {usdt_onhand_amount:.2f})")
            if final_pnl <= usdt_onhand_amount:
                discord.print_log(f"バックテストPnLがプラスにならないため、エントリーを無効化します。(PnL={final_pnl:.2f} ≤ 初期={usdt_onhand_amount:.2f})")
                trade_paused = True
            else:
                discord.print_log(f"バックテストPnLがプラス (+{final_pnl - usdt_onhand_amount:.2f}) → トレード許可")

        await asyncio.sleep(1)
        usdt_onhand_amount = await local_api.get_account()
        
        # Bitget 用の残高から目標ポジションサイズと目標停止額（2倍）を設定
        if usdt_onhand_amount > 0:
            target_val = max(10.0, float(int(usdt_onhand_amount / 10.0) * 10))
            import bitget5_44_2api_dual
            bitget5_44_2api_dual.BITGET_TARGET_POSITION_VALUE_USDT = target_val * TARGET_LEVERAGE
            target_amount = usdt_onhand_amount * 2.0
        else:
            target_val = 10.0
            import bitget5_44_2api_dual
            bitget5_44_2api_dual.BITGET_TARGET_POSITION_VALUE_USDT = 10.0 * TARGET_LEVERAGE
            target_amount = 100.0
            
        global TARGET_POSITION_VALUE_USDT
        TARGET_POSITION_VALUE_USDT = bitget5_44_2api_dual.BITGET_TARGET_POSITION_VALUE_USDT

        bg_mode_str = "LIVE" if bitget5_44_2api_dual.bitget_mode == 'live' else "DEMO"
        discord.print_log(
            f"[BITGET {bg_mode_str}] Initial balance: {usdt_onhand_amount:.2f} USDT -> Target: {target_amount:.2f} USDT (Position Target: {TARGET_POSITION_VALUE_USDT:.2f} USDT)"
        )
        
        # データが取得できなかった場合はログ・グラフ出力をスキップ
        if df_best is None or df_best.empty:
            discord.print_log("初期化エラー: ローソク足データが取得できませんでした。APIキーとネットワークを確認してください。")
            return
        
        discord.print_logs(df_best, usdt_onhand_amount, onhand_amount, max_lot, TARGET_POSITION_VALUE_USDT, trade_side=trade_side)
        # await asyncio.sleep(1)
        df_best.to_csv("./backtest_data/klines100.csv")
        # await asyncio.sleep(1)
        # discord.plot_kline(symbol=symbol)
        # await asyncio.sleep(1)
        # discord.plot_backtest(label=f"MP{best_mp}_ER{best_er}", symbol=symbol)
        # await asyncio.sleep(2)

        # --- Bitget Startup Backtest Chart ---
        # try:
        #     startup_target_val = 50.0
        #     if 'target_val' in locals():
        #         startup_target_val = target_val
        #     await generate_bitget_backtest_chart(
        #         symbol=symbol,
        #         interval=interval,
        #         df_loop=df_best,
        #         best_mp=best_mp,
        #         best_margin=best_margin,
        #         best_er=best_er,
        #         trade_side=trade_side,
        #         api=local_api,
        #         current_bg_target_value=startup_target_val,
        #         bybit_exec_history=bt.exec_history,
        #         bybit_lot_size=lot_size
        #     )
        # except Exception as e:
        #     print(f"Startup Bitget backtest chart generation failed: {e}")
        #
        # await asyncio.sleep(2)
        await plot_pnl()
        # await asyncio.sleep(1)

    async def apply_symbol_state(target_symbol: str, plan: Optional[Dict[str, Any]], side: str) -> None:
        nonlocal lot_size
        global symbol, trade_side, instrument_spec, pnl_symbols, trade_paused
        symbol = target_symbol
        trade_side = side
        trade_paused = False
        remember_symbols([symbol])
        pnl_symbols = build_pnl_symbol_pool(plan, symbol)
        instrument_spec = (
            await fetch_instrument_spec_async(symbol, category, mode)
        ) or DEFAULT_INSTRUMENT_SPEC.copy()
        # discord.print_log(
        #     f"[BYBIT] {symbol} spec qty_step={instrument_spec['qty_step']} min_qty={instrument_spec['min_qty']} "
        #     f"max_qty={instrument_spec['max_qty']} max_leverage={instrument_spec['max_leverage']}"
        # )

        leverage_cap = instrument_spec.get("max_leverage", LEVERAGE_FACTOR) or LEVERAGE_FACTOR
        requested_leverage = leverage_cap * LEVERAGE_USAGE_RATIO
        leverage_factor = max(1.0, min(LEVERAGE_FACTOR, requested_leverage))
        # discord.print_log(
        #     f"[BYBIT] Leverage cap {leverage_cap}x -> using {leverage_factor:.2f}x (ratio {LEVERAGE_USAGE_RATIO:.2f})"
        # )

        base_target_value = max(100.0, usdt_onhand_amount)
        latest_price = await fetch_last_price(symbol, category, mode)
        lot_size = compute_lot_size(latest_price, base_target_value, instrument_spec)
        position_value = lot_size * latest_price if latest_price else None
        effective_margin = (
            position_value / leverage_factor if (position_value is not None and leverage_factor > 0) else None
        )

        # if latest_price:
        #     discord.print_log(
        #         f"[BYBIT] {symbol} price {latest_price:.4f} -> lot {lot_size:.2f}, est position {position_value:.2f} USDT"
        #     )
        #     if effective_margin is not None:
        #         discord.print_log(f"[BYBIT] Margin at {leverage_factor:.0f}x leverage = {effective_margin:.2f} USDT")
        # else:
        #     discord.print_log(f"[BYBIT] Failed to fetch price for {symbol}; using fallback lot {lot_size}.")

        # Bitget spec & calculation
        bg_instrument_spec = (
            await asyncio.to_thread(fetch_instrument_spec_bitget, symbol, 'USDT-FUTURES', 'live')
        ) or DEFAULT_INSTRUMENT_SPEC.copy()
        discord.print_log(
            f"[BITGET] {symbol} spec qty_step={bg_instrument_spec['qty_step']} min_qty={bg_instrument_spec['min_qty']} "
            f"max_qty={bg_instrument_spec['max_qty']} max_leverage={bg_instrument_spec['max_leverage']}"
        )

        bg_leverage_cap = bg_instrument_spec.get("max_leverage", LEVERAGE_FACTOR) or LEVERAGE_FACTOR
        bg_requested_leverage = bg_leverage_cap * LEVERAGE_USAGE_RATIO
        bg_leverage_factor = max(1.0, min(LEVERAGE_FACTOR, bg_requested_leverage))
        discord.print_log(
            f"[BITGET] Leverage cap {bg_leverage_cap}x -> using {bg_leverage_factor:.2f}x (ratio {LEVERAGE_USAGE_RATIO:.2f})"
        )

        bg_latest_price = await fetch_last_price_bitget(symbol, 'USDT-FUTURES', 'live')
        base_bg_target_value = base_target_value * 0.5
        bg_lot_size = compute_bitget_lot_size(bg_latest_price, base_bg_target_value, bg_instrument_spec) if bg_latest_price else bg_instrument_spec.get("min_qty", 0.01)
        bg_position_value = bg_lot_size * bg_latest_price if bg_latest_price else None
        bg_effective_margin = (
            bg_position_value / bg_leverage_factor if (bg_position_value is not None and bg_leverage_factor > 0) else None
        )

        if bg_latest_price:
            discord.print_log(
                f"[BITGET] {symbol} price {bg_latest_price:.4f} -> lot {bg_lot_size:.2f}, est position {bg_position_value:.2f} USDT"
            )
            if bg_effective_margin is not None:
                discord.print_log(f"[BITGET] Margin at {bg_leverage_factor:.0f}x leverage = {bg_effective_margin:.2f} USDT")
        else:
            discord.print_log(f"[BITGET] Failed to fetch price for {symbol}; using fallback lot {bg_lot_size}.")

        await initialize_trading_state()

    paused_before_backtest = trade_paused
    await initialize_trading_state()

    now_jst = datetime.now(JST)
    next_mix_analysis_at = now_jst.replace(
        hour=DAILY_ANALYSIS_HOUR, minute=DAILY_ANALYSIS_MINUTE, second=0, microsecond=0
    ) + timedelta(days=1)
    # 実際には当日まだ分析時刻前なら当日に実行する
    today_analysis_time = now_jst.replace(
        hour=DAILY_ANALYSIS_HOUR, minute=DAILY_ANALYSIS_MINUTE, second=0, microsecond=0
    )
    if now_jst < today_analysis_time:
        next_mix_analysis_at = today_analysis_time
    pending_flatten: bool = False
    pending_flatten_reason: str = ""
    symbol_switch_plan: Optional[Dict[str, Any]] = None
    last_flatten_attempt_at: Optional[datetime] = None
    paused_force_close_at: Optional[datetime] = None  # Force market close time when trade_paused

    # Startup cleanup: flatten all symbols if any position remains.
    startup_symbols = await fetch_all_position_symbols(category, coin, mode)
    if startup_symbols:
        discord.print_log(
            f"Startup: existing positions detected for {', '.join(startup_symbols)}. "
            f"Top3 candidates for limit flatten: {', '.join(startup_top_symbols) if startup_top_symbols else '[none]'}."
        )
        for sym in startup_symbols:
            force_market_flatten = sym not in startup_top_symbols
            reason = f"Startup flatten [{'market' if force_market_flatten else 'limit'}] {sym}"
            success = await flatten_current_position(
                category,
                sym,
                coin,
                mode,
                reason,
                TAKE_PROFIT_PCT_FOR_FLATTEN,
                force_market=force_market_flatten,
            )
            if not success:
                pending_flatten = True
                pending_flatten_reason = reason
                discord.print_log(f"{reason}: pending retry in main loop.")
            else:
                discord.print_log(f"{reason}: completed.")
        if pending_flatten:
            last_flatten_attempt_at = None
        else:
            discord.print_log("Startup flatten: all symbols cleared.")

    if trade_paused and paused_before_backtest:
        # 銘柄選定失敗による中止 → flatten実行
        discord.print_log("Starting in paused mode (no valid daily selection). Waiting for next analysis window.")
        # Set force market close time for startup paused mode
        paused_force_close_at = datetime.now(JST) + timedelta(hours=PAUSED_FORCE_MARKET_DELAY_HOURS)
        discord.print_log(f"Force market close at {paused_force_close_at.strftime('%H:%M')} if positions remain.")
        if not pending_flatten:
            pending_flatten = True
            pending_flatten_reason = "No valid selection; flattening positions while paused."
    elif trade_paused:
        # PnL不足による中止 → クローズロジックのみ（flattenなし）
        discord.print_log("Starting with entry disabled (backtest PnL ≤ 100). Close logic only.")

    while True:
        now_jst = datetime.now(JST)
        dt_now = datetime.now()
        flatten_ready = True
        force_market_flatten = False
        if symbol_switch_plan:
            flatten_ready = now_jst >= symbol_switch_plan["flatten_start"]
            force_market_flatten = now_jst >= symbol_switch_plan["force_close_at"]
            if flatten_ready and not pending_flatten:
                pending_flatten = True
                pending_flatten_reason = symbol_switch_plan.get("reason", "Symbol switch flatten")
                last_flatten_attempt_at = None
        # Check force market close for paused mode (no trade today)
        if trade_paused and pending_flatten and paused_force_close_at:
            if now_jst >= paused_force_close_at:
                force_market_flatten = True
                discord.print_log(f"Paused mode: force market close triggered (past {paused_force_close_at.strftime('%H:%M')}).")
        if now_jst >= next_mix_analysis_at:
            discord.print_log("Running daily mix analysis.")
            trade_plan = run_mix_analysis(mode)
            trade_paused = not bool(trade_plan and trade_plan.get("trade_symbol"))
            if trade_paused:
                pending_flatten = True
                pending_flatten_reason = "Daily analysis unavailable: flattening positions until next run."
                # Set force market close time for paused mode
                paused_force_close_at = now_jst + timedelta(hours=PAUSED_FORCE_MARKET_DELAY_HOURS)
                discord.print_log(
                    f"Daily analysis: top score unavailable/negative -> pause trading until next run. "
                    f"Force market close at {paused_force_close_at.strftime('%H:%M')}."
                )
                next_mix_analysis_at += timedelta(days=1)
                continue
            selected_symbol = trade_plan.get("trade_symbol")
            trade_side_plan = trade_plan.get("trade_side", "long")
            trade_label = trade_plan.get("trade_label")
            if selected_symbol:
                trade_paused = False
                
                # --- Update daily candidates ---
                trade_side_new = str(trade_side_plan).lower()
                all_new_candidates = select_candidates_from_scores(trade_side_new, limit=15)
                
                if all_new_candidates:
                    base_symbol = all_new_candidates[0]
                    profitable_cands, best_strategy, best_interval, best_mp, best_er, best_margin = await validate_profitable_candidates(
                        trade_side_new, mode, base_symbol, interval, interval_int
                    )
                    current_strategy_type = best_strategy
                    new_candidates = profitable_cands
                    discord.print_log(f"Daily analysis: New validated candidates: {', '.join(new_candidates)}")
                else:
                    new_candidates = []
                
                if not new_candidates:
                    trade_paused = True
                    pending_flatten = True
                    pending_flatten_reason = "Daily analysis returned no profitable symbol."
                    paused_force_close_at = now_jst + timedelta(hours=PAUSED_FORCE_MARKET_DELAY_HOURS)
                    discord.print_log(
                        f"Daily analysis returned no profitable symbol after validation. Pausing trading. "
                        f"Force market close at {paused_force_close_at.strftime('%H:%M')}."
                    )
                    next_mix_analysis_at += timedelta(days=1)
                    continue

                # Check if our current symbol is still in the new candidates and has the same trade side
                keep_current = False
                if symbol in new_candidates and trade_side_new == trade_side:
                    # Check if we actually have an active position in the current symbol
                    current_positions = await fetch_all_position_symbols(category, coin, mode)
                    if symbol in current_positions:
                        keep_current = True
                        
                if keep_current:
                    new_symbol = symbol
                    startup_top_symbols = new_candidates
                    discord.print_log(f"Daily analysis: Keeping current symbol {symbol} as it is in the new candidates and positioned.")
                else:
                    new_symbol = new_candidates[0]
                    startup_top_symbols = new_candidates
                    
                if new_symbol != symbol:
                    flatten_start = now_jst.replace(hour=SYMBOL_SWITCH_FLATTEN_START_HOUR, minute=0, second=0, microsecond=0)
                    if now_jst >= flatten_start:
                        flatten_start = now_jst
                    force_close_at = now_jst.replace(hour=SYMBOL_SWITCH_FORCE_CLOSE_HOUR, minute=0, second=0, microsecond=0)
                    if force_close_at <= flatten_start:
                        force_close_at = flatten_start + timedelta(hours=2)
                    symbol_switch_plan = {
                        "new_symbol": new_symbol,
                        "trade_side": trade_side_plan,
                        "trade_plan": trade_plan,
                        "flatten_start": flatten_start,
                        "force_close_at": force_close_at,
                        "reason": "Symbol switch flatten",
                    }
                    if flatten_ready and now_jst >= flatten_start:
                        pending_flatten = True
                        pending_flatten_reason = symbol_switch_plan["reason"]
                        last_flatten_attempt_at = None
                    discord.print_log(
                        f"Daily analysis: switching symbol to {new_symbol} after cleanup start "
                        f"{flatten_start.strftime('%H:%M')}, force close {force_close_at.strftime('%H:%M')}."
                    )
                else:
                    symbol_switch_plan = None
                    await apply_symbol_state(new_symbol, trade_plan, trade_side_plan)
                    discord.print_log(f"Daily analysis: keeping symbol {new_symbol}.")
            else:
                trade_paused = True
                pending_flatten = True
                pending_flatten_reason = "Daily analysis returned no tradable symbol (score negative?)."
                # Set force market close time for paused mode
                paused_force_close_at = now_jst + timedelta(hours=PAUSED_FORCE_MARKET_DELAY_HOURS)
                discord.print_log(
                    f"Daily analysis returned no tradable symbol (score negative?). Pausing trading. "
                    f"Force market close at {paused_force_close_at.strftime('%H:%M')}."
                )
                next_mix_analysis_at += timedelta(days=1)
                continue
            if trade_label:
                discord.print_log(trade_label)
            next_mix_analysis_at += timedelta(days=1)
            continue
        flatten_window_open = flatten_ready
        flatten_due = pending_flatten and flatten_window_open and (
            last_flatten_attempt_at is None
            or (now_jst - last_flatten_attempt_at).total_seconds() >= FLATTEN_REISSUE_MINUTES * 60
            or force_market_flatten
        )
        if flatten_due:
            reason = pending_flatten_reason or "Pending flatten"
            success = await flatten_all_positions(
                category,
                coin,
                mode,
                reason,
                TAKE_PROFIT_PCT_FOR_FLATTEN,
                force_market=force_market_flatten,
            )
            pending_flatten = not success
            last_flatten_attempt_at = now_jst
            if success and symbol_switch_plan:
                await apply_symbol_state(
                    symbol_switch_plan["new_symbol"],
                    symbol_switch_plan.get("trade_plan"),
                    symbol_switch_plan.get("trade_side", trade_side),
                )
                symbol_switch_plan = None
                pending_flatten_reason = ""
                last_flatten_attempt_at = None
                continue
        if symbol_switch_plan and not pending_flatten and flatten_ready:
            await apply_symbol_state(
                symbol_switch_plan["new_symbol"],
                symbol_switch_plan.get("trade_plan"),
                symbol_switch_plan.get("trade_side", trade_side),
            )
            symbol_switch_plan = None
            pending_flatten_reason = ""
            last_flatten_attempt_at = None
            continue
        if dt_now.minute % interval_int == 0:
            await asyncio.sleep(1)
            
            # --- Portfolio rotation logic: Check positions and scan signals ---
            active_cand = None
            try:
                current_positions = await fetch_all_position_symbols(category, coin, mode)
                # Check if the currently selected symbol has an active position
                if symbol in current_positions:
                    active_cand = symbol
                else:
                    # Otherwise, check if any of the other top candidates have a position
                    for cand in startup_top_symbols:
                        if cand in current_positions:
                            active_cand = cand
                            break
            except Exception as e:
                discord.print_log(f"Error fetching current positions: {e}")
                
            if active_cand:
                # Case A: Position active on a candidate symbol (or current symbol)
                if active_cand != symbol:
                    discord.print_log(f"Position detected on candidate {active_cand}. Switching active symbol from {symbol} to {active_cand}.")
                    await apply_symbol_state(active_cand, None, trade_side)
                trade_paused = False
            else:
                # Case B: Flat portfolio. Scan candidates in priority order.
                found_signal_symbol = None
                for cand in startup_top_symbols:
                    # print(f"Scanning candidate {cand} for breakout signal...")
                    try:
                        cand_spec = (await fetch_instrument_spec_async(cand, category, mode)) or DEFAULT_INSTRUMENT_SPEC.copy()
                        cand_api = api_bitget(cand, category, coin, mode, instrument_spec=cand_spec)
                        
                        cand_df = pd.DataFrame()
                        cand_df = await cand_api.get_candle(cand_df, interval, interval_int)
                        if cand_df is None or cand_df.empty or len(cand_df) <= 1:
                            # print(f"Failed to fetch candles for candidate {cand}, skipping scan.")
                            continue
                        
                        cand_df = cand_df.iloc[:-1].reset_index(drop=True)
                        cand_df_loop = resample_candles(cand_df, best_interval)
                        cand_df_loop = logic.make_logic(cand_df_loop, market_profile_period=best_mp, er_threshold=best_er, strategy_type=current_strategy_type)
                        
                        if cand_df_loop.empty:
                            continue
                        
                        long_sig = bool(cand_df_loop['long'].iloc[-1])
                        short_sig = bool(cand_df_loop['short'].iloc[-1])
                        
                        if (trade_side == "long" and long_sig) or (trade_side == "short" and short_sig):
                            discord.print_log(f"Breakout signal detected on candidate {cand}! (long={long_sig}, short={short_sig})")
                            found_signal_symbol = cand
                            break
                    except Exception as e:
                        discord.print_log(f"Error scanning candidate {cand}: {e}")
                        continue
                
                if found_signal_symbol:
                    discord.print_log(f"Switching active symbol to signal candidate {found_signal_symbol}.")
                    await apply_symbol_state(found_signal_symbol, None, trade_side)
                    trade_paused = False
                else:
                    # No active positions and no new signals: pause entries.
                    trade_paused = True

            api = api_bitget(symbol, category, coin, mode, instrument_spec=instrument_spec)
            exit_reason: Optional[str] = None
            action_messages: List[str] = []
            evaluated = False
            longclose_signal = False
            shortclose_signal = False
            fade_signal = False
            force_flatten_signal = pending_flatten
            fade_window_active = 7 <= now_jst.hour < 9
            loop_trade_paused = False


            # 待機モード: 1時間ごとにバックテスト+利確ロジックのみ実行
            if trade_paused and not pending_flatten:
                # 1時間ごと（毎正時）以外はスキップ
                if now_jst.minute != 0:
                    await asyncio.sleep(5)
                    continue
                action_messages.append("待機モード: 1時間チェック (エントリー無効、利確のみ有効)")

            open_orders = await api.get_open_orders()
            reduce_only_orders = [o for o in open_orders if o.get("_is_reduce_only")]
            non_reduce_orders = [o for o in open_orders if not o.get("_is_reduce_only")]
            position = await api.get_positions(lot_size=lot_size)
            if position is None:
                position = {
                    "buy": 0.0,
                    "sell": 0.0,
                    "buy_pos": 0.0,
                    "sell_pos": 0.0,
                    "profit": 0.0,
                    "pos_count": 0.0,
                    "buy_count": 0,
                    "sell_count": 0,
                }
                action_messages.append("Position fetch failed; retrying later.")
                exit_reason = decide_exit_reason(
                    {
                        "force_flatten": force_flatten_signal,
                        "fade_take_profit": fade_signal,
                        "longclose_signal": longclose_signal,
                        "shortclose_signal": shortclose_signal,
                    }
                )
                log_state(
                    format_state_message(symbol, position, len(open_orders), pending_flatten, trade_side)
                )
                log_decision(
                    format_decision_message(exit_reason, evaluated, longclose_signal, shortclose_signal, fade_signal, now_jst.strftime('%H:%M'), position.get('profit', 0.0), trade_side)
                )
                log_action("; ".join(action_messages))
                await asyncio.sleep(5)
                continue

            open_orders_exist = bool(open_orders)

            if non_reduce_orders:
                discord.print_log(
                    f"Entry gated: active non-reduce orders exist ({len(non_reduce_orders)})"
                )
                profit_snapshot = float(position.get("profit", 0.0))
                cancel_success = await api.active_order_cancel()
                if cancel_success:
                    action_messages.append("Canceled all active orders (non-reduce present).")
                else:
                    action_messages.append("Failed to cancel non-reduce orders; will retry.")
                await asyncio.sleep(1)
                open_orders = await api.get_open_orders()
                reduce_only_orders = [o for o in open_orders if o.get("_is_reduce_only")]
                non_reduce_orders = [o for o in open_orders if not o.get("_is_reduce_only")]
                if open_orders:
                    sample = [
                        f"{o.get('orderType','?')}/{o.get('side','?')}@{o.get('price','?')} reduceOnly={o.get('reduceOnly')}"
                        for o in open_orders[:3]
                    ]
                    action_messages.append(f"After cancel: open_orders={len(open_orders)} sample={'; '.join(sample)}")
                if profit_snapshot > 0 and (position.get("buy", 0) > 0 or position.get("sell", 0) > 0):
                    flatten_reason = "Non-reduce order cleanup (best bid/ask)"
                    flatten_success = await flatten_current_position(
                        category, symbol, coin, mode, flatten_reason, TAKE_PROFIT_PCT_FOR_FLATTEN
                    )
                    if flatten_success:
                        action_messages.append("Cleanup flatten completed for non-reduce orders.")
                        force_flatten_signal = True
                        open_orders = await api.get_open_orders()
                        reduce_only_orders = [o for o in open_orders if o.get("_is_reduce_only")]
                        non_reduce_orders = [o for o in open_orders if not o.get("_is_reduce_only")]
                    else:
                        action_messages.append("Cleanup flatten attempt failed; continuing trade loop.")
                if non_reduce_orders:
                    action_messages.append(f"Non-reduce orders remain ({len(non_reduce_orders)}); entry gated.")
                await asyncio.sleep(1)

            open_orders_exist = bool(open_orders)
            no_pos = (position.get("buy", 0) == 0) and (position.get("sell", 0) == 0)
            no_orders = not open_orders_exist
            if pending_flatten and no_pos and no_orders:
                pending_flatten = False
                action_messages.append(
                    f"Flatten completed for {symbol}: positions=0 and open_orders=0. Resume entries."
                )
            elif pending_flatten and no_pos and open_orders_exist:
                # Only reduce-only orders left without position: try to cancel once and re-evaluate.
                cancel_ok = await api.active_order_cancel()
                await asyncio.sleep(1)
                open_orders = await api.get_open_orders()
                open_orders_exist = bool(open_orders)
                if not open_orders_exist:
                    pending_flatten = False
                    action_messages.append(
                        f"Flatten completed after cancel for {symbol}: positions=0 and open_orders=0."
                    )
                else:
                    action_messages.append(
                        f"Flatten pending: positions=0 but {len(open_orders)} open orders remain after cancel attempt."
                    )

            if pending_flatten:
                force_flatten_signal = True
                exit_reason = decide_exit_reason(
                    {
                        "force_flatten": force_flatten_signal,
                        "fade_take_profit": fade_signal,
                        "longclose_signal": longclose_signal,
                        "shortclose_signal": shortclose_signal,
                    }
                )
                log_state(
                    format_state_message(symbol, position, len(open_orders), pending_flatten, trade_side)
                )
                log_decision(
                    format_decision_message(exit_reason, evaluated, longclose_signal, shortclose_signal, fade_signal, now_jst.strftime('%H:%M'), position.get('profit', 0.0), trade_side)
                )
                action_messages.append(
                    f"Pending flatten for {symbol}: positions buy={position['buy']} sell={position['sell']}, "
                    f"open_orders={len(open_orders)}, profit={position.get('profit', 0.0)}"
                )
                log_action("; ".join(action_messages))
                await asyncio.sleep(5)
                continue

            await asyncio.sleep(1)
            df = await api.get_candle(df, interval, interval_int)
            await asyncio.sleep(1)
            # 未確定の最新足を除外（確定した足のみでインジケータ等を計算）
            if df is not None and len(df) > 1:
                df = df.iloc[:-1].reset_index(drop=True)
            df_loop = resample_candles(df, best_interval)
            df_loop = logic.make_logic(df_loop, market_profile_period=best_mp, strategy_type=current_strategy_type)
            await asyncio.sleep(1)
            onhand_amount = position['sell'] if trade_side == 'short' else position['buy']
            
            # BTCフィルターに基づく動的ロット倍率の取得と適用
            lot_multiplier, btc_reason = await get_btc_lot_multiplier(mode, trade_side)
            
            # Bybit側のターゲットポジション価値（証拠金切り捨てのTARGET_LEVERAGE倍＝レバレッジ10倍設定時に証拠金不足にならないよう調整）
            bybit_margin = max(10.0, float(int(usdt_onhand_amount / 10.0) * 10))
            base_target_value = bybit_margin * TARGET_LEVERAGE
            
            # Bitget側のターゲットポジション価値（証拠金切り捨てのTARGET_LEVERAGE倍＝レバレッジ10倍設定時に証拠金不足にならないよう調整）
            bg_margin_val = bitget_onhand_amount if 'bitget_onhand_amount' in locals() and bitget_onhand_amount > 0 else (usdt_onhand_amount * 0.5)
            bitget_margin = max(10.0, float(int(bg_margin_val / 10.0) * 10))
            base_bg_target_value = bitget_margin * TARGET_LEVERAGE
            
            current_target_value = base_target_value * lot_multiplier
            current_bg_target_value = base_bg_target_value * lot_multiplier
            
            latest_price = await fetch_last_price(symbol, category, mode)
            bg_latest_price = await fetch_last_price_bitget(symbol, 'USDT-FUTURES', 'live')
            
            actual_lot_size = compute_lot_size(latest_price, current_target_value, instrument_spec)
            
            # positionのbuy_count, sell_count, pos_countをlot_size（固定値）を基に再計算
            # ※ actual_lot_sizeは現在価格・残高で毎ループ変動するため、
            #    ポジション保有中のstage計算に使うとstageが誤って増幅する。
            #    lot_size（起動時に計算した固定値）で割ることで安定したstage判定を行う。
            if position is not None:
                buy_qty = position.get("buy", 0.0)
                sell_qty = position.get("sell", 0.0)
                stage_lot = lot_size if lot_size > 0 else actual_lot_size
                if stage_lot > 0:
                    buy_size = max(round(buy_qty / stage_lot), 0)
                    sell_size = max(round(sell_qty / stage_lot), 0)
                    position['buy_count'] = buy_size
                    position['sell_count'] = sell_size
                    from bitget5_44_3logic import PositionSizer
                    PositionSizer.build_cumulative_dict()
                    stage = PositionSizer.get_stage_from_position_size(max(buy_size, sell_size))
                    position['pos_count'] = float(stage)
                else:
                    position['buy_count'] = 0 if buy_qty == 0 else 1
                    position['sell_count'] = 0 if sell_qty == 0 else 1
                    position['pos_count'] = float(max(position['buy_count'], position['sell_count']))
            
            import bitget5_44_2api_dual
            bitget5_44_2api_dual.BITGET_TARGET_POSITION_VALUE_USDT = current_bg_target_value
            
            # エントリーサイズ判定用の一時無効化（BTCが下落していないときはエントリーしない）
            if lot_multiplier == 0.0:
                action_messages.append(f"Short entry gated: {btc_reason}")
                loop_trade_paused = True
            else:
                loop_trade_paused = False
                if lot_multiplier != 1.0:
                    action_messages.append(f"Dynamic lot active: multiplier={lot_multiplier}x ({btc_reason})")

            await asyncio.sleep(1)
            bt = backtester()
            df_loop = bt.run_backtest(df=df_loop, lot=actual_lot_size, data_equity=usdt_onhand_amount, side_mode=trade_side,
                                 mp_period=best_mp, er_threshold=best_er, strategy_type=current_strategy_type, sl_margin_pct=best_margin)
            # ★ 毎時バックテストPnLゲートの判定とDiscord通知
            if not df_loop.empty and 'pnl' in df_loop.columns:
                final_pnl = df_loop['pnl'].iloc[-1]
                discord.print_log(f"★ 毎時バックテストPnL: {final_pnl:.2f} USDT (初期残高: {usdt_onhand_amount:.2f})")
                if final_pnl <= usdt_onhand_amount:
                    discord.print_log(f"毎時バックテストPnLがプラスにならないため、エントリーを無効（一時停止）にします。(PnL={final_pnl:.2f} ≤ 残高={usdt_onhand_amount:.2f})")
                    trade_paused = True
                else:
                    if trade_paused:
                        discord.print_log(f"毎時バックテストPnLが改善しプラスになりました (+{final_pnl - usdt_onhand_amount:.2f}) → トレード再開")
                    trade_paused = False
            await asyncio.sleep(1)

            # Track entry candle timestamp for Stage 1 exit management
            global entry_candle_time
            has_pos = (position.get("buy", 0) > 0) or (position.get("sell", 0) > 0)
            is_new_candle = False
            if not has_pos:
                entry_candle_time = None
            elif not df_loop.empty:
                current_candle_time = df_loop['timestamp'].iloc[-1] if 'timestamp' in df_loop.columns else df_loop.index[-1]
                if entry_candle_time is None:
                    entry_candle_time = current_candle_time
                is_new_candle = (current_candle_time != entry_candle_time)

            profit_val = float(position.get("profit", 0.0))
            if not df_loop.empty:
                evaluated = True
                fade_signal = bool(
                    fade_window_active
                    and profit_val > 0
                    and ((position.get("buy", 0) > 0) or (position.get("sell", 0) > 0))
                )

            exit_reason = decide_exit_reason(
                {
                    "force_flatten": force_flatten_signal,
                    "fade_take_profit": fade_signal,
                }
            )

            log_state(
                format_state_message(symbol, position, len(open_orders), pending_flatten, trade_side)
            )
            log_decision(
                format_decision_message(exit_reason, evaluated, fade_signal, now_jst.strftime('%H:%M'), profit_val, trade_side)
            )

            if exit_reason == "force_flatten":
                usdt_onhand_amount = await api.get_account()
                action_messages.append("force_flatten active; skipped close/entry this loop")
            else:
                if trade_side == "long" and position.get("buy", 0) > 0:
                    trail_triggered = await api.long_close(df_loop, position, commission, sl_margin_pct=best_margin, strategy_type=current_strategy_type, is_new_candle=is_new_candle, entry_candle_time=entry_candle_time)
                    if trail_triggered:
                        action_messages.append("TrailingSL Triggered (LONG)")
                elif trade_side == "short" and position.get("sell", 0) > 0:
                    trail_triggered = await api.short_close(df_loop, position, commission, sl_margin_pct=best_margin, strategy_type=current_strategy_type, is_new_candle=is_new_candle, entry_candle_time=entry_candle_time)
                    if trail_triggered:
                        action_messages.append("TrailingSL Triggered (SHORT)")

                if exit_reason == "fade_take_profit":
                    action_messages.append("fade_take_profit active; observing without exit order")
                
                usdt_onhand_amount = await api.get_account()
                bitget_onhand_amount = usdt_onhand_amount
                await asyncio.sleep(1)
                
                if trade_paused or loop_trade_paused:
                    action_messages.append("待機モード: エントリー無効")
                else:
                    # 目標金額（2倍）に達したかを判定
                    target_reached_long = onhand_amount == 0 and usdt_onhand_amount > target_amount
                    target_reached_short = position.get("sell", 0) == 0 and usdt_onhand_amount > target_amount
                    
                    if trade_side == "long":
                        if target_reached_long:
                            action_messages.append(f"[BITGET] {usdt_onhand_amount:.2f}[USDT] target_amount reached - stop long entry")
                        else:
                            long_entered = await api.long_entry(df_loop, position, usdt_onhand_amount, actual_lot_size, max_lot)
                            if long_entered:
                                action_messages.append("Long Entry Placed")
                    
                    if trade_side == "short":
                        if target_reached_short:
                            action_messages.append(f"[BITGET] {usdt_onhand_amount:.2f}[USDT] target_amount reached - stop short entry")
                        else:
                            short_entered = await api.short_entry(df_loop, position, usdt_onhand_amount, actual_lot_size, max_lot)
                            if short_entered:
                                action_messages.append("Short Entry Placed")
                await asyncio.sleep(1)

            # 実際のトレードアクション（エントリー/決済等）がない待機モード通知のみの場合はACTIONログ送信を抑制
            real_actions = [m for m in action_messages if not m.startswith("待機モード")]
            if real_actions:
                action_msg_str = "; ".join(action_messages)
                log_action(action_msg_str)
            elif not action_messages:
                log_action("none")
            await asyncio.sleep(1)
        checktime = should_display_chart(dt_now, interval_int)
        if dt_now.minute % interval_int == 0 and checktime:
            try:
                await asyncio.sleep(10)
                df_loop.to_csv("./backtest_data/klines100.csv")
                await asyncio.sleep(1)
                
                # --- Best parameters Backtest Chart ---
                discord.plot_backtest(label=f"MP{best_mp}_ER{best_er}", symbol=symbol)
                await asyncio.sleep(2)

                # --- Bitget Backtest Chart (Using Bitget prices & Bybit signals) ---
                # await generate_bitget_backtest_chart(
                #     symbol=symbol,
                #     interval=interval,
                #     df_loop=df_loop,
                #     best_mp=best_mp,
                #     best_margin=best_margin,
                #     best_er=best_er,
                #     trade_side=trade_side,
                #     api=api,
                #     current_bg_target_value=current_bg_target_value,
                #     bybit_exec_history=bt.exec_history,
                #     bybit_lot_size=actual_lot_size
                # )
                #     
                # await asyncio.sleep(2)
                await plot_pnl()
            except Exception as e:
                discord.print_log(f'image plot error : {e}')
                print(f'Image plot error: {e}')
                pass
        next_minute = (dt_now + timedelta(minutes=1)).replace(second=10, microsecond=0)
        sleep_time = (next_minute - dt_now).total_seconds()
        await asyncio.sleep(max(0, sleep_time))


if __name__  == '__main__':
    # ==================== 【取引モード設定】 ====================
    # [1] Bitget ライブ環境（本番） or デモ環境
    #     True  = 本番取引 (Live)
    #     False = デモ取引 (Demo)
    BITGET_IS_LIVE = False
    
    # [2] Bitget エアーモード（発注監視のみ） or 通常取引（実際に注文）
    #     True  = エアーモード (注文は出さず、監視のみ実行)
    #     False = 通常取引 (実際にBitgetへ注文を送信)
    BITGET_IS_AIR = False
    # ============================================================

    mode = 'live' if BITGET_IS_LIVE else 'demo'
    
    import bitget5_44_2api_dual
    bitget5_44_2api_dual.bitget_mode = mode
    bitget5_44_2api_dual.is_air = BITGET_IS_AIR
    bitget5_44_2api_dual.BITGET_TARGET_POSITION_VALUE_USDT = 50.0  # 取引サイズ（起動時に残高に応じて自動的に再調整されます）
    
    SKIP_MIX_ANALYSIS = False  # 分析をスキップするかどうか（必要に応じて True に変更）
    interval = '60'  # 全て1時間足（60分足）で実行

    # === 取引規格・通貨設定 (Bitget専用) ===
    product_type = 'USDT-FUTURES'
    margin_coin = 'USDT'
    coin = margin_coin
    category = product_type
    amount = 100.0
    target_amount = 2 * amount
    default_symbol = 'BTCUSDT'
    symbol = default_symbol
    trade_side = 'long'

    start_msg = (
        "```\n"
        " __  __  ___  _____  _  ___ \n"
        "|  \\/  |/ _ \\|_   _|| |/ _ \\\n"
        "| |\\/| | (_) | | |  | | (_) |\n"
        "|_|  |_|\\___/  |_|  |_|\\___/ \n"
        "----------------------------\n"
        "  AUTO TRADING SYSTEM START \n"
        "----------------------------\n"
        "```\n"
        "[Bitget Auto Trading System]"
    )

    discord.print_log(start_msg)

    max_lot = 10
    commission = 0.0002  # bybit makerFeeRate: 0.0002

    asyncio.run(start(mode, max_lot, interval))

# source ~/pybot-env/bin/activate
# の後に実行する
