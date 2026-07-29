#!/usr/bin/env python
# coding: utf-8
"""
Dynamic Portfolio Walk-Forward Analysis (WFA) バックテストスクリプト

日次で銘柄選定（スコアリング・地合い判定によるローテーション）を行いながら、
IS (In-Sample) 期間で戦略パラメータを最適化し、
OOS (Out-of-Sample) 期間で実運用通り1日ごとに銘柄をチェンジしながら検証を行う。

使い方:
    python walk_forward_portfolio_backtest.py [--is-days 180] [--oos-days 60] [--step-days 60] [--top-n 3]
"""

import os
import sys
import time
import copy
import argparse
import datetime
from datetime import timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- パス設定 ---
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from bitget5_44_5backtest_mm import make_mm_pl, AirExchange
from bitget5_44_3logic import logicinstance, backtester, resample_candles

JST = timezone(timedelta(hours=9))

def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}")

# ============================================================
# データ読み込み
# ============================================================

def load_all_symbols_data(csv_path: Path) -> dict:
    """ヒストリカルCSVから銘柄ごとのDataFrame辞書を作成する"""
    log(f"Master CSVを読み込んでいます: {csv_path.name} ...")
    df_all = pd.read_csv(csv_path)
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True).dt.tz_convert(JST)
    df_all["timestamp"] = df_all["timestamp"].dt.tz_localize(None)
    df_all = df_all.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    symbols = sorted(df_all["symbol"].unique())
    dfs = {}
    for sym in symbols:
        df_sym = df_all[df_all["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
        # カラム名マッピング
        df_out = pd.DataFrame({
            "timestamp": df_sym["timestamp"],
            "open":   df_sym["bybit_open"].astype(float),
            "high":   df_sym["bybit_high"].astype(float),
            "low":    df_sym["bybit_low"].astype(float),
            "close":  df_sym["bybit_close"].astype(float),
            "volume": df_sym["bybit_volume"].astype(float),
            "bybit_open": df_sym["bybit_open"].astype(float),
            "bybit_high": df_sym["bybit_high"].astype(float),
            "bybit_low": df_sym["bybit_low"].astype(float),
            "bybit_close": df_sym["bybit_close"].astype(float),
            "bybit_volume": df_sym["bybit_volume"].astype(float),
        })
        if "bybit_openInterest" in df_sym.columns:
            df_out["bybit_openInterest"] = df_sym["bybit_openInterest"].astype(float)
        if "bybit_fundingRate" in df_sym.columns:
            df_out["bybit_fundingRate"] = df_sym["bybit_fundingRate"].astype(float)
        if "coinbase_premium" in df_sym.columns:
            df_out["coinbase_premium"] = df_sym["coinbase_premium"].astype(float)
        
        dfs[sym] = df_out

    min_ts = df_all["timestamp"].min()
    max_ts = df_all["timestamp"].max()
    log(f"  読み込み完了: {len(symbols)} 銘柄 ({min_ts.strftime('%Y-%m-%d')} ~ {max_ts.strftime('%Y-%m-%d')})")
    return dfs

# ============================================================
# 日次銘柄選定ロジック (generate_selection_scores 共通化)
# ============================================================

def get_zscore(series: pd.Series, invert: bool = False) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    z = (series - series.mean()) / std
    return -z if invert else z

def determine_market_state(mean_norm: float, mean_cb_premium: float) -> str:
    if mean_norm >= 1.0 or mean_cb_premium > 0.0:
        return "long_only"
    else:
        return "short_only"

def select_daily_symbols(dfs: dict, target_dt: datetime.datetime, top_n: int = 3) -> tuple:
    """指定日（毎朝11:00）時点のデータから地合いと選定銘柄リストを返す"""
    # A. 地合い判定 (5日ルックバック)
    window_start_temp = target_dt - timedelta(days=5)
    temp_metrics = []

    for symbol, df_sym in dfs.items():
        df_5d = df_sym[(df_sym["timestamp"] >= window_start_temp) & (df_sym["timestamp"] <= target_dt)]
        if len(df_5d) < 24 * 4:
            continue
        close_now = float(df_5d["close"].iloc[-1])
        close_5d_ago = float(df_5d["close"].iloc[0])
        norm_perf_5d = close_now / close_5d_ago if close_5d_ago > 0 else 1.0
        cb_prem = float(df_5d["coinbase_premium"].iloc[-1]) if "coinbase_premium" in df_5d.columns else 0.0
        temp_metrics.append({"symbol": symbol, "norm_perf_5d": norm_perf_5d, "coinbase_premium": cb_prem})

    if not temp_metrics:
        return "long_only", []

    df_temp = pd.DataFrame(temp_metrics)
    mean_norm = float(df_temp["norm_perf_5d"].mean())
    mean_cb_premium = float(df_temp["coinbase_premium"].mean())
    market_state = determine_market_state(mean_norm, mean_cb_premium)

    # B. 本番スコア計算
    window_days = 30
    window_start = target_dt - timedelta(days=window_days)

    btc_series = None
    if "BTCUSDT" in dfs:
        df_btc = dfs["BTCUSDT"]
        df_btc_w = df_btc[(df_btc["timestamp"] >= window_start) & (df_btc["timestamp"] <= target_dt)]
        if len(df_btc_w) >= 24 * (window_days - 1):
            btc_series = df_btc_w["close"]

    metrics_list = []
    for symbol, df_sym in dfs.items():
        df_w = df_sym[(df_sym["timestamp"] >= window_start) & (df_sym["timestamp"] <= target_dt)]
        if len(df_w) < 24 * (window_days - 1):
            continue
        close_series = df_w["close"]
        close_now = float(close_series.iloc[-1])
        close_start = float(close_series.iloc[0])
        norm_perf = close_now / close_start if close_start > 0 else 1.0

        norm_perf_btc = 1.0
        if btc_series is not None and len(close_series) == len(btc_series):
            alt_btc = close_series.values / btc_series.values
            if alt_btc[0] > 0:
                norm_perf_btc = alt_btc[-1] / alt_btc[0]

        diffs = close_series.diff().dropna()
        momentum = float((diffs > 0).mean()) if len(diffs) > 0 else 0.5

        rolling_max = close_series.cummax()
        drawdowns = (close_series / rolling_max - 1.0) * 100.0
        max_dd = float(drawdowns.min()) if not drawdowns.empty else 0.0

        vol_series = df_w["volume"].replace(0.0, np.nan).dropna()
        vol_change = 0.0
        if len(vol_series) >= 4:
            mid = len(vol_series) // 2
            v1, v2 = float(vol_series.iloc[:mid].mean()), float(vol_series.iloc[mid:].mean())
            if v1 > 0:
                vol_change = (v2 / v1 - 1.0) * 100.0

        oi_change = 0.0
        if "bybit_openInterest" in df_w.columns:
            oi_series = df_w["bybit_openInterest"].replace(0.0, np.nan).dropna()
            if len(oi_series) >= 2 and float(oi_series.iloc[0]) > 0:
                oi_change = (float(oi_series.iloc[-1]) / float(oi_series.iloc[0]) - 1.0) * 100.0

        cb_prem_avg = float(df_w["coinbase_premium"].mean()) if "coinbase_premium" in df_w.columns else 0.0
        daily_turnover = float((df_w["volume"] * df_w["close"]).dropna().mean() * 24)
        is_low_liq = 1 if daily_turnover < 5000000.0 else 0
        is_oversold = 1 if norm_perf <= 0.8 else 0
        ban_short = 1 if (is_low_liq or is_oversold) else 0

        metrics_list.append({
            "symbol": symbol,
            "norm_perf": norm_perf,
            "norm_perf_btc": norm_perf_btc,
            "momentum": momentum,
            "max_dd": max_dd,
            "vol_change": vol_change,
            "oi_change": oi_change,
            "cb_prem_avg": cb_prem_avg,
            "ban_short": ban_short,
        })

    if not metrics_list:
        return market_state, []

    df_m = pd.DataFrame(metrics_list)
    df_m["z_perf"] = get_zscore(df_m["norm_perf"])
    df_m["z_perf_btc"] = get_zscore(df_m["norm_perf_btc"])
    df_m["z_momentum"] = get_zscore(df_m["momentum"])
    df_m["z_dd"] = get_zscore(df_m["max_dd"].abs(), invert=True)
    df_m["z_vol"] = get_zscore(df_m["vol_change"])
    df_m["z_oi"] = get_zscore(df_m["oi_change"])
    df_m["z_cb"] = get_zscore(df_m["cb_prem_avg"])

    if market_state == "long_only":
        df_m["score"] = (1.0 * df_m["z_perf"] + 0.8 * df_m["z_perf_btc"] +
                         0.5 * df_m["z_momentum"] + 2.0 * df_m["z_dd"] +
                         0.5 * df_m["z_vol"] + 0.5 * df_m["z_oi"] + 0.5 * df_m["z_cb"])
    else:
        df_m["score"] = (1.5 * df_m["z_perf"] + 0.8 * df_m["z_perf_btc"] +
                         0.5 * df_m["z_momentum"] - 1.0 * df_m["z_vol"] + 0.5 * df_m["z_cb"])

    df_sorted = df_m.sort_values("score", ascending=False).reset_index(drop=True)

    selected = []
    if market_state == "long_only":
        for _, row in df_sorted.head(top_n * 2).iterrows():
            selected.append((row["symbol"], "long"))
            if len(selected) >= top_n:
                break
    else:
        df_asc = df_m.sort_values("score", ascending=True).reset_index(drop=True)
        for _, row in df_asc.iterrows():
            if row["ban_short"] == 1:
                continue
            selected.append((row["symbol"], "short"))
            if len(selected) >= top_n:
                break

    return market_state, selected

# ============================================================
# ウィンドウ分割
# ============================================================

def generate_dates(dfs: dict, start_days_offset: int = 21) -> list:
    """全データの中から毎朝11:00時点の日付リストを取得"""
    all_ts = []
    for sym, df in dfs.items():
        all_ts.extend(df["timestamp"].tolist())
    min_ts = min(all_ts)
    max_ts = max(all_ts)

    start_day = (min_ts + timedelta(days=start_days_offset)).replace(hour=11, minute=0, second=0, microsecond=0)
    end_day = max_ts.replace(hour=11, minute=0, second=0, microsecond=0)

    dates = []
    curr = start_day
    while curr <= end_day:
        dates.append(curr)
        curr += timedelta(days=1)
    return dates

def generate_windows(dates: list, is_days: int, oos_days: int, step_days: int):
    windows = []
    start_dt = dates[0]
    max_dt = dates[-1]

    offset = 0
    while True:
        is_start = start_dt + timedelta(days=offset)
        is_end = is_start + timedelta(days=is_days)
        oos_start = is_end
        oos_end = oos_start + timedelta(days=oos_days)

        if oos_end > max_dt + timedelta(hours=1):
            break

        windows.append((is_start, is_end, oos_start, oos_end))
        offset += step_days

    return windows

# ============================================================
# IS グリッドサーチ (日次ポートフォリオ連動) - 高速化版
# ============================================================

import io
import contextlib

def run_is_portfolio_grid_search(dfs: dict, is_start: datetime.datetime, is_end: datetime.datetime,
                                 daily_selections: dict, top_n: int = 3) -> dict:
    """
    IS 期間の日次選定銘柄群に対してパラメータグリッドサーチを実行し、
    最もポートフォリオ累積PnLが高い最適パラメータを返す。
    """
    logic = logicinstance()
    bt_instance = backtester()
    fixed_initial_equity = 100.0

    # パラメータグリッドの軽量化（高速化）
    strategy_types = ["range", "breakout"]
    intervals = [60, 120]
    best_score = -float("inf")
    best_pnl = -float("inf")
    best_params = None

    # パラメータグリッド
    strategy_types = ["adaptive", "breakout", "range"]
    intervals = [60, 120]
    mp_periods = [24, 48, 72]
    er_thresholds = [0.3, 0.6]
    margin_pcts = [0.5, 1.0, 1.5, 2.0]
    atr_tp_multis = [1.5, 2.5]

    # IS期間内の11:00日付リスト
    is_dates = [d for d in daily_selections.keys() if is_start <= d < is_end]
    if not is_dates:
        return {"best_params": {"strategy_type": "adaptive", "interval": 60, "mp_period": 48, "er_threshold": 0.3, "sl_margin_pct": 1.0, "atr_tp_multi": 1.5}, "best_pnl": 0.0}

    total_combos = len(strategy_types) * len(intervals) * len(mp_periods) * len(er_thresholds) * len(margin_pcts) * len(atr_tp_multis)
    combo_count = 0

    # stdout非表示用
    f_null = io.StringIO()

    for strat in strategy_types:
        for interval in intervals:
            for mp_period in mp_periods:
                for er_th in er_thresholds:
                    for margin in margin_pcts:
                        for tp_multi in atr_tp_multis:
                            combo_count += 1
                            daily_pnls_list = []

                            for dt in is_dates:
                                m_state, selected = daily_selections[dt]
                                if not selected:
                                    daily_pnls_list.append(0.0)
                                    continue

                                next_dt = dt + timedelta(days=1)
                                day_pnls = []

                                for sym, side in selected:
                                    df_sym = dfs.get(sym)
                                    if df_sym is None:
                                        continue
                                    df_day = df_sym[(df_sym["timestamp"] >= dt - timedelta(hours=interval//60 * mp_period)) & 
                                                    (df_sym["timestamp"] <= next_dt)].copy()
                                    if len(df_day) < 10:
                                        continue

                                    df_resampled = resample_candles(df_day, interval)
                                    if df_resampled is None or len(df_resampled) < 10:
                                        continue

                                    df_logic = logic.make_logic(df_resampled, market_profile_period=mp_period,
                                                                er_threshold=er_th, strategy_type=strat)
                                    df_run = df_logic.copy()
                                    if strat == "breakout":
                                        df_run["long"] = df_run["long_breakout"] & (df_run["er"] > er_th)
                                    elif strat == "adaptive":
                                        df_run["long"] = np.where(df_run["er"] > er_th, df_run["long_breakout"], df_run["long_range"])
                                    else:
                                        df_run["long"] = df_run["long_range"]

                                    # バックテスト中のプリント文をミュート化してIOオーバーヘッドをカット
                                    with contextlib.redirect_stdout(f_null):
                                        res_df = bt_instance.run_backtest(
                                            df=df_run, lot=1.0, data_equity=100.0,
                                            side_mode=side, mp_period=mp_period, atr_tp_multi=tp_multi,
                                            er_threshold=er_th, strategy_type=strat, sl_margin_pct=margin
                                        )
                                    if not res_df.empty and "pnl" in res_df.columns:
                                        final_pnl = res_df["pnl"].iloc[-1]
                                        pnl_diff = final_pnl - 100.0  # 1日あたりの純損益(USDT)
                                        day_pnls.append(pnl_diff)

                                if day_pnls:
                                    avg_day_pnl = sum(day_pnls) / len(day_pnls)
                                    daily_pnls_list.append(avg_day_pnl)
                                else:
                                    daily_pnls_list.append(0.0)

                            # リスク調整後リターン（シャープレシオ & リカバリーファクター）の計算
                            arr_pnls = np.array(daily_pnls_list)
                            cum_curve = np.cumsum(arr_pnls) + fixed_initial_equity
                            total_pnl = cum_curve[-1] - fixed_initial_equity

                            running_max = np.maximum.accumulate(cum_curve)
                            drawdown = running_max - cum_curve
                            max_dd = float(np.max(drawdown))

                            std_pnl = float(np.std(arr_pnls))
                            mean_pnl = float(np.mean(arr_pnls))
                            sharpe = (mean_pnl / std_pnl * np.sqrt(365)) if std_pnl > 1e-6 else 0.0

                            if total_pnl > 0:
                                recovery_factor = total_pnl / (max_dd + 5.0)
                                score = recovery_factor * max(0.1, sharpe)
                            else:
                                score = total_pnl - max_dd

                            if score > best_score:
                                best_score = score
                                best_pnl = total_pnl
                                best_params = {
                                    "strategy_type": strat,
                                    "interval": interval,
                                    "mp_period": mp_period,
                                    "er_threshold": er_th,
                                    "sl_margin_pct": margin,
                                    "atr_tp_multi": tp_multi
                                }

                            if combo_count % max(1, total_combos // 5) == 0:
                                pct = combo_count / total_combos * 100
                                log(f"    IS grid search: {combo_count}/{total_combos} ({pct:.0f}%)  best PnL: {best_pnl:.2f} USDT (Score: {best_score:.2f})")

    if best_params is None:
        best_params = {"strategy_type": "adaptive", "interval": 60, "mp_period": 48, "er_threshold": 0.3, "sl_margin_pct": 1.0, "atr_tp_multi": 1.5}

    return {"best_params": best_params, "best_pnl": best_pnl}


# ============================================================
# OOS バックテスト (日次銘柄チェンジ連動)
# ============================================================

def run_oos_portfolio_backtest(dfs: dict, oos_start: datetime.datetime, oos_end: datetime.datetime,
                               daily_selections: dict, params: dict, top_n: int = 3) -> dict:
    """
    OOS 期間において、日次選定銘柄に決定したパラメータを適用し、
    日ごとの損益を統合した結果を返す。
    """
    logic = logicinstance()
    bt_instance = backtester()
    fixed_initial_equity = 100.0
    f_null = io.StringIO()

    strat = params["strategy_type"]
    interval = params["interval"]
    mp_period = params["mp_period"]
    er_th = params["er_threshold"]
    margin = params["sl_margin_pct"]
    tp_multi = params.get("atr_tp_multi", 1.5)

    oos_dates = [d for d in daily_selections.keys() if oos_start <= d < oos_end]

    cumulative_equity = fixed_initial_equity
    equity_curve = []
    timestamps = []
    trade_logs = []

    for dt in oos_dates:
        m_state, selected = daily_selections[dt]
        next_dt = dt + timedelta(days=1)
        day_pnls = []

        if selected:
            for sym, side in selected:
                df_sym = dfs.get(sym)
                if df_sym is None:
                    continue
                df_day = df_sym[(df_sym["timestamp"] >= dt - timedelta(hours=interval//60 * mp_period)) & 
                                (df_sym["timestamp"] <= next_dt)].copy()
                if len(df_day) < 10:
                    continue

                df_resampled = resample_candles(df_day, interval)
                if df_resampled is None or len(df_resampled) < 10:
                    continue

                df_logic = logic.make_logic(df_resampled, market_profile_period=mp_period,
                                            er_threshold=er_th, strategy_type=strat)
                df_run = df_logic.copy()
                if strat == "breakout":
                    df_run["long"] = df_run["long_breakout"] & (df_run["er"] > er_th)
                elif strat == "adaptive":
                    df_run["long"] = np.where(df_run["er"] > er_th, df_run["long_breakout"], df_run["long_range"])
                else:
                    df_run["long"] = df_run["long_range"]

                with contextlib.redirect_stdout(f_null):
                    res_df = bt_instance.run_backtest(
                        df=df_run, lot=1.0, data_equity=100.0,
                        side_mode=side, mp_period=mp_period, atr_tp_multi=tp_multi,
                        er_threshold=er_th, strategy_type=strat, sl_margin_pct=margin
                    )
                if not res_df.empty and "pnl" in res_df.columns:
                    final_pnl = res_df["pnl"].iloc[-1]
                    pnl_diff = final_pnl - 100.0
                    day_pnls.append(pnl_diff)
                    trade_logs.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "side": side,
                        "return_pct": pnl_diff
                    })

        avg_day_pnl = sum(day_pnls) / len(day_pnls) if day_pnls else 0.0
        cumulative_equity += avg_day_pnl
        timestamps.append(dt)
        equity_curve.append(cumulative_equity)

    final_pnl = cumulative_equity - fixed_initial_equity
    return_pct = (cumulative_equity - fixed_initial_equity) / fixed_initial_equity * 100.0

    return {
        "final_pnl": final_pnl,
        "return_pct": return_pct,
        "timestamps": timestamps,
        "equity_curve": equity_curve,
        "trade_logs": trade_logs
    }

# ============================================================
# プロット
# ============================================================

def plot_wfa_portfolio_equity(window_results: list, save_path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Dynamic Portfolio WFA — OOS Equity Curve", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    cumulative_equity = 100.0
    all_timestamps = []
    all_equities = []
    window_boundaries = []

    for wr in window_results:
        ts_list = wr.get("oos_timestamps", [])
        eq_list = wr.get("oos_equity_curve", [])
        if not ts_list or not eq_list:
            continue

        scale = cumulative_equity / eq_list[0] if eq_list[0] != 0 else 1.0
        scaled_eq = [e * scale for e in eq_list]

        all_timestamps.extend(ts_list)
        all_equities.extend(scaled_eq)
        cumulative_equity = scaled_eq[-1]
        window_boundaries.append(ts_list[0])

    if all_timestamps:
        ax1.plot(all_timestamps, all_equities, color="#4CAF50", linewidth=1.8, label="Portfolio OOS Equity")
        ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="Initial Equity (100)")
        for wb in window_boundaries:
            ax1.axvline(x=wb, color="orange", linestyle=":", alpha=0.7)

        ax1.set_ylabel("Equity (USDT)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    ax2 = axes[1]
    labels, is_returns, oos_returns = [], [], []
    for i, wr in enumerate(window_results):
        labels.append(f"W{i+1}")
        is_returns.append(wr.get("is_return_pct", 0))
        oos_returns.append(wr.get("oos_return_pct", 0))

    if labels:
        x = np.arange(len(labels))
        width = 0.35
        ax2.bar(x - width/2, is_returns, width, label="IS Return %", color="#2196F3", alpha=0.7)
        ax2.bar(x + width/2, oos_returns, width, label="OOS Return %", color="#4CAF50", alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Return (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.axhline(y=0, color="gray", linewidth=0.8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"統合エクイティカーブ画像を保存しました: {save_path}")

# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Dynamic Portfolio Walk-Forward Analysis")
    parser.add_argument("--is-days", type=int, default=180, help="IS期間の日数 (default: 180)")
    parser.add_argument("--oos-days", type=int, default=60, help="OOS期間の日数 (default: 60)")
    parser.add_argument("--step-days", type=int, default=60, help="ウィンドウ移動日数 (default: 60)")
    parser.add_argument("--top-n", type=int, default=3, help="日次選定銘柄数 (default: 3)")
    parser.add_argument("--last-n-windows", type=int, default=None, help="直近の実行ウィンドウ数制限 (default: 全て実行)")
    args = parser.parse_args()

    log("=" * 60)
    log("Dynamic Portfolio Walk-Forward Analysis (日次銘柄選定WFA) 開始")
    log("=" * 60)
    log(f"IS期間     : {args.is_days} 日")
    log(f"OOS期間    : {args.oos_days} 日")
    log(f"ステップ   : {args.step_days} 日")
    log(f"選定銘柄数 : Top {args.top_n}")
    if args.last_n_windows is not None:
        log(f"実行制限   : 直近 {args.last_n_windows} ウィンドウ")

    csv_path = SCRIPT_DIR / "Data" / "historical_all_symbols_merged.csv"
    if not csv_path.exists():
        log(f"ERROR: {csv_path} が見つかりません。")
        sys.exit(1)

    dfs = load_all_symbols_data(csv_path)

    # 全期間の日次選定を事前計算キャッシュ化 (高速化)
    log("全期間の日次銘柄選定スコアリングを計算中...")
    dates = generate_dates(dfs)
    daily_selections = {}
    for dt in dates:
        m_state, selected = select_daily_symbols(dfs, dt, top_n=args.top_n)
        daily_selections[dt] = (m_state, selected)
    log(f"  {len(daily_selections)} 日分の日次銘柄選定完了。")

    windows = generate_windows(dates, args.is_days, args.oos_days, args.step_days)
    if args.last_n_windows is not None:
        if args.last_n_windows > 0:
            windows = windows[-args.last_n_windows:]
            log(f"制限適用: 直近の {args.last_n_windows} ウィンドウのみを実行します。")
        else:
            log("ERROR: --last-n-windows は1以上の整数を指定してください。")
            sys.exit(1)

    log(f"\nWFA ウィンドウ数: {len(windows)}")
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
        log(f"  Window {i+1}: IS[{is_s.strftime('%Y-%m-%d')} ~ {is_e.strftime('%Y-%m-%d')}] "
            f"→ OOS[{oos_s.strftime('%Y-%m-%d')} ~ {oos_e.strftime('%Y-%m-%d')}]")

    window_results = []
    total_start = time.time()

    for i, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
        log(f"\n{'='*60}")
        log(f"Window {i+1}/{len(windows)}")
        log(f"{'='*60}")

        # IS グリッドサーチ
        log("  IS ポートフォリオパラメータ最適化開始...")
        t0 = time.time()
        is_res = run_is_portfolio_grid_search(dfs, is_start, is_end, daily_selections, top_n=args.top_n)
        t_is = time.time() - t0
        log(f"  IS 最適化完了 ({t_is:.1f}秒)")

        bp = is_res["best_params"]
        log(f"  IS 最適パラメータ: {bp['strategy_type']}_{bp['interval']}m_MP{bp['mp_period']}_ER{bp['er_threshold']}_SL{bp['sl_margin_pct']}%_TP{bp.get('atr_tp_multi', 1.5)}x")
        log(f"  IS PnL: {is_res['best_pnl']:.2f} USDT")

        is_ret_pct = is_res["best_pnl"]

        # OOS バックテスト (日次銘柄ローテーション)
        log("  OOS 日次銘柄ローテーションバックテスト開始...")
        t0 = time.time()
        oos_res = run_oos_portfolio_backtest(dfs, oos_start, oos_end, daily_selections, bp, top_n=args.top_n)
        t_oos = time.time() - t0
        log(f"  OOS バックテスト完了 ({t_oos:.1f}秒)")
        log(f"  OOS PnL: {oos_res['final_pnl']:.2f} USDT (Return: {oos_res['return_pct']:.2f}%)")

        window_results.append({
            "window": i + 1,
            "is_start": is_start,
            "is_end": is_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
            "best_params": bp,
            "is_return_pct": is_ret_pct,
            "oos_return_pct": oos_res["return_pct"],
            "oos_timestamps": oos_res["timestamps"],
            "oos_equity_curve": oos_res["equity_curve"],
            "trade_logs": oos_res["trade_logs"]
        })

    total_elapsed = time.time() - total_start

    # ============================================================
    # 統合評価・レポート
    # ============================================================
    oos_total_pnl = sum(wr["oos_return_pct"] for wr in window_results)
    total_days = sum((wr["oos_end"] - wr["oos_start"]).days for wr in window_results)

    is_avg_pnl = float(np.mean([wr["is_return_pct"] for wr in window_results]))
    oos_annual = (oos_total_pnl / max(1, total_days)) * 365.0
    is_annual = (is_avg_pnl / max(1, args.is_days)) * 365.0
    wfe = (oos_annual / is_annual * 100.0) if is_annual > 0 else float("nan")

    print()
    print("=" * 60)
    print("  Dynamic Portfolio WFA 最終レポート")
    print("=" * 60)
    print(f"  分析ウィンドウ数   : {len(window_results)}")
    print(f"  OOS合計期間        : {total_days} 日")
    print(f"  OOS 統合損益 (USDT): {oos_total_pnl:+.2f} USDT")
    print(f"  OOS 年間換算損益   : {oos_annual:+.2f} USDT/年")
    print(f"  IS  平均損益 (USDT): {is_avg_pnl:+.2f} USDT")
    print(f"  IS  年間換算損益   : {is_annual:+.2f} USDT/年")
    print(f"  WFE (検証効率)     : {wfe:.1f}%", end="")
    if not np.isnan(wfe):
        print("  [PASS] 合格 (50%以上)" if wfe >= 50 else "  [FAIL] 不合格 (50%未満)")
    else:
        print("  (算出不可)")
    print(f"  総処理時間         : {total_elapsed:.1f}秒")
    print("=" * 60)

    # 結果保存
    os.makedirs(SCRIPT_DIR / "backtest_data", exist_ok=True)
    img_out = str(SCRIPT_DIR / "backtest_data" / "wfa_portfolio_equity.jpg")
    plot_wfa_portfolio_equity(window_results, img_out)
    log("Dynamic Portfolio WFA 正常完了!")

if __name__ == "__main__":
    main()
