#!/usr/bin/env python
# coding: utf-8
"""
Walk-Forward Analysis (WFA) バックテストスクリプト

過剰最適化（カーブフィッティング）を防止するため、データを
IS（In-Sample: 学習）期間と OOS（Out-of-Sample: 検証）期間に分割し、
ローリングウィンドウで検証を行う。

使い方:
    python walk_forward_backtest.py [--symbol LINKUSDT] [--side long]
                                     [--is-days 180] [--oos-days 60] [--step-days 60]
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
from bitget5_44_3logic import logicinstance, backtester, resample_candles, MPStrategy

JST = timezone(timedelta(hours=9))

# ============================================================
# ユーティリティ
# ============================================================

def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}")


def load_symbol_1h(symbol: str, csv_path: Path) -> pd.DataFrame:
    """ヒストリカルCSVから指定銘柄の1時間足OHLCVを抽出する。"""
    log(f"Loading {symbol} from {csv_path.name} ...")
    df_all = pd.read_csv(csv_path)
    df_sym = df_all[df_all["symbol"] == symbol].copy()
    if df_sym.empty:
        raise ValueError(f"Symbol {symbol} not found in {csv_path}")

    df_sym["timestamp"] = pd.to_datetime(df_sym["timestamp"], utc=True).dt.tz_convert(JST)
    # backtester 内部の merge で tz-aware/naive 不一致を防ぐため tz を除去
    df_sym["timestamp"] = df_sym["timestamp"].dt.tz_localize(None)
    df_sym = df_sym.sort_values("timestamp").reset_index(drop=True)

    # backtester / resample_candles が期待するカラム名にリネーム
    df_out = pd.DataFrame({
        "timestamp": df_sym["timestamp"],
        "open":   df_sym["bybit_open"].astype(float),
        "high":   df_sym["bybit_high"].astype(float),
        "low":    df_sym["bybit_low"].astype(float),
        "close":  df_sym["bybit_close"].astype(float),
        "volume": df_sym["bybit_volume"].astype(float),
    })
    log(f"  Loaded {len(df_out)} rows  ({df_out['timestamp'].iloc[0]} ~ {df_out['timestamp'].iloc[-1]})")
    return df_out


# ============================================================
# ウィンドウ分割
# ============================================================

def generate_windows(df: pd.DataFrame, is_days: int, oos_days: int, step_days: int):
    """
    ローリングウィンドウのリストを生成する。
    各要素は (is_start, is_end, oos_start, oos_end) のタプル。
    """
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    total_days = (ts_max - ts_min).days

    windows = []
    offset = 0
    while True:
        is_start = ts_min + timedelta(days=offset)
        is_end   = is_start + timedelta(days=is_days)
        oos_start = is_end
        oos_end   = oos_start + timedelta(days=oos_days)

        # OOS 終端がデータ範囲を超えたら終了
        if oos_end > ts_max + timedelta(hours=1):
            break

        windows.append((is_start, is_end, oos_start, oos_end))
        offset += step_days

    return windows


# ============================================================
# IS グリッドサーチ（run_interval_comparison 相当）
# ============================================================

def run_is_grid_search(df_60m: pd.DataFrame, lot: float, data_equity: float,
                       side_mode: str = "long") -> dict:
    """
    IS 期間のデータで全パラメータ組み合わせをグリッドサーチし、
    最良パラメータを返す。

    Returns:
        dict: {
            'best_params': {...},
            'best_pnl': float,
            'best_metrics': {...},
            'all_results': {label: {...}, ...}
        }
    """
    logic = logicinstance()
    bt_instance = backtester()
    results = {}
    fixed_initial_equity = 100.0

    strategy_types = ["range", "breakout"]
    intervals = [60, 120, 180]
    mp_periods = [12, 24, 36, 48, 60, 72, 96, 120, 144, 168]
    er_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # 生データから計算済みカラムを除去
    base_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if "turnover" in df_60m.columns:
        base_cols.append("turnover")
    df_raw = df_60m[[c for c in base_cols if c in df_60m.columns]].copy()

    total = len(strategy_types) * len(intervals) * len(mp_periods) * len(er_thresholds)
    count = 0

    for strat in strategy_types:
        for interval in intervals:
            # リサンプリング
            df_interval = resample_candles(df_raw, interval)
            if df_interval is None or df_interval.empty or len(df_interval) < 30:
                count += len(mp_periods) * len(er_thresholds)
                continue

            for mp_period in mp_periods:
                # ガードレール1: データ本数の半分を超えるMP期間はスキップ (データ不足防止)
                if mp_period > len(df_interval) // 2:
                    count += len(er_thresholds)
                    continue

                # make_logic (ER=0 で全シグナル取得)
                df_copy = df_interval.copy()
                df_copy = logic.make_logic(df_copy, market_profile_period=mp_period,
                                           er_threshold=0, strategy_type=strat)

                for er_th in er_thresholds:
                    count += 1
                    df_run = df_copy.copy()
                    if strat == "breakout":
                        df_run["long"] = df_run["long_breakout"] & (df_run["er"] > er_th)
                    else:
                        df_run["long"] = df_run["long_range"]

                    label = f"{strat}_{interval}m_MP{mp_period}_ER{er_th}"

                    result_df = bt_instance.run_backtest(
                        df=df_run, lot=lot, data_equity=fixed_initial_equity,
                        side_mode=side_mode, mp_period=mp_period, atr_tp_multi=1.5,
                        er_threshold=er_th, strategy_type=strat, sl_margin_pct=1.0
                    )

                    metrics = getattr(bt_instance, "last_metrics", {})
                    final_pnl = (result_df["pnl"].iloc[-1]
                                 if not result_df.empty and "pnl" in result_df.columns
                                 else 0)

                    results[label] = {
                        "strategy_type": strat,
                        "interval": interval,
                        "mp_period": mp_period,
                        "er_threshold": er_th,
                        "final_pnl": final_pnl,
                        "DD_max": metrics.get("DD_max", 0),
                        "DD_per": metrics.get("DD_per", 0),
                        "win_rate": metrics.get("win_rate", 0),
                        "PF": metrics.get("PF", float("inf")),
                        "trade_count": metrics.get("trade_count", 0),
                    }

                    # プログレス (10% ごと)
                    if count % max(1, total // 10) == 0:
                        pct = count / total * 100
                        print(f"  IS grid search: {count}/{total} ({pct:.0f}%)  "
                              f"latest: {label} PnL={final_pnl:.2f}")

    if not results:
        return {
            "best_params": {"strategy_type": "range", "interval": 60,
                            "mp_period": 48, "er_threshold": 0.3,
                            "sl_margin_pct": 1.0},
            "best_pnl": 0.0,
            "best_metrics": {},
            "all_results": results,
        }

    # ガードレール2: 最低3回以上取引がある候補からベスト選定（運勝ち排除）
    valid_candidates = {k: v for k, v in results.items() if v.get("trade_count", 0) >= 3}
    if not valid_candidates:
        # 3回以上のトレードがない場合は1回以上の候補
        valid_candidates = {k: v for k, v in results.items() if v.get("trade_count", 0) >= 1}
    if not valid_candidates:
        valid_candidates = results

    best_key = max(valid_candidates.keys(), key=lambda k: valid_candidates[k]["final_pnl"])
    best = valid_candidates[best_key]

    # --- 第2段階: SL マージン最適化 ---
    margin_pcts = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    best_margin = 1.0
    best_margin_pnl = best["final_pnl"]

    df_interval = resample_candles(df_raw, best["interval"])
    df_logic = logic.make_logic(df_interval, market_profile_period=best["mp_period"],
                                er_threshold=0, strategy_type=best["strategy_type"])

    for margin_pct in margin_pcts:
        df_run = df_logic.copy()
        if best["strategy_type"] == "breakout":
            df_run["long"] = df_run["long_breakout"] & (df_run["er"] > best["er_threshold"])
        else:
            df_run["long"] = df_run["long_range"]

        result_df = bt_instance.run_backtest(
            df=df_run, lot=lot, data_equity=fixed_initial_equity,
            side_mode=side_mode, mp_period=best["mp_period"],
            atr_tp_multi=1.5, er_threshold=best["er_threshold"],
            strategy_type=best["strategy_type"], sl_margin_pct=margin_pct
        )
        final_pnl = (result_df["pnl"].iloc[-1]
                     if not result_df.empty and "pnl" in result_df.columns
                     else 0)
        if final_pnl > best_margin_pnl:
            best_margin_pnl = final_pnl
            best_margin = margin_pct

    best_params = {
        "strategy_type": best["strategy_type"],
        "interval": best["interval"],
        "mp_period": best["mp_period"],
        "er_threshold": best["er_threshold"],
        "sl_margin_pct": best_margin,
    }

    return {
        "best_params": best_params,
        "best_pnl": best_margin_pnl,
        "best_metrics": best,
        "all_results": results,
    }


# ============================================================
# OOS バックテスト
# ============================================================

def run_oos_backtest(df_60m: pd.DataFrame, params: dict, lot: float,
                     data_equity: float, side_mode: str = "long") -> dict:
    """
    OOS 期間のデータに IS で選定したパラメータを適用してバックテストを実行する。

    Returns:
        dict: {'final_pnl', 'metrics', 'result_df'}
    """
    logic = logicinstance()
    bt_instance = backtester()
    fixed_initial_equity = 100.0

    base_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if "turnover" in df_60m.columns:
        base_cols.append("turnover")
    df_raw = df_60m[[c for c in base_cols if c in df_60m.columns]].copy()

    interval = params["interval"]
    strat = params["strategy_type"]
    mp_period = params["mp_period"]
    er_th = params["er_threshold"]
    margin = params["sl_margin_pct"]

    df_interval = resample_candles(df_raw, interval)
    if df_interval is None or df_interval.empty or len(df_interval) < 10:
        return {"final_pnl": 0.0, "metrics": {}, "result_df": pd.DataFrame()}

    df_logic = logic.make_logic(df_interval, market_profile_period=mp_period,
                                er_threshold=0, strategy_type=strat)

    df_run = df_logic.copy()
    if strat == "breakout":
        df_run["long"] = df_run["long_breakout"] & (df_run["er"] > er_th)
    else:
        df_run["long"] = df_run["long_range"]

    result_df = bt_instance.run_backtest(
        df=df_run, lot=lot, data_equity=fixed_initial_equity,
        side_mode=side_mode, mp_period=mp_period, atr_tp_multi=1.5,
        er_threshold=er_th, strategy_type=strat, sl_margin_pct=margin
    )

    metrics = getattr(bt_instance, "last_metrics", {})
    final_pnl = (result_df["pnl"].iloc[-1]
                 if not result_df.empty and "pnl" in result_df.columns
                 else 0)

    return {
        "final_pnl": final_pnl,
        "initial_equity": fixed_initial_equity,
        "return_pct": (final_pnl - fixed_initial_equity) / fixed_initial_equity * 100
                      if fixed_initial_equity > 0 else 0,
        "metrics": metrics,
        "result_df": result_df,
    }


# ============================================================
# WFE 算出
# ============================================================

def calc_annualized_return(return_pct: float, days: int) -> float:
    """日次ベースの累積リターンを年間換算する。"""
    if days <= 0:
        return 0.0
    factor = 1.0 + return_pct / 100.0
    if factor <= 0:
        return -100.0
    return (factor ** (365.0 / days) - 1.0) * 100.0


# ============================================================
# プロット
# ============================================================

def plot_wfa_equity(window_results: list, save_path: str) -> None:
    """OOS 統合エクイティカーブを描画して保存する。"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Walk-Forward Analysis — OOS Equity Curve", fontsize=14, fontweight="bold")

    # --- 上段: OOS エクイティカーブ ---
    ax1 = axes[0]
    cumulative_equity = 100.0
    all_timestamps = []
    all_equities = []
    window_boundaries = []

    for wr in window_results:
        rdf = wr.get("oos_result_df")
        if rdf is None or rdf.empty or "pnl" not in rdf.columns:
            continue

        # OOS 期間の PnL カーブを累積に変換
        pnl_series = rdf["pnl"].values
        if len(pnl_series) == 0:
            continue

        # 100ベースの pnl → 変化率に変換して累積に乗せる
        initial_val = pnl_series[0] if pnl_series[0] != 0 else 100.0
        scale = cumulative_equity / initial_val

        timestamps = pd.to_datetime(rdf["timestamp"])
        equities = pnl_series * scale

        all_timestamps.extend(timestamps.tolist())
        all_equities.extend(equities.tolist())

        cumulative_equity = equities[-1]
        window_boundaries.append(timestamps.iloc[0])

    if all_timestamps:
        ax1.plot(all_timestamps, all_equities, color="#2196F3", linewidth=1.5, label="OOS Equity")
        ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="Initial (100)")

        for wb in window_boundaries:
            ax1.axvline(x=wb, color="orange", linestyle=":", alpha=0.7)

        ax1.set_ylabel("Equity (USDT)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    # --- 下段: ウィンドウごとの棒グラフ ---
    ax2 = axes[1]
    labels = []
    is_returns = []
    oos_returns = []
    for i, wr in enumerate(window_results):
        labels.append(f"W{i+1}")
        is_returns.append(wr.get("is_return_pct", 0))
        oos_returns.append(wr.get("oos_return_pct", 0))

    if labels:
        x = np.arange(len(labels))
        width = 0.35
        bars_is = ax2.bar(x - width/2, is_returns, width, label="IS Return %",
                          color="#4CAF50", alpha=0.7)
        bars_oos = ax2.bar(x + width/2, oos_returns, width, label="OOS Return %",
                           color="#2196F3", alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Return (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.axhline(y=0, color="gray", linewidth=0.8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"Equity curve saved: {save_path}")


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Analysis Backtest")
    parser.add_argument("--symbol", type=str, default="LINKUSDT",
                        help="対象銘柄 (default: LINKUSDT)")
    parser.add_argument("--side", type=str, default="long",
                        choices=["long", "short", "both"],
                        help="トレード方向 (default: long)")
    parser.add_argument("--is-days", type=int, default=180,
                        help="IS (In-Sample) 期間の日数 (default: 180)")
    parser.add_argument("--oos-days", type=int, default=60,
                        help="OOS (Out-of-Sample) 期間の日数 (default: 60)")
    parser.add_argument("--step-days", type=int, default=60,
                        help="ウィンドウ移動日数 (default: 60)")
    parser.add_argument("--lot", type=float, default=1.0,
                        help="ロットサイズ (default: 1.0)")
    parser.add_argument("--equity", type=float, default=100.0,
                        help="初期資金 (default: 100.0)")
    args = parser.parse_args()

    log("=" * 60)
    log("Walk-Forward Analysis (WFA) バックテスト開始")
    log("=" * 60)
    log(f"銘柄       : {args.symbol}")
    log(f"方向       : {args.side}")
    log(f"IS期間     : {args.is_days} 日")
    log(f"OOS期間    : {args.oos_days} 日")
    log(f"ステップ   : {args.step_days} 日")
    log(f"ロット     : {args.lot}")
    log(f"初期資金   : {args.equity}")

    # --- データ読み込み ---
    csv_path = SCRIPT_DIR / "Data" / "historical_all_symbols_merged.csv"
    if not csv_path.exists():
        log(f"ERROR: {csv_path} が見つかりません。download_historical_candles.py を先に実行してください。")
        sys.exit(1)

    df_1h = load_symbol_1h(args.symbol, csv_path)

    # --- ウィンドウ生成 ---
    windows = generate_windows(df_1h, args.is_days, args.oos_days, args.step_days)
    if not windows:
        log("ERROR: 十分なデータがありません。IS+OOS期間がデータ範囲を超えています。")
        sys.exit(1)

    log(f"\nウィンドウ数: {len(windows)}")
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
        log(f"  Window {i+1}: IS[{is_s.strftime('%Y-%m-%d')} ~ {is_e.strftime('%Y-%m-%d')}] "
            f"→ OOS[{oos_s.strftime('%Y-%m-%d')} ~ {oos_e.strftime('%Y-%m-%d')}]")

    # --- 各ウィンドウ処理 ---
    window_results = []
    total_start = time.time()

    for i, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
        log(f"\n{'='*60}")
        log(f"Window {i+1}/{len(windows)}")
        log(f"{'='*60}")

        # IS 期間データ抽出
        df_is = df_1h[(df_1h["timestamp"] >= is_start) & (df_1h["timestamp"] < is_end)].copy()
        df_is = df_is.reset_index(drop=True)
        log(f"  IS data: {len(df_is)} rows  ({is_start.strftime('%Y-%m-%d')} ~ {is_end.strftime('%Y-%m-%d')})")

        if len(df_is) < 100:
            log(f"  SKIP: IS データが少なすぎます ({len(df_is)} rows)")
            continue

        # IS グリッドサーチ
        t0 = time.time()
        log(f"  IS グリッドサーチ開始 ...")
        is_result = run_is_grid_search(df_is, args.lot, args.equity, args.side)
        t_is = time.time() - t0
        log(f"  IS グリッドサーチ完了 ({t_is:.1f}秒)")

        bp = is_result["best_params"]
        log(f"  IS 最適パラメータ: {bp['strategy_type']}_{bp['interval']}m"
            f"_MP{bp['mp_period']}_ER{bp['er_threshold']}_SL{bp['sl_margin_pct']}%")
        log(f"  IS PnL: {is_result['best_pnl']:.2f} USDT")

        is_return_pct = ((is_result["best_pnl"] - 100.0) / 100.0 * 100.0)

        # OOS 期間データ抽出
        df_oos = df_1h[(df_1h["timestamp"] >= oos_start) & (df_1h["timestamp"] < oos_end)].copy()
        df_oos = df_oos.reset_index(drop=True)
        log(f"  OOS data: {len(df_oos)} rows  ({oos_start.strftime('%Y-%m-%d')} ~ {oos_end.strftime('%Y-%m-%d')})")

        if len(df_oos) < 24:
            log(f"  SKIP: OOS データが少なすぎます ({len(df_oos)} rows)")
            continue

        # OOS バックテスト
        t0 = time.time()
        oos_result = run_oos_backtest(df_oos, bp, args.lot, args.equity, args.side)
        t_oos = time.time() - t0
        log(f"  OOS バックテスト完了 ({t_oos:.1f}秒)")
        log(f"  OOS PnL: {oos_result['final_pnl']:.2f} USDT "
            f"(Return: {oos_result['return_pct']:.2f}%)")

        window_results.append({
            "window": i + 1,
            "is_start": is_start,
            "is_end": is_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
            "best_params": bp,
            "is_pnl": is_result["best_pnl"],
            "is_return_pct": is_return_pct,
            "oos_pnl": oos_result["final_pnl"],
            "oos_return_pct": oos_result["return_pct"],
            "oos_metrics": oos_result["metrics"],
            "oos_result_df": oos_result.get("result_df"),
        })

    total_elapsed = time.time() - total_start

    if not window_results:
        log("\nERROR: 有効なウィンドウ結果がありませんでした。")
        sys.exit(1)

    # ============================================================
    # 統合結果の算出
    # ============================================================
    log(f"\n{'='*60}")
    log("WFA 統合結果")
    log(f"{'='*60}")

    # OOS 統合リターン (複利)
    oos_cumulative = 1.0
    for wr in window_results:
        oos_cumulative *= (1.0 + wr["oos_return_pct"] / 100.0)
    oos_total_return_pct = (oos_cumulative - 1.0) * 100.0
    oos_total_days = sum((wr["oos_end"] - wr["oos_start"]).days for wr in window_results)

    # IS 平均リターン
    is_avg_return_pct = np.mean([wr["is_return_pct"] for wr in window_results])

    # 年間換算
    oos_annual = calc_annualized_return(oos_total_return_pct, oos_total_days)
    is_annual = calc_annualized_return(is_avg_return_pct,
                                       int(np.mean([args.is_days] * len(window_results))))

    # WFE
    if is_annual > 0:
        wfe = oos_annual / is_annual * 100.0
    elif is_annual == 0:
        wfe = 0.0
    else:
        # IS がマイナスの場合は WFE 算出不可
        wfe = float("nan")

    # パラメータ安定性
    strategies_used = [wr["best_params"]["strategy_type"] for wr in window_results]
    most_common_strategy = max(set(strategies_used), key=strategies_used.count)
    strategy_consistency = strategies_used.count(most_common_strategy) / len(strategies_used) * 100

    # --- 表示 ---
    print()
    print("=" * 60)
    print("  Walk-Forward Analysis 最終レポート")
    print("=" * 60)
    print(f"  銘柄               : {args.symbol}")
    print(f"  分析ウィンドウ数   : {len(window_results)}")
    print(f"  OOS合計期間        : {oos_total_days} 日")
    print()

    print("  --- 各ウィンドウ詳細 ---")
    for wr in window_results:
        bp = wr["best_params"]
        print(f"  W{wr['window']}: IS [{wr['is_start'].strftime('%m/%d')}~{wr['is_end'].strftime('%m/%d')}]"
              f" → OOS [{wr['oos_start'].strftime('%m/%d')}~{wr['oos_end'].strftime('%m/%d')}]")
        print(f"       パラメータ: {bp['strategy_type']}_{bp['interval']}m"
              f"_MP{bp['mp_period']}_ER{bp['er_threshold']}_SL{bp['sl_margin_pct']}%")
        print(f"       IS: {wr['is_return_pct']:+.2f}%  →  OOS: {wr['oos_return_pct']:+.2f}%")
        metrics = wr.get("oos_metrics", {})
        print(f"       OOS DD: {metrics.get('DD_max', 0):.2f}  "
              f"勝率: {metrics.get('win_rate', 0)*100:.1f}%  "
              f"PF: {metrics.get('PF', 0):.2f}  "
              f"取引: {metrics.get('trade_count', 0)}")
    print()

    print("  --- 統合指標 ---")
    print(f"  OOS 統合リターン   : {oos_total_return_pct:+.2f}% ({oos_total_days}日間)")
    print(f"  OOS 年間換算       : {oos_annual:+.2f}%")
    print(f"  IS  平均リターン   : {is_avg_return_pct:+.2f}%")
    print(f"  IS  年間換算       : {is_annual:+.2f}%")
    print(f"  WFE                : {wfe:.1f}%", end="")
    if not np.isnan(wfe):
        if wfe >= 50:
            print("  [PASS] 合格")
        else:
            print("  [FAIL] 不合格 (50%未満)")
    else:
        print("  (算出不可: IS リターンがマイナス)")
    print(f"  パラメータ安定性   : {most_common_strategy} が "
          f"{strategies_used.count(most_common_strategy)}/{len(strategies_used)} 回選択 "
          f"({strategy_consistency:.0f}%)")
    print(f"  総処理時間         : {total_elapsed:.1f}秒")
    print("=" * 60)

    # --- CSV 出力 ---
    os.makedirs(SCRIPT_DIR / "backtest_data", exist_ok=True)
    csv_rows = []
    for wr in window_results:
        bp = wr["best_params"]
        metrics = wr.get("oos_metrics", {})
        csv_rows.append({
            "window": wr["window"],
            "is_start": wr["is_start"].strftime("%Y-%m-%d"),
            "is_end": wr["is_end"].strftime("%Y-%m-%d"),
            "oos_start": wr["oos_start"].strftime("%Y-%m-%d"),
            "oos_end": wr["oos_end"].strftime("%Y-%m-%d"),
            "strategy_type": bp["strategy_type"],
            "interval": bp["interval"],
            "mp_period": bp["mp_period"],
            "er_threshold": bp["er_threshold"],
            "sl_margin_pct": bp["sl_margin_pct"],
            "is_return_pct": round(wr["is_return_pct"], 4),
            "oos_return_pct": round(wr["oos_return_pct"], 4),
            "oos_dd_max": round(metrics.get("DD_max", 0), 4),
            "oos_win_rate": round(metrics.get("win_rate", 0), 4),
            "oos_pf": round(metrics.get("PF", 0), 4),
            "oos_trade_count": metrics.get("trade_count", 0),
        })

    # サマリー行を追加
    csv_rows.append({
        "window": "TOTAL",
        "is_start": "",
        "is_end": "",
        "oos_start": "",
        "oos_end": "",
        "strategy_type": "",
        "interval": "",
        "mp_period": "",
        "er_threshold": "",
        "sl_margin_pct": "",
        "is_return_pct": round(is_avg_return_pct, 4),
        "oos_return_pct": round(oos_total_return_pct, 4),
        "oos_dd_max": "",
        "oos_win_rate": "",
        "oos_pf": "",
        "oos_trade_count": f"WFE={wfe:.1f}%",
    })

    df_csv = pd.DataFrame(csv_rows)
    csv_out = SCRIPT_DIR / "backtest_data" / "wfa_results.csv"
    df_csv.to_csv(csv_out, index=False)
    log(f"結果CSV保存: {csv_out}")

    # --- グラフ出力 ---
    img_out = str(SCRIPT_DIR / "backtest_data" / "wfa_equity_curve.jpg")
    plot_wfa_equity(window_results, img_out)

    log("\nWalk-Forward Analysis 完了!")


if __name__ == "__main__":
    main()
