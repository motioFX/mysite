import os
import sys
import math
import datetime
from datetime import timezone, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

JST = timezone(timedelta(hours=9))

# 同一ディレクトリの bitget5_44_5backtest_mm から PnL 集計ロジックをインポート
sys.path.append(str(Path(__file__).resolve().parent))
try:
    from bitget5_44_5backtest_mm import make_mm_pl
except ImportError as e:
    print(f"Warning: Failed to import make_mm_pl from bitget5_44_5backtest_mm.py: {e}")
    make_mm_pl = None

# --- パラメータ（ウェイト係数）の定義 ---
# バックテスト実行時にこれらの値を変えることで、銘柄選定ロジックを最適化できます。
W_PERF = 1.0
W_PERF_BTC = 0.8
W_MOMENTUM = 0.5
W_DD = 0.5
W_VOL = 0.5
W_OI = 0.5
W_FUNDING = 0.3
W_CB_PREMIUM = 0.5  # コインベースプレミアムのウェイト

script_dir = Path(__file__).resolve().parent
data_dir = script_dir / "Data"
input_path = data_dir / "historical_all_symbols_merged.csv"
output_path = data_dir / "historical_selection_scores_1year.csv"

# --- Zスコア計算関数 ---
def get_zscore(series: pd.Series, invert: bool = False) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    z = (series - series.mean()) / std
    return -z if invert else z

# --- 地合い判定関数（本番と共通化するための判定ロジック） ---
def determine_market_state(mean_norm: float, mean_cb_premium: float) -> str:
    # 平均騰落率が1.0（前日比フラット）以上、またはコインベースプレミアム平均がプラスの場合に強気と判定
    if mean_norm >= 1.0 or mean_cb_premium > 0.0:
        return "long_only"
    else:
        return "short_only"

def main():
    print("Loading integrated master dataset...")
    if not input_path.exists():
        print(f"Error: Integrated dataset not found at {input_path}! Run download_historical_candles.py first.")
        sys.exit(1)
        
    df_master_raw = pd.read_csv(input_path)
    df_master_raw['timestamp'] = pd.to_datetime(df_master_raw['timestamp'], utc=True).dt.tz_convert(JST)
    
    # タイムスタンプでソート
    df_master_raw = df_master_raw.sort_values(by=['timestamp', 'symbol']).reset_index(drop=True)
    
    symbols = sorted(df_master_raw['symbol'].unique())
    min_ts = df_master_raw['timestamp'].min()
    max_ts = df_master_raw['timestamp'].max()
    print(f"Loaded data for {len(symbols)} symbols. Range: {min_ts} to {max_ts}")
    
    # 銘柄ごとに分割して辞書に保持（処理高速化のため）
    dfs = {}
    for sym in symbols:
        dfs[sym] = df_master_raw[df_master_raw['symbol'] == sym].sort_values('timestamp').reset_index(drop=True)
        
    # 日次の判定時点を設定（ルックバック最大20日に対応するため、開始日を21日後とする）
    start_day = (min_ts + timedelta(days=21)).replace(hour=11, minute=0, second=0, microsecond=0)
    end_day = max_ts.replace(hour=11, minute=0, second=0, microsecond=0)
    
    trade_dates = []
    curr = start_day
    while curr <= end_day:
        # 現在の判定日時点のデータが存在する日のみリストアップ
        trade_dates.append(curr)
        curr += timedelta(days=1)
        
    print(f"Generating scores and simulating backtest for {len(trade_dates)} days (from {start_day.date()} to {end_day.date()})...")
    
    all_daily_records = []
    daily_pnls = []      # 日々のポートフォリオ平均損益率
    simulated_dates = [] # シミュレーションが正常実行された日付
    
    for idx, target_dt in enumerate(trade_dates, 1):
        if idx % 30 == 0 or idx == len(trade_dates):
            print(f"  Processing day {idx}/{len(trade_dates)}: {target_dt.strftime('%Y-%m-%d')}...")
            
        # A. 地合い判定フェーズ (仮の5日ルックバックで計算)
        window_start_temp = target_dt - timedelta(days=5)
        temp_metrics = []
        
        # BTC データの取得
        btc_series_temp = None
        if "BTCUSDT" in dfs:
            df_btc = dfs["BTCUSDT"]
            df_btc_5d = df_btc[(df_btc['timestamp'] >= window_start_temp) & (df_btc['timestamp'] <= target_dt)]
            if len(df_btc_5d) >= 24 * 4:
                btc_series_temp = df_btc_5d['bybit_close'].astype(float)
                
        for symbol, df_sym in dfs.items():
            df_5d = df_sym[(df_sym['timestamp'] >= window_start_temp) & (df_sym['timestamp'] <= target_dt)]
            if len(df_5d) < 24 * 4:
                continue
            close_now = float(df_5d['bybit_close'].iloc[-1])
            close_5d_ago = float(df_5d['bybit_close'].iloc[0])
            norm_perf_5d = close_now / close_5d_ago if close_5d_ago > 0 else 1.0
            
            temp_metrics.append({
                "symbol": symbol,
                "norm_perf_5d": norm_perf_5d,
                "coinbase_premium": float(df_5d['coinbase_premium'].iloc[-1]) # 最新プレミアム
            })
            
        if not temp_metrics:
            continue
            
        df_temp = pd.DataFrame(temp_metrics)
        mean_norm = float(df_temp["norm_perf_5d"].mean())
        mean_cb_premium = float(df_temp["coinbase_premium"].mean())
        
        # 地合い判定の実行
        market_state = determine_market_state(mean_norm, mean_cb_premium)
        
        # B. 本番スコア計算フェーズ
        window_days = 30
        window_start = target_dt - timedelta(days=window_days)
        
        # 本番集計用のBTCデータの取得
        btc_series = None
        df_btc_window = None
        if "BTCUSDT" in dfs:
            df_btc = dfs["BTCUSDT"]
            df_btc_window = df_btc[(df_btc['timestamp'] >= window_start) & (df_btc['timestamp'] <= target_dt)].copy()
            if len(df_btc_window) >= 24 * (window_days - 1):
                btc_series = df_btc_window['bybit_close'].astype(float)
                
        metrics_list = []
        for symbol, df_sym in dfs.items():
            df_w = df_sym[(df_sym['timestamp'] >= window_start) & (df_sym['timestamp'] <= target_dt)].copy()
            if len(df_w) < 24 * (window_days - 1):
                continue
                
            close_series = df_w['bybit_close'].astype(float)
            close_now = float(close_series.iloc[-1])
            close_start = float(close_series.iloc[0])
            
            # 1. 期間標準パフォーマンス
            norm_perf_5d = close_now / close_start if close_start > 0 else 1.0
            
            # 2. BTC相対パフォーマンス
            norm_perf_5d_btc = 1.0
            if btc_series is not None:
                df_sym_btc = pd.merge(
                    df_w[['timestamp', 'bybit_close']],
                    df_btc_window[['timestamp', 'bybit_close']].rename(columns={'bybit_close': 'btc_close'}),
                    on='timestamp',
                    how='left'
                )
                if 'btc_close' in df_sym_btc.columns:
                    alt_btc = df_sym_btc['bybit_close'] / df_sym_btc['btc_close']
                    alt_btc = alt_btc.dropna()
                    if len(alt_btc) >= 2:
                        norm_perf_5d_btc = float(alt_btc.iloc[-1] / alt_btc.iloc[0]) if alt_btc.iloc[0] > 0 else 1.0
            
            # 3. モメンタム（前時間比で上昇した割合）
            diffs = close_series.diff().dropna()
            momentum_dir_5d = float((diffs > 0).mean()) if len(diffs) > 0 else 0.5
            price_dir = 1.0 if close_now >= close_start else -1.0
            
            # 4. 最大ドローダウン
            rolling_max = close_series.cummax()
            drawdowns = (close_series / rolling_max - 1.0) * 100.0
            max_drawdown_5d = float(drawdowns.min()) if not drawdowns.empty else 0.0
            
            # 5. 出来高変化率
            vol_series = df_w['bybit_volume'].astype(float).replace(0.0, np.nan).dropna()
            volume_change_5d = 0.0
            if len(vol_series) >= 4:
                mid = len(vol_series) // 2
                vol_first = float(vol_series.iloc[:mid].mean())
                vol_second = float(vol_series.iloc[mid:].mean())
                if vol_first > 0:
                    volume_change_5d = (vol_second / vol_first - 1.0) * 100.0 * price_dir
                    
            # 6. 建玉(OI)変化率
            oi_series = df_w['bybit_openInterest'].astype(float).replace(0.0, np.nan).dropna()
            oi_change_5d = 0.0
            if len(oi_series) >= 2:
                oi_first = float(oi_series.iloc[0])
                oi_last = float(oi_series.iloc[-1])
                if oi_first > 0:
                    oi_change_5d = (oi_last / oi_first - 1.0) * 100.0 * price_dir
                    
            # 7. 平均ファンディングレート
            fund_series = df_w['bybit_fundingRate'].astype(float).replace(0.0, np.nan).dropna()
            funding_avg_5d = float(fund_series.mean()) if not fund_series.empty else 0.0
            
            # 8. コインベースプレミアム平均
            cb_prem_series = df_w['coinbase_premium'].astype(float).dropna()
            cb_premium_avg_5d = float(cb_prem_series.mean()) if not cb_prem_series.empty else 0.0
            
            # 流動性 & オーバーソールドチェック (ショート禁止判定)
            daily_turnover = float((df_w['bybit_volume'] * df_w['bybit_close']).dropna().mean() * 24)
            is_low_liq = 1 if daily_turnover < 5000000.0 else 0
            is_oversold = 1 if norm_perf_5d <= 0.8 else 0
            ban_short = 1 if (is_low_liq or is_oversold) else 0
            
            metrics_list.append({
                'symbol': symbol,
                'norm_perf_5d': norm_perf_5d,
                'norm_perf_5d_btc': norm_perf_5d_btc,
                'momentum_dir_5d': momentum_dir_5d,
                'max_drawdown_5d': max_drawdown_5d,
                'volume_change_5d': volume_change_5d,
                'oi_change_5d': oi_change_5d,
                'funding_avg_5d': funding_avg_5d,
                'cb_premium_avg_5d': cb_premium_avg_5d,
                'daily_turnover': daily_turnover,
                'is_low_liq': is_low_liq,
                'is_oversold': is_oversold,
                'ban_short': ban_short
            })
            
        if not metrics_list:
            continue
            
        df_daily = pd.DataFrame(metrics_list)
        
        # 各種 Z-score を計算
        df_daily['z_perf_5d'] = get_zscore(df_daily['norm_perf_5d'])
        df_daily['z_perf_5d_btc'] = get_zscore(df_daily['norm_perf_5d_btc'])
        df_daily['z_momentum_5d'] = get_zscore(df_daily['momentum_dir_5d'])
        df_daily['dd_abs'] = df_daily['max_drawdown_5d'].abs()
        df_daily['z_dd_5d'] = get_zscore(df_daily['dd_abs'], invert=True)
        df_daily['z_vol_5d'] = get_zscore(df_daily['volume_change_5d'])
        df_daily['z_oi_5d'] = get_zscore(df_daily['oi_change_5d'])
        df_daily['fund_abs'] = df_daily['funding_avg_5d'].abs()
        df_daily['z_funding_5d'] = get_zscore(df_daily['fund_abs'], invert=True)
        df_daily['z_cb_premium'] = get_zscore(df_daily['cb_premium_avg_5d'])
        
        # 欠損値を 0.0 埋め
        z_cols = ['z_perf_5d', 'z_perf_5d_btc', 'z_momentum_5d', 'z_dd_5d', 'z_vol_5d', 'z_oi_5d', 'z_funding_5d', 'z_cb_premium']
        for col in z_cols:
            df_daily[col] = df_daily[col].fillna(0.0)
            
        # 地合い（market_state）に応じた総合スコアリング
        if market_state == "long_only":
            df_daily['score'] = (
                W_PERF * df_daily['z_perf_5d'] +
                W_PERF_BTC * df_daily['z_perf_5d_btc'] +
                W_MOMENTUM * df_daily['z_momentum_5d'] +
                2.0 * df_daily['z_dd_5d'] + # 本番同様にドローダウンを重く
                W_VOL * df_daily['z_vol_5d'] +
                W_OI * df_daily['z_oi_5d'] +
                W_CB_PREMIUM * df_daily['z_cb_premium']
            )
        else: # short_only
            df_daily['score'] = (
                1.5 * df_daily['z_perf_5d'] + # 本番同様に下落率を重く
                W_PERF_BTC * df_daily['z_perf_5d_btc'] +
                W_MOMENTUM * df_daily['z_momentum_5d'] -
                1.0 * df_daily['z_vol_5d'] + # ショート時は出来高増加でマイナス側（適正）へ振る
                W_CB_PREMIUM * df_daily['z_cb_premium']
            )
            
        # 出力互換用に各取引所のスコアも設定
        df_daily['score_bybit'] = df_daily['score']
        df_daily['score_bitget'] = df_daily['score']
        df_daily['date'] = target_dt.strftime('%Y-%m-%d %H:%M:%S')
        df_daily['market_state'] = market_state
        
        # 保存用にソートして蓄積
        df_daily_sorted = df_daily.sort_values(by='score', ascending=False).reset_index(drop=True)
        all_daily_records.append(df_daily_sorted)
        
        # --- 簡易バックテストシミュレータ ---
        # 翌日11:00の価格情報を取り出すために翌日タイムスタンプを設定
        next_dt = target_dt + timedelta(days=1)
        
        # エントリー銘柄の選定（上位3銘柄）
        selected_trades = []
        if market_state == "long_only":
            # 降順でスコアが高い上位3銘柄
            candidates = df_daily_sorted.head(10) # 候補多めに取得
            count = 0
            for _, row in candidates.iterrows():
                if count >= 3:
                    break
                selected_trades.append({"symbol": row['symbol'], "side": "Long"})
                count += 1
        else: # short_only
            # 昇順でスコアが低い順（マイナスに大きい順）から、ban_shortではない上位3銘柄
            candidates_desc = df_daily_sorted.iloc[::-1] # スコア低い順
            count = 0
            for _, row in candidates_desc.iterrows():
                if count >= 3:
                    break
                if row['ban_short'] == 1:
                    # ショート禁止銘柄はスキップ
                    continue
                selected_trades.append({"symbol": row['symbol'], "side": "Short"})
                count += 1
                
        # 各銘柄の損益率（PnL %）をシミュレート
        trade_pnls = []
        for trade in selected_trades:
            sym = trade["symbol"]
            side = trade["side"]
            
            df_sym = dfs.get(sym)
            if df_sym is not None:
                # 当日11:00時点の価格（始値）
                row_curr = df_sym[df_sym['timestamp'] == target_dt]
                # 翌日11:00時点の価格（終値）
                row_next = df_sym[df_sym['timestamp'] == next_dt]
                
                if not row_curr.empty and not row_next.empty:
                    open_price = float(row_curr['bybit_open'].iloc[0])
                    close_price = float(row_next['bybit_close'].iloc[0])
                    
                    if open_price > 0:
                        if side == "Long":
                            pnl = (close_price - open_price) / open_price
                        else: # Short
                            pnl = (open_price - close_price) / open_price
                        trade_pnls.append(pnl)
                        
        if len(trade_pnls) > 0:
            # 3銘柄への等金額分散として平均を算出
            daily_pnl = sum(trade_pnls) / len(trade_pnls)
            daily_pnls.append(daily_pnl)
            simulated_dates.append(target_dt)
            
    # 全日程ループ終了後
    if not all_daily_records:
        print("Error: No daily scores could be computed.")
        sys.exit(1)
        
    # スコアマスタCSVの結合と保存
    df_master = pd.concat(all_daily_records, ignore_index=True)
    cols = ["date", "symbol", "score", "score_bitget", "score_bybit", "ban_short", "is_oversold", "is_low_liq", "market_state"]
    other_cols = [c for c in df_master.columns if c not in cols]
    df_master = df_master[cols + other_cols]
    
    print(f"Saving selection scores history to {output_path}...")
    df_master.to_csv(output_path, index=False)
    print("SUCCESS: Selection scores database updated!")
    
    # --- バックテスト結果の評価集計 ---
    if len(daily_pnls) > 0:
        print("\n=== Running Backtest Performance Evaluation ===")
        equity = 100.0
        history_rows = []
        
        # 最初の日の取引エントリー
        history_rows.append({
            "time": simulated_dates[0].strftime("%Y-%m-%d %H:%M:%S"),
            "sizes": 1.0,  # 1ユニット分の投資
            "price": equity,
            "high": equity,
            "low": equity
        })
        
        for dt, pnl_rate in zip(simulated_dates[1:], daily_pnls[1:]):
            prev_equity = equity
            equity *= (1.0 + pnl_rate)
            high_val = max(prev_equity, equity)
            low_val = min(prev_equity, equity)
            
            history_rows.append({
                "time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "sizes": 0.0,  # ポジション維持
                "price": equity,
                "high": high_val,
                "low": low_val
            })
            
        # 最終日のクローズ
        history_rows[-1]["sizes"] = -1.0
        
        df_pl_input = pd.DataFrame(history_rows)
        
        if make_mm_pl is not None:
            # make_mm_pl を実行して詳細な統計指標を算出
            # 手数料(片道 0.05% と仮定)を引いて実態に近づける
            res_df, final_pnl, metrics = make_mm_pl(
                df_pl_input, 
                maker_fee=0.0005, 
                taker_fee=0.0005, 
                initial=100.0, 
                has_ordertype=False
            )
            
            # 結果の出力
            print("\n------------------------------------------------")
            print("  Backtest Summary (Equal-Weight Top 3 Portfolio)")
            print("------------------------------------------------")
            print(f"シミュレーション期間: {simulated_dates[0].date()} 〜 {simulated_dates[-1].date()}")
            print(f"対象日数          : {len(simulated_dates)} 日")
            print(f"初期資金          : 100.00 USDT")
            print(f"最終評価額        : {equity:.2f} USDT")
            print(f"実現損益 (PnL)    : {final_pnl:.2f} USDT ({final_pnl:.2f}%)")
            print(f"プロフィットファクター: {metrics['PF']:.4f}")
            print(f"勝率 (日次ベース) : {metrics['win_rate']*100:.2f}%")
            print(f"最大ドローダウン  : {metrics['DD_max']:.2f} USDT ({metrics['DD_per']:.4f}%)")
            print(f"最大含み損        : {metrics['max_unrealized_loss']:.2f} USDT")
            print("------------------------------------------------\n")
        else:
            # 単純集計によるフォールバック
            win_days = sum(1 for pnl in daily_pnls if pnl > 0)
            win_rate = win_days / len(daily_pnls) if len(daily_pnls) > 0 else 0
            cumulative_ret = (equity - 100.0)
            print(f"シミュレーション日数: {len(simulated_dates)} 日")
            print(f"累積リターン        : {cumulative_ret:.2f}%")
            print(f"勝率 (日次ベース)   : {win_rate*100:.2f}%")
            print(f"最終資産            : {equity:.2f} USDT")
            print("Warning: Detailed metrics unavailable as make_mm_pl could not be imported.")
    else:
        print("No trades were simulated. Make sure you have downloaded continuous daily data.")

if __name__ == "__main__":
    main()
