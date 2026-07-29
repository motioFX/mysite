from datetime import datetime, timezone, timedelta
import asyncio
import time
import os
import requests
import pybotters
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md
from rich import print
import sys
from bitget5_44_5backtest_mm import make_mm_pl, Backtest, AirExchange
from config_loader import get_webhook_url


def calc_add_pct(n: int) -> float:
    """Return add-on percentage based on position count."""
    if n <= 3:
        return 0.001  # 0.2%
    if n <= 5:
        return 0.002  # 0.3%
    if n <= 8:
        return 0.01   # 0.4%
    return 0.02       # 0.5%


class logicinstance:
    def crossover(self, x, y):
        return ((x.shift(1) - y.shift(1)) > 0) & ((x.shift(2) - y.shift(2)) <= 0)
    def crossunder(self, x, y):
        return ((x.shift(1) - y.shift(1)) < 0) & ((x.shift(2) - y.shift(2)) >= 0)
    def crossover_t(self, x, y):
        return ((x.shift(1) - y) > 0) & ((x.shift(2) - y) <= 0)
    def crossunder_t(self, x, y):
        return ((x.shift(1) - y) < 0) & ((x.shift(2) - y) >= 0)




    def highest(self, df, period=5):
        maxvalue = df["close"].shift(1).rolling(period).max()
        return maxvalue

    def lowest(self, df, period=5):
        lowvalue = df["close"].shift(1).rolling(period).min()
        return lowvalue


    def make_atr(self, df, span=14):
        high = df['high']
        low = df['low']
        close = df['close']
        # True Range の計算
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # ATR の計算
        atr = tr.ewm(span=span, adjust=False).mean()
        return round(atr, 1)
        # df[f'atr_{span}'] = np.round(atr, 1)
        # return df

    def make_market_profile(self, df, period=100, value_area_pct=0.70, num_bins=50):
        """
        マーケットプロファイルのVAH, VAL, POCを計算する（高速化版）
        
        Parameters:
        -----------
        df : DataFrame
            OHLCデータを含むDataFrame (high, low, close, volumeが必要)
        period : int
            計算期間（デフォルト: 100本）
        value_area_pct : float
            バリューエリアの割合（デフォルト: 0.70 = 70%）
        num_bins : int
            価格帯の分割数（デフォルト: 50）
        
        Returns:
        --------
        DataFrame : VAH, VAL, POC列が追加されたDataFrame
        """
        df = df.copy()
        n = len(df)
        
        # 結果を格納する配列
        poc_arr = np.full(n, np.nan)
        vah_arr = np.full(n, np.nan)
        val_arr = np.full(n, np.nan)
        
        # 最低10本以上あれば計算開始
        min_required = min(10, period)
        
        # 高速化: 配列として事前に取得
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values if 'volume' in df.columns else np.ones(n)
        
        for i in range(min_required, n):
            # 期間内のデータインデックス (現在の足を含める)
            start_idx = max(0, i - period + 1)
            
            # ウィンドウ内のデータ (i番目を含む: start_idx から i まで)
            w_highs = highs[start_idx:i+1]
            w_lows = lows[start_idx:i+1]
            w_volumes = volumes[start_idx:i+1]
            w_len = len(w_highs)
            
            # 価格レンジの決定
            price_high = w_highs.max()
            price_low = w_lows.min()
            
            if price_high == price_low:
                poc_arr[i] = price_high
                vah_arr[i] = price_high
                val_arr[i] = price_low
                continue
            
            # 価格帯を作成
            price_bins = np.linspace(price_low, price_high, num_bins + 1)
            bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
            bin_width = price_bins[1] - price_bins[0]
            
            # ベクトル化: 各足が各ビンに重なるかを計算
            # shape: (w_len, num_bins)
            bin_lows = price_bins[:-1]  # (num_bins,)
            bin_highs = price_bins[1:]  # (num_bins,)
            
            # ブロードキャスト用に形状を調整
            w_highs_2d = w_highs[:, np.newaxis]  # (w_len, 1)
            w_lows_2d = w_lows[:, np.newaxis]    # (w_len, 1)
            w_volumes_2d = w_volumes[:, np.newaxis]  # (w_len, 1)
            
            # 重なり判定: row_low <= bin_high and row_high >= bin_low
            overlap_mask = (w_lows_2d <= bin_highs) & (w_highs_2d >= bin_lows)
            
            # 重なり部分の計算
            overlap_low = np.maximum(w_lows_2d, bin_lows)
            overlap_high = np.minimum(w_highs_2d, bin_highs)
            row_range = np.maximum(w_highs_2d - w_lows_2d, 1e-10)  # ゼロ除算防止
            overlap_ratio = (overlap_high - overlap_low) / row_range
            overlap_ratio = np.clip(overlap_ratio, 0, 1)
            
            # ボリュームプロファイル計算
            volume_contribution = w_volumes_2d * overlap_ratio * overlap_mask
            volume_profile = volume_contribution.sum(axis=0)
            
            # POC（最大ボリュームの価格帯）
            poc_idx = np.argmax(volume_profile)
            poc_price = bin_centers[poc_idx]
            
            # バリューエリアの計算
            total_volume = volume_profile.sum()
            target_volume = total_volume * value_area_pct
            
            # POCから開始して上下に広げる
            value_area_volume = volume_profile[poc_idx]
            lower_idx = poc_idx
            upper_idx = poc_idx
            
            while value_area_volume < target_volume:
                can_go_up = upper_idx < num_bins - 1
                can_go_down = lower_idx > 0
                
                if not can_go_up and not can_go_down:
                    break
                
                up_vol = volume_profile[upper_idx + 1] if can_go_up else -1
                down_vol = volume_profile[lower_idx - 1] if can_go_down else -1
                
                if up_vol >= down_vol and can_go_up:
                    upper_idx += 1
                    value_area_volume += volume_profile[upper_idx]
                elif can_go_down:
                    lower_idx -= 1
                    value_area_volume += volume_profile[lower_idx]
                else:
                    upper_idx += 1
                    value_area_volume += volume_profile[upper_idx]
            
            # VAH, VALを設定
            vah_price = price_bins[upper_idx + 1] if upper_idx < num_bins else price_high
            val_price = price_bins[lower_idx]
            
            poc_arr[i] = poc_price
            vah_arr[i] = vah_price
            val_arr[i] = val_price
        
        # 結果をDataFrameに設定
        df['POC'] = poc_arr
        df['VAH'] = vah_arr
        df['VAL'] = val_arr
        
        # 前方の値を埋める
        df['POC'] = df['POC'].ffill().bfill()
        df['VAH'] = df['VAH'].ffill().bfill()
        df['VAL'] = df['VAL'].ffill().bfill()
        
        return df

    def make_logic(self, df, market_profile_period=720, er_threshold=0.3, strategy_type="range"):  # 1時間足用: 720本 = 30日分 (約1ヶ月)

        # ATRロジック
        atr_period = 10
        df[f'atr_{atr_period}'] = self.make_atr(df, atr_period)
        
        # マーケットプロファイルを追加
        df = self.make_market_profile(df, period=market_profile_period)
        
        # 効率比 (Efficiency Ratio) の計算
        er_period = 10
        direction = abs(df['close'] - df['close'].shift(er_period))
        volatility = df['close'].diff().abs().rolling(er_period).sum()
        df['er'] = (direction / volatility).fillna(0)

        # VAH breakout signal
        df['long_breakout'] = self.crossover(df['close'], df['VAH']) & (df['er'] > er_threshold)
        df['short_breakout'] = self.crossunder(df['close'], df['VAL']) & (df['er'] > er_threshold)
        
        # VAL/VAH range signal (VAL反発 / VAH反発)
        df['long_range'] = self.crossover(df['close'], df['VAL']) & (df['er'] > er_threshold)
        df['short_range'] = self.crossunder(df['close'], df['VAH']) & (df['er'] > er_threshold)
        
        # Select strategy
        if strategy_type == "breakout":
            df['long'] = df['long_breakout']
            df['short'] = df['short_breakout']
        elif strategy_type == "adaptive":
            df['long'] = np.where(df['er'] > er_threshold, df['long_breakout'], df['long_range'])
            df['short'] = np.where(df['er'] > er_threshold, df['short_breakout'], df['short_range'])
        else:
            df['long'] = df['long_range']
            df['short'] = df['short_range']
        
        # Support Low logic: lowest price in market profile period
        df['lowest_support'] = df['low'].rolling(window=market_profile_period, min_periods=10).min().ffill()
        
        # Close signals are handled by trailing SL or coordinator, set to False default
        df['longclose'] = False
        df['shortclose'] = False

        # NaN処理 - dropnaで全データが消えないよう注意
        rows_before = len(df)
        df_cleaned = df.dropna()
        rows_after = len(df_cleaned)
        
        if rows_after == 0 and rows_before > 0:
            # dropnaで全部消えた場合は、bfill/ffillで埋めてから使う
            print("[WARNING] dropna removed all rows, using bfill/ffill instead")
            df = df.bfill().ffill()
        else:
            df = df_cleaned
            df = df.bfill()
        return df


import threading

class send_discord:
    _shared_lock = threading.Lock()
    _shared_buffers = {}
    _shared_timers = {}

    def __init__(self):
        self.webhook_url = get_webhook_url()
        self.bitget_webhook = get_webhook_url("bitget")
        self.bybit_webhook = get_webhook_url("bybit")
        if not self.webhook_url and not self.bitget_webhook and not self.bybit_webhook:
            raise RuntimeError("Discord webhook URL is not configured")
        
        # Initialize Rich console for ANSI color generation compatible with Discord
        from rich.console import Console
        self.console = Console(color_system="standard", force_terminal=True, highlight=False)

        self.lock = send_discord._shared_lock
        self.buffers = send_discord._shared_buffers
        self.timers = send_discord._shared_timers

    def _get_target_webhooks(self, text: str) -> list[str]:
        # sys.platform が win32 (Windows) の場合は win32 用の Webhook を最優先にします
        if sys.platform == 'win32':
            if self.webhook_url:
                return [self.webhook_url]
            if self.bitget_webhook:
                return [self.bitget_webhook]
            if self.bybit_webhook:
                return [self.bybit_webhook]
        else:
            # それ以外 (Linux/VPS環境など) は Bitget 側の Webhook に統一して送信します
            if self.bitget_webhook:
                return [self.bitget_webhook]
            if self.webhook_url:
                return [self.webhook_url]
            if self.bybit_webhook:
                return [self.bybit_webhook]
        return []

    def flush_buffer(self, url):
        with self.lock:
            if url not in self.buffers or not self.buffers[url]:
                return
            entries = self.buffers[url]
            self.buffers[url] = []
            if url in self.timers:
                self.timers[url] = None
        
        merged_ansi = "\n".join(entries)
        
        # Discord limit is 2000 characters. Chunk the message if necessary.
        chunk_limit = 1900
        chunks = []
        current_chunk = []
        current_length = 0

        for line in entries:
            line_len = len(line) + 1  # include newline
            if current_length + line_len > chunk_limit and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_length = line_len
            else:
                current_chunk.append(line)
                current_length += line_len
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        for chunk in chunks:
            discord_message = f"```ansi\n{chunk}\n```"
            try:
                payload = {"content": discord_message}
                response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Failed to send buffered message to {url}:", e)

    def flush_all(self):
        with self.lock:
            urls = list(self.buffers.keys())
        for url in urls:
            self.flush_buffer(url)

    def print_log(self, text):
        # Only color the margin balance and account info lines in green to reduce visual noise
        from rich.markup import escape
        is_balance = any(kw in text for kw in ["証拠金残高", "残高", "Account Info", "balance", "Balance"])
        if is_balance:
            styled_text = f"[bold green]{escape(text)}[/bold green]"
        else:
            styled_text = escape(text)
        
        # Capture ANSI escape sequences generated by Rich console
        with self.console.capture() as capture:
            self.console.print(styled_text, end="")
        ansi_text = capture.get()
        
        # Output to local terminal
        print(ansi_text)
        
        webhooks = self._get_target_webhooks(text)
        
        # If the log message contains backticks, flush buffer and send immediately to avoid breaking markdown formatting
        if "```" in text:
            for url in webhooks:
                self.flush_buffer(url)
                try:
                    payload = {"content": text}
                    response = requests.post(url, json=payload, timeout=10)
                    response.raise_for_status()
                except Exception as e:
                    print(f"Failed to send message to {url}:", e)
            return

        # Buffer regular logs
        import threading
        with self.lock:
            for url in webhooks:
                if url not in self.buffers:
                    self.buffers[url] = []
                self.buffers[url].append(ansi_text)
                
                # Reset timer to bundle messages sent close in time (1.2s window)
                if url in self.timers and self.timers[url] is not None:
                    self.timers[url].cancel()
                
                timer = threading.Timer(1.2, self.flush_buffer, args=[url])
                self.timers[url] = timer
                timer.start()

    def _send_file(self, content_text, file_path, file_name, mime_type):
        webhooks = self._get_target_webhooks(content_text)
        for url in webhooks:
            # Flush existing logs before uploading file to preserve chronological order
            self.flush_buffer(url)
            try:
                payload = {"content": content_text}
                with open(file_path, f"rb") as f:
                    files = {"file": (file_name, f, mime_type)}
                    response = requests.post(url, data=payload, files=files, timeout=30)
                    response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Failed to send file {file_name} to {url}:", e)
                pass

    def plot_kline(self, symbol=""):
        df = pd.read_csv("backtest_data/klines100.csv")
        date = pd.to_datetime(df["timestamp"])
        a_plot = df["close"]
        
        # マーケットプロファイルのデータ
        has_market_profile = all(col in df.columns for col in ['POC', 'VAH', 'VAL'])

        #グラフ - pnlと同じサイズ
        matplotlib.rcParams["timezone"] = "Asia/Tokyo"
        fig = plt.figure(figsize=(8, 3))
        # topを0.95から0.85に下げてタイトル表示エリアを確保
        fig.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.85)
        
        #第1軸 close
        ax1 = fig.add_subplot(1, 1, 1)
        
        # 1日ごとの区切り線 (00:00) を追加
        try:
            d_series = pd.to_datetime(date)
            tz_info = d_series.dt.tz if hasattr(d_series.dt, "tz") else None
            day_starts = pd.date_range(
                start=d_series.iloc[0].normalize(),
                end=d_series.iloc[-1].normalize() + pd.Timedelta(days=1),
                freq="D",
                tz=tz_info
            )
            for day_start in day_starts:
                if d_series.iloc[0] <= day_start <= d_series.iloc[-1]:
                    ax1.axvline(day_start, color="gray", linestyle=":", linewidth=0.8, alpha=0.4, zorder=1)
        except Exception:
            pass

        ax1.plot(date, a_plot, "C0", label="close", linewidth=1.5)
        # 左のY軸（close）の設定
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # マーケットプロファイルのVAH, VAL, POCをプロット
        if has_market_profile:
            poc_plot = df['POC']
            vah_plot = df['VAH']
            val_plot = df['VAL']
            
            # POC（赤い太線）
            ax1.plot(date, poc_plot, color='red', linestyle='-', linewidth=2, 
                     label='POC', alpha=0.8)
            
            # VAH（緑の点線）
            ax1.plot(date, vah_plot, color='green', linestyle='--', linewidth=1.5, 
                     label='VAH', alpha=0.7)
            
            # VAL（青の点線）
            ax1.plot(date, val_plot, color='blue', linestyle='--', linewidth=1.5, 
                     label='VAL', alpha=0.7)
            
            # バリューエリアを塗りつぶし
            ax1.fill_between(date, val_plot, vah_plot, alpha=0.15, color='purple')
            
            # 右側に最新の値を表示
            latest_close = df['close'].iloc[-1]
            latest_poc = df['POC'].iloc[-1]
            latest_vah = df['VAH'].iloc[-1]
            latest_val = df['VAL'].iloc[-1]
            
            # 右端に値を表示（テキストアノテーション）
            ax1.annotate(f'close: {latest_close:.2f}', xy=(1.01, latest_close), 
                        xycoords=('axes fraction', 'data'), fontsize=8, color='C0', va='center')
            ax1.annotate(f'POC: {latest_poc:.2f}', xy=(1.01, latest_poc), 
                        xycoords=('axes fraction', 'data'), fontsize=8, color='red', va='center')
            ax1.annotate(f'VAH: {latest_vah:.2f}', xy=(1.01, latest_vah), 
                        xycoords=('axes fraction', 'data'), fontsize=8, color='green', va='center')
            ax1.annotate(f'VAL: {latest_val:.2f}', xy=(1.01, latest_val), 
                        xycoords=('axes fraction', 'data'), fontsize=8, color='blue', va='center')
        
        ax1.legend(loc='upper left', fontsize=8)
        ax1.set_ylabel("Price [USDT]")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(md.DateFormatter("%m/%d %H:%M"))
        fig.autofmt_xdate(rotation=10)
        
        plt.close()  # 不要な図を閉じる
        os.makedirs("backtest_data", exist_ok=True)
        fig.savefig("backtest_data/kline_img.jpg", format='jpg', dpi=80)

        self._send_file(f"{symbol} klines with Market Profile", "backtest_data/kline_img.jpg", "kline_img.jpg", "image/jpeg")

    def plot_backtest(self, label="MP", csv_file="backtest_data/klines100.csv", img_file=None, symbol=""):
        """バックテスト結果をプロット
        
        Args:
            label: グラフタイトル用ラベル（例: "MP120"）
            csv_file: 読み込むCSVファイル名
            img_file: 出力画像ファイル名（Noneの場合は自動生成）
        """
        if img_file is None:
            img_file = f"backtest_data/backtest_{label}_img.jpg"
        
        df = pd.read_csv(csv_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        date = df['timestamp']
        a_plot = df['close']
        b_plot = df['pnl']

        #グラフ
        matplotlib.rcParams["timezone"] = "Asia/Tokyo"
        fig = plt.figure(figsize=(8, 3))
        # topを0.95から0.85に下げてタイトル表示エリアを確保
        fig.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.85)
        # 第1Y軸（close）
        ax1 = fig.add_subplot(1, 1, 1)

        # 1日ごとの区切り線 (00:00) を追加
        try:
            d_series = pd.to_datetime(date)
            tz_info = d_series.dt.tz if hasattr(d_series.dt, "tz") else None
            day_starts = pd.date_range(
                start=d_series.iloc[0].normalize(),
                end=d_series.iloc[-1].normalize() + pd.Timedelta(days=1),
                freq="D",
                tz=tz_info
            )
            for day_start in day_starts:
                if d_series.iloc[0] <= day_start <= d_series.iloc[-1]:
                    ax1.axvline(day_start, color="gray", linestyle=":", linewidth=0.8, alpha=0.4, zorder=1)
        except Exception:
            pass

        ax1.plot(date, a_plot, "C0", label="close")
        if 'exec_buy_price' in df.columns:
            buy_mask = df['exec_buy_price'].notna()
            if buy_mask.any():
                ax1.scatter(
                    df.loc[buy_mask, 'timestamp'],
                    df.loc[buy_mask, 'exec_buy_price'],
                    marker='^',
                    color='green',
                    label='exec buy',
                    zorder=5,
                    s=40,
                )
        if 'exec_sell_price' in df.columns:
            sell_mask = df['exec_sell_price'].notna()
            if sell_mask.any():
                ax1.scatter(
                    df.loc[sell_mask, 'timestamp'],
                    df.loc[sell_mask, 'exec_sell_price'],
                    marker='v',
                    color='red',
                    label='exec sell',
                    zorder=5,
                    s=40,
                )
        ax1.set_ylabel("close [USDT]")
        ax1.grid(True)
        
        # ボリュームプロファイル（POC, VAH, VAL）をプロット
        has_market_profile = all(col in df.columns for col in ['POC', 'VAH', 'VAL'])
        if has_market_profile:
            poc_plot = df['POC']
            vah_plot = df['VAH']
            val_plot = df['VAL']
            
            # POC（赤い太線）
            ax1.plot(date, poc_plot, color='red', linestyle='-', linewidth=1.5, 
                     label='POC', alpha=0.7)
            
            # VAH（緑の点線）
            ax1.plot(date, vah_plot, color='green', linestyle='--', linewidth=1, 
                     label='VAH', alpha=0.6)
            
            # VAL（青の点線）
            ax1.plot(date, val_plot, color='blue', linestyle='--', linewidth=1, 
                     label='VAL', alpha=0.6)
            
            # バリューエリアを薄く塗りつぶし
            ax1.fill_between(date, val_plot, vah_plot, alpha=0.1, color='purple')
        
        # 第2Y軸（pnl）
        ax2 = ax1.twinx()
        ax2.plot(date, b_plot, "C1", label="pl")
        ax2.set_ylabel("pnl [USDT]")
        ax2.set_xlabel("Date")
        ax2.xaxis.set_major_locator(md.AutoDateLocator())
        fig.autofmt_xdate(rotation=10)
        
        # タイトルにラベルと最終PnLを表示
        final_pnl = df['pnl'].iloc[-1] if 'pnl' in df.columns else 0
        ax1.set_title(f"{symbol} {label} | Final PnL: {final_pnl:.2f} USDT", fontsize=10)
        
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='lower left')
        ax2.grid(False)
        plt.close()
        fig.savefig(img_file, format='jpg', dpi=80)

        self._send_file(f"{symbol} backtest pnl ({label})", img_file, f"backtest_{label}.jpg", "image/jpeg")
        
        return img_file

    def print_logs(self, df, jpy_onhand_amount, onhand_amount, max_lot, target_position_value=None, trade_side="long"):
        # Empty DataFrame check - more robust
        if df is None or df.empty or len(df) == 0:
            self.print_log("ログ出力スキップ: データフレームが空です")
            return
        
        self.print_log(f"1 timestamp : {df['timestamp'].iloc[-1]}")
        time.sleep(1)

        if str(trade_side).lower() == "short":
            short_val = df['short'].iloc[-1] if 'short' in df.columns else False
            shortclose_val = df['shortclose'].iloc[-1] if 'shortclose' in df.columns else (df['stshortclose'].iloc[-1] if 'stshortclose' in df.columns else False)
            self.print_log(f"close:{df['close'].iloc[-1]} short:{short_val} shortclose:{shortclose_val}")
        else:
            long_val = df['long'].iloc[-1] if 'long' in df.columns else False
            longclose_val = df['longclose'].iloc[-1] if 'longclose' in df.columns else (df['stlongclose'].iloc[-1] if 'stlongclose' in df.columns else False)
            self.print_log(f"close:{df['close'].iloc[-1]} long:{long_val} longclose:{longclose_val}")
        time.sleep(1)

        price = df['close'].iloc[-1]
        if target_position_value is not None and target_position_value > 0:
            optimal_lot = target_position_value / price
            optimal_lot_usdt = target_position_value
        else:
            optimal_lot = jpy_onhand_amount / price / max_lot * 100
            optimal_lot_usdt = optimal_lot * price

        optimal_lot = round(optimal_lot, 5)
        onhand_amount_usdt = onhand_amount * price
        self.print_log(f"best lot : {optimal_lot_usdt:.2f} USDT, onhand_amount : {onhand_amount_usdt:.2f} USDT")
        time.sleep(1)
        self.print_log("-----------------------------------------")
        time.sleep(1)


discord = send_discord()
# Bitget用PnL計算クラス
class PnLCalculator:
    def __init__(self, apis_config, rest_api_url=None, symbol='BTCUSDT', mode='demo'):
        self.apis = apis_config
        self.mode = mode
        self.symbol = symbol

    async def calculate_pnl(self, days_back=30, symbol='BTCUSDT', symbols=None):
        """
        Bitgetの約定履歴から損益(PnL)を計算してDataFrameとして返します。
        """
        fills = await self.get_bitget_trade_history(self.apis, product_type="USDT-FUTURES", days_back=days_back)
        if not fills:
            discord.print_log("Bitget取引履歴が見つかりませんでした（新規口座、または最近の取引がない可能性があります）")
            return None
            
        df = pd.DataFrame(fills)
        combined = self.calculate_bitget_pnl_from_df(df)
        return combined

    async def get_bitget_trade_history(self, apis_bitget, product_type="USDT-FUTURES", days_back=30):
        """Bitgetの約定履歴（fills）を取得"""
        # apis_bitget のフォーマット調整
        try:
            import bitget5_44_2api_dual
            bitget_mode = getattr(bitget5_44_2api_dual, 'bitget_mode', 'live')
        except ImportError:
            bitget_mode = 'live'

        api_key_name = 'bitget_demo' if bitget_mode in ('paper', 'demo') and 'bitget_demo' in apis_bitget else 'bitget'
        client_apis = {'bitget': apis_bitget.get(api_key_name) or apis_bitget.get('bitget') or apis_bitget}
        
        headers = {"paptrading": "1"} if bitget_mode in ('paper', 'demo') else {}
        base_url = "https://api.bitget.com"
        
        async with pybotters.Client(apis=client_apis, base_url=base_url, headers=headers) as client:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            all_trades = []
            current_end = end_time
            
            # 7日間ずつ分割
            while current_end > start_time:
                current_start = max(current_end - timedelta(days=7), start_time)
                
                params = {
                    'productType': product_type,
                    'startTime': int(current_start.timestamp() * 1000),
                    'endTime': int(current_end.timestamp() * 1000),
                    'limit': '100'
                }
                
                # API v2 約定履歴エンドポイント
                endpoint = '/api/v2/mix/order/fills'
                
                try:
                    resp = await client.get(endpoint, params=params, headers=headers)
                    data = await resp.json()
                    
                    if data and data.get('code') == '00000' and data.get('data'):
                        res_data = data['data']
                        if isinstance(res_data, dict):
                            fills = res_data.get('fillList', [])
                        elif isinstance(res_data, list):
                            fills = res_data
                        else:
                            fills = []
                        all_trades.extend(fills)
                except Exception:
                    pass
                    
                current_end = current_start
                await asyncio.sleep(0.05)
                
            if all_trades:
                # 重複排除とソート（tradeIdで判定）
                all_trades.sort(key=lambda x: int(x.get('cTime', 0)))
                unique_trades = []
                seen_ids = set()
                for trade in all_trades:
                    tid = trade.get('tradeId')
                    if tid and tid not in seen_ids:
                        unique_trades.append(trade)
                        seen_ids.add(tid)
                return unique_trades
            else:
                return []

    def calculate_bitget_pnl_from_df(self, df):
        """Bitget の fills データフレームから PNL を計算します。"""
        if df.empty:
            return None
        df = df.copy()
        if 'cTime' in df.columns:
            df['execTime'] = pd.to_datetime(pd.to_numeric(df['cTime']), unit='ms')
        else:
            return None
        df = df.sort_values('execTime')
        
        # もし 'size' がなく 'baseVolume' があればマッピング
        if 'size' not in df.columns and 'baseVolume' in df.columns:
            df['size'] = df['baseVolume']
            
        # もし 'fee' がなく 'feeDetail' があれば、そこから合計手数料を抽出
        if 'fee' not in df.columns and 'feeDetail' in df.columns:
            def get_fee_from_detail(detail):
                if not detail or not isinstance(detail, list):
                    return 0.0
                total_fee = 0.0
                for d in detail:
                    if isinstance(d, dict):
                        fee_str = d.get('totalFee')
                        if fee_str:
                            try:
                                total_fee += float(fee_str)
                            except ValueError:
                                pass
                return total_fee
            df['fee'] = df['feeDetail'].apply(get_fee_from_detail)
        elif 'fee' not in df.columns:
            df['fee'] = 0.0

        # 数値型に変換
        for col in ('price', 'size', 'fee', 'profit'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                
        df['realized_pnl'] = 0.0
        df['cumulative_pnl'] = 0.0
        
        # APIが 'profit' (実現損益) フィールドを返している場合は、それを利用して直接計算する
        if 'profit' in df.columns:
            df['realized_pnl'] = df['profit'] + df['fee']
            df['cumulative_pnl'] = df['realized_pnl'].cumsum()
            return df
            
        # シンボルごとにグループ化して PNL を計算
        dfs = []
        for sym, group in df.groupby('symbol'):
            group = group.copy().sort_values('execTime')
            entries = [] # ポジションプールの管理 [{'price': price, 'qty': qty}]
            EPS = 1e-8
            
            for idx, row in group.iterrows():
                side = str(row['side']).lower() # "buy" or "sell"
                price = float(row['price'])
                qty = float(row['size'])
                fee = abs(float(row['fee']))
                
                realized = 0.0
                
                # entries 内の現在のポジション数量を合計
                current_qty = sum(e['qty'] for e in entries) # 正ならロング、負ならショート
                
                if side == 'buy':
                    # 買い取引
                    if current_qty < -EPS:
                        # 既存のショートポジションを決済する
                        remaining_qty = qty
                        while remaining_qty > 0 and entries:
                            entry = entries[0]
                            entry_qty = abs(entry['qty'])
                            close_qty = min(entry_qty, remaining_qty)
                            
                            # ショート決済の損益: (ショートエントリー価格 - 決済買い価格) * 決済数量
                            realized += (entry['price'] - price) * close_qty
                            
                            remaining_qty -= close_qty
                            entry['qty'] += close_qty # ショート（負）をカバー
                            if abs(entry['qty']) < EPS:
                                entries.pop(0)
                        
                        # 手数料を引く
                        realized -= fee
                        # 余った買い数量はロングとして追加
                        if remaining_qty > EPS:
                            entries.append({'price': price, 'qty': remaining_qty})
                    else:
                        # 新規ロング追加
                        entries.append({'price': price, 'qty': qty})
                        realized = -fee # 手数料のみマイナス
                else:
                    # 売り取引
                    if current_qty > EPS:
                        # 既存のロングポジションを決済する
                        remaining_qty = qty
                        while remaining_qty > 0 and entries:
                            entry = entries[0]
                            entry_qty = entry['qty']
                            close_qty = min(entry_qty, remaining_qty)
                            
                            # ロング決済の損益: (決済売り価格 - ロングエントリー価格) * 決済数量
                            realized += (price - entry['price']) * close_qty
                            
                            remaining_qty -= close_qty
                            entry['qty'] -= close_qty
                            if abs(entry['qty']) < EPS:
                                entries.pop(0)
                                
                        realized -= fee
                        # 余った売り数量はショートとして追加
                        if remaining_qty > EPS:
                            entries.append({'price': price, 'qty': -remaining_qty})
                    else:
                        # 新規ショート追加
                        entries.append({'price': price, 'qty': -qty})
                        realized = -fee
                        
                group.at[idx, 'realized_pnl'] = realized
                
            group['cumulative_pnl'] = group['realized_pnl'].cumsum()
            dfs.append(group)
            
        if not dfs:
            return None
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('execTime').reset_index(drop=True)
        combined['cumulative_pnl'] = combined['realized_pnl'].cumsum()
        return combined

    def plot_pnl(self, df, save_path="backtest_data/pnl_graph.jpg", label="Bitget"):
        if df is None:
            print("No data to plot")
            return

        # DataFrameが空の場合
        if df.empty:
            discord.print_log("取引履歴が空です。グラフを描画しません。")
            return

        # 必要なカラムが存在するかチェック
        required_columns = ['execTime', 'realized_pnl', 'cumulative_pnl']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            discord.print_log(f"必要なカラムが見つかりません: {missing_columns}")
            discord.print_log(f"利用可能なカラム: {list(df.columns)}")
            return

        if len(df) == 1:
            discord.print_log("取引データが1件のみです。初期取引の情報を表示します。")
            discord.print_log(f"取引時刻: {df['execTime'].iloc[0]}")
            discord.print_log(f"取引損益: {df['realized_pnl'].iloc[0]} USDT")
            discord.print_log(f"累積損益: {df['cumulative_pnl'].iloc[0]} USDT")
            discord.print_log("今後取引が増えると、グラフ表示に切り替わります。")
            return

        # 損益がすべて0の場合はグラフを描画しない
        if df['realized_pnl'].sum() == 0:
            discord.print_log("取引損益が0のため、グラフを描画しません")
            return

        fig,ax1 = plt.subplots(figsize=(8, 3))
        time_diff = df['execTime'].diff().median()
        bar_width = time_diff * 0.5 if pd.notnull(time_diff) else pd.Timedelta(minutes=10)
        ax1.bar(df['execTime'], df['realized_pnl'], 
                width=bar_width,  
                alpha=0.3, 
                color='blue', 
                label='Individual PNL')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Individual Net Profit (USDT)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.tick_params(axis='x', rotation=10)

        ax2 = ax1.twinx()
        ax2.plot(df['execTime'], df['cumulative_pnl'], color='green', linewidth=2, label='Cumulative PNL')
        ax2.set_ylabel('Cumulative Profit (USDT)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        total_realized_pnl = df['realized_pnl'].sum()
        plt.title(f'{label} Total Realized PnL: {total_realized_pnl:.2f} USDT', fontsize=13)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, format='jpg', dpi=80)
        plt.close()
        # print(f"Graph saved: {save_path}")

        discord._send_file(f"{label} PNL Graph", save_path, f"{label.lower()}_pnl_graph.jpg", "image/jpeg")


class PositionSizer:
    fibolot_dict = {
        0: 1, 1: 2, 2: 3, 3: 5, 4: 8, 5: 13, 6: 21, 7: 21, 8: 21, 9: 21, 10: 21
    }

    cumulative_position_size_dict = {}

    @staticmethod
    def build_cumulative_dict():
        total_long = 0
        for i in range(0, 11):
            total_long += PositionSizer.fibolot_dict.get(i, 21)
            PositionSizer.cumulative_position_size_dict[i] = total_long

    @staticmethod
    def get_stage_from_position_size(pos_size: float) -> int:
        PositionSizer.build_cumulative_dict()
        if pos_size > 0:
            for stage in range(0, 11):
                if pos_size <= PositionSizer.cumulative_position_size_dict[stage]:
                    return max(0, stage + 1)
            return 10
        else:
            return 0  # ポジションゼロ

    @staticmethod
    def get_next_lot(pos_size: float) -> int:
        """
        累積ポジションサイズに基づいて、次に積むべきロットサイズを返す
        :param pos_size: 累積ポジションサイズ（ロット単位）
        :return: 次に積むべき倍率
        """
        # pos_size=0 なら stage=0 -> fibolot_dict[0]=1
        # pos_size=1 なら stage=1 -> fibolot_dict[1]=2
        stage = PositionSizer.get_stage_from_position_size(pos_size)
        return PositionSizer.fibolot_dict.get(stage, 1)





class MPStrategy(Backtest):
    """Grid Trade Strategy Backtest class.
    
    Single Position (VAL Entry), Trailing Stop Loss, VAL * 0.99 Stop Loss floor.
    """

    @staticmethod
    def fibolot(n):
        fibolot_dict = {
            0:0,1:1,2:2,3:3,4:5,5:8,6:13,7:21,8:34,9:55,10:89,
            -1:-1,-2:-2,-3:-3,-4:-5,-5:-8,-6:-13,-7:-21,-8:-34,-9:-55,-10:-89,
        }
        return fibolot_dict.get(n, 0)

    def __init__(self, air_exchange, df, size, columns, side_mode="long", atr_period=10, atr_tp_multi=3, er_threshold=0.3, strategy_type="range", sl_margin_pct=1.0):
        super().__init__(air_exchange=air_exchange, df=df, columns=columns)
        self.size = size
        self.side_mode = side_mode
        self.atr_period = atr_period
        self.atr_tp_multi = atr_tp_multi  # Trailing offset multiplier (ATR * multi)
        self.er_threshold = er_threshold
        self.strategy_type = strategy_type
        self.sl_margin_pct = sl_margin_pct
        self.mean_prices = []
        self.stlong = []
        self.stlongclose = []
        self.stshort = []
        self.stshortclose = []
        self.max_price_since_entry = None
        self.min_price_since_entry = None
        self.crossed_vah = False
        self.crossed_val = False

    def action(self):
        position = np.float64(self._get_position()["qty"])
        mean = self._get_position()["avgEntry"]
        self.mean_prices.append(mean)

        close = self.array[self.column_dic["close"]]
        vah = self.array[self.column_dic["VAH"]]
        val = self.array[self.column_dic["VAL"]]
        poc = self.array[self.column_dic["POC"]]
        atr = self.array[self.column_dic[f"atr_{self.atr_period}"]]

        long_signal = bool(self.array[self.column_dic["long"]]) if "long" in self.column_dic else False
        short_signal = bool(self.array[self.column_dic["short"]]) if "short" in self.column_dic else False

        stlong = False
        stshort = False
        stlongclose = False
        stshortclose = False

        is_short_mode = str(self.side_mode).lower() == "short"

        if is_short_mode:
            # ============================================================
            # ショート用 決済ロジック
            # ============================================================
            if position < 0:
                if self.min_price_since_entry is None or self.min_price_since_entry == 0.0:
                    self.min_price_since_entry = min(mean, close)
                    self.crossed_val = False
                else:
                    self.min_price_since_entry = min(self.min_price_since_entry, close)

                if getattr(self, 'strategy_type', 'range') == 'breakout':
                    base_line = val
                else:
                    if close < val:
                        self.crossed_val = True
                    base_line = val if getattr(self, 'crossed_val', False) else vah

                margin_pct = getattr(self, 'sl_margin_pct', 1.0)
                base_sl = base_line * (1.0 + margin_pct / 100.0)
                trailing_sl = self.min_price_since_entry * (1.0 + margin_pct / 100.0)
                sl_price = min(base_sl, trailing_sl)

                atr_tp_multi = getattr(self, 'atr_tp_multi', 1.5)
                if self.min_price_since_entry < mean - (atr * 0.5):
                    atr_trailing_tp = self.min_price_since_entry + (atr * atr_tp_multi)
                    sl_price = min(sl_price, atr_trailing_tp)

                if close > sl_price:
                    stshortclose = True
                    self._cancel_all_orders()
                    self._market_order(size=-position)
                    self.min_price_since_entry = None
                    self.crossed_val = False
            else:
                self.min_price_since_entry = None
                self.crossed_val = False

            # ショート用 エントリーロジック
            if not stshortclose:
                if position == 0:
                    if short_signal:
                        stshort = True
                        self._cancel_all_orders()
                        self._market_order(size=-self.size)
                        self.min_price_since_entry = close
        else:
            # ============================================================
            # ロング用 決済ロジック
            # ============================================================
            if position > 0:
                if self.max_price_since_entry is None or self.max_price_since_entry == 0.0:
                    self.max_price_since_entry = max(mean, close)
                    self.crossed_vah = False
                else:
                    self.max_price_since_entry = max(self.max_price_since_entry, close)
                
                if not hasattr(self, 'crossed_vah'):
                    self.crossed_vah = False
                    
                if getattr(self, 'strategy_type', 'range') == 'breakout':
                    base_line = vah
                else:
                    if close > vah:
                        self.crossed_vah = True
                    base_line = vah if getattr(self, 'crossed_vah', False) else val

                margin_pct = getattr(self, 'sl_margin_pct', 1.0)
                base_sl = base_line * (1.0 - margin_pct / 100.0)
                trailing_sl = self.max_price_since_entry * (1.0 - margin_pct / 100.0)
                sl_price = max(base_sl, trailing_sl)

                atr_tp_multi = getattr(self, 'atr_tp_multi', 1.5)
                if self.max_price_since_entry > mean + (atr * 0.5):
                    atr_trailing_tp = self.max_price_since_entry - (atr * atr_tp_multi)
                    sl_price = max(sl_price, atr_trailing_tp)

                if close < sl_price:
                    stlongclose = True
                    self._cancel_all_orders()
                    self._market_order(size=-position)
                    self.max_price_since_entry = None
                    self.crossed_vah = False
            else:
                self.max_price_since_entry = None
                self.crossed_vah = False

            # ロング用 エントリーロジック
            if not stlongclose:
                if position == 0:
                    if long_signal:
                        stlong = True
                        self._cancel_all_orders()
                        self._market_order(size=self.size)
                        self.max_price_since_entry = close

        self.stlong.append(stlong)
        self.stlongclose.append(stlongclose)
        self.stshort.append(stshort)
        self.stshortclose.append(stshortclose)

    def run(self):
        super().run()
        self.mean_prices = pd.Series(self.mean_prices, index=self.df.index)
        self.stlong = pd.Series(self.stlong, index=self.df.index)
        self.stlongclose = pd.Series(self.stlongclose, index=self.df.index)
        self.stshort = pd.Series(self.stshort, index=self.df.index)
        self.stshortclose = pd.Series(self.stshortclose, index=self.df.index)

    def get_data(self):
        return pd.DataFrame({
            'timestamp': pd.to_datetime(self.df.index),
            'stlong': self.stlong,
            'stlongclose': self.stlongclose,
            'stshort': self.stshort,
            'stshortclose': self.stshortclose,
            'mean_price': self.mean_prices
        })

class backtester:
    def run_backtest(self, df, lot, data_equity, side_mode="long", mp_period=120, atr_tp_multi=3, er_threshold=0.3, bybit_exec_history=None, bybit_lot_size=None, strategy_type="range", sl_margin_pct=1.0):
        """Market Profile期間を指定してバックテストを実行（新ロジック）
        
        Args:
            df: OHLCデータ（make_logicで処理済み）
            lot: ロットサイズ
            data_equity: 現在の資産額
            side_mode: "long", "short", "both"
            mp_period: マーケットプロファイル期間（デフォルト120）
            atr_tp_multi: Stage1 TP用ATR倍率（デフォルト3）
            bybit_exec_history: Bybitの約定履歴（指定された場合、Bitgetの終値にマッピングして損益を計算します）
            bybit_lot_size: Bybitのバックテストで使われたロットサイズ
        """
        # Check for empty DataFrame before processing
        if df is None or df.empty:
            print("エラー: データフレームが空です。APIからデータを取得できませんでした。")
            discord.print_log("バックテスト失敗: ローソク足データが取得できませんでした。APIキーを確認してください。")
            empty_df = pd.DataFrame(columns=['timestamp', 'close', 'PL_graph', 'pnl', 'des'])
            return empty_df
        
        import copy
        df_sorted = df.copy()
        df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"])
        
        # Drop columns if they already exist to avoid duplicate column name suffixes on merge
        for col in ['PL_graph', 'pnl', 'exec_buy_price', 'exec_sell_price']:
            if col in df_sorted.columns:
                df_sorted = df_sorted.drop(columns=[col])
        
        self.exec_history = []
        
        if bybit_exec_history is not None:
            # Bybitの取引履歴（タイミング・サイズ）をBitgetの終値にマッピング
            air_exchange = AirExchange()
            
            # タイムスタンプから終値へのマッピングを作成
            price_map = {}
            for idx, row in df_sorted.iterrows():
                ts = pd.to_datetime(row['timestamp'])
                price_map[ts] = float(row['close'])
                
            scale = 1.0
            if bybit_lot_size is not None and bybit_lot_size > 0 and lot > 0:
                scale = lot / bybit_lot_size
                
            for entry in bybit_exec_history:
                executed = copy.copy(entry)
                entry_ts = pd.to_datetime(entry["timestamp"])
                
                # スケールされたサイズを適用
                executed["size"] = entry["size"] * scale
                
                if entry_ts in price_map:
                    executed["price"] = price_map[entry_ts]
                else:
                    # 最も近いタイムスタンプを探して補完
                    closest_ts = min(price_map.keys(), key=lambda t: abs(t - entry_ts))
                    executed["price"] = price_map[closest_ts]
                    
                executed["timestamp"] = entry_ts
                air_exchange.exec_history.append(executed)
                
            self.exec_history = air_exchange.exec_history
            exec_df = pd.DataFrame(air_exchange.exec_history)
        else:
            # 通常のBybitバックテスト
            air_exchange = AirExchange()
            btc_price = df_sorted['close'].iloc[-1]
            size_in_usdt = lot * btc_price
            
            # MPStrategy（新ロジック）で実行
            bt = MPStrategy(
                air_exchange=air_exchange, 
                df=df_sorted, 
                size=size_in_usdt,
                columns=["open", "atr_10", "POC", "VAH", "VAL", "long", "short", "er", "lowest_support", "close"],
                side_mode=side_mode,
                atr_period=10,
                atr_tp_multi=atr_tp_multi,
                er_threshold=er_threshold,
                strategy_type=strategy_type,
                sl_margin_pct=sl_margin_pct,
            )
            bt.run()
    
            bt_data = bt.get_data()
            bt_data['timestamp'] = pd.to_datetime(bt_data['timestamp'])
            bt_cols = ['timestamp', 'stlong', 'stlongclose', 'stshort', 'stshortclose']
            bt_cols_exist = [c for c in bt_cols if c in bt_data.columns]
            df_sorted = pd.merge(df_sorted, bt_data[bt_cols_exist], on='timestamp', how='left')
            
            self.exec_history = bt.air_exchange.exec_history
            exec_df = pd.DataFrame(bt.air_exchange.exec_history)
    
        if not exec_df.empty:
            exec_df["timestamp"] = pd.to_datetime(exec_df["timestamp"])
            buy_exec = exec_df[exec_df["size"] > 0][["timestamp", "price"]].copy()
            buy_exec = buy_exec.groupby("timestamp").last().rename(columns={"price": "exec_buy_price"})
            sell_exec = exec_df[exec_df["size"] < 0][["timestamp", "price"]].copy()
            sell_exec = sell_exec.groupby("timestamp").last().rename(columns={"price": "exec_sell_price"})
            exec_prices = buy_exec.join(sell_exec, how="outer").reset_index()
            df_sorted = pd.merge(df_sorted, exec_prices, on="timestamp", how="left")
        else:
            df_sorted["exec_buy_price"] = np.nan
            df_sorted["exec_sell_price"] = np.nan
    
        order_df = exec_df.rename(columns = {"size":"sizes", "timestamp":"time"})
        metrics = {'DD_max': 0, 'DD_per': 0, 'max_unrealized_loss': 0, 'win_rate': 0, 'PF': float('inf'), 'trade_count': 0}
        
        # 資産初期値の設定 (無効な場合は100にフォールバック)
        initial_capital = data_equity if (data_equity is not None and data_equity > 0) else 100.0
        
        if not order_df.empty:
            # make_mm_pl expects high/low列があるので無い場合はpriceで補完
            if "high" not in order_df.columns:
                order_df["high"] = order_df["price"]
            if "low" not in order_df.columns:
                order_df["low"] = order_df["price"]
            pl, _, metrics = make_mm_pl(order_df, maker_fee=0.0002, taker_fee=0.0006, initial=initial_capital, has_ordertype=True)
            pl1 = pl.rename(columns={'time': 'timestamp'})
            df_sorted = pd.merge(df_sorted, pl1[['timestamp', 'PL_graph']], on='timestamp', how='left')
            df_sorted['PL_graph'] = df_sorted['PL_graph'].ffill()
        else:
            # print("order_df is empty. Skipping make_mm_pl and related processes.")
            df_sorted['PL_graph'] = initial_capital
        
        df_sorted['pnl'] = df_sorted['PL_graph']
        self.last_metrics = metrics
    
        return df_sorted 


def resample_candles(df, interval_minutes):
    """60分足データを指定分足にリサンプリング
    
    Args:
        df: 60分足のDataFrame（timestamp, open, high, low, close, volume, turnover）
        interval_minutes: 目標の分足（例: 120, 180）
    
    Returns:
        リサンプリングされたDataFrame
    """
    if interval_minutes <= 60:
        return df.copy()
    
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy = df_copy.sort_values('timestamp').reset_index(drop=True)
    
    # interval_minutesでリサンプリング
    rule = f'{interval_minutes}min'  # 直接分単位で指定（'T'は非推奨）
    df_copy = df_copy.set_index('timestamp')
    
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    if 'turnover' in df_copy.columns:
        agg_dict['turnover'] = 'sum'
    
    resampled = df_copy.resample(rule, closed='left', label='left').agg(agg_dict)
    resampled = resampled.dropna(subset=['open', 'close'])
    resampled = resampled.reset_index()
    
    return resampled


def run_interval_comparison(df_60m, lot, data_equity, side_mode="long"):
    """複数の時間足 × 戦略タイプ(range/breakout) × MP期間 × ER閾値でバックテストを実行し、最良の組み合わせを返す
    
    Args:
        df_60m: 60分足のDataFrame（make_logic適用前の生データ推奨、適用済みでも可）
        lot: ロットサイズ
        data_equity: 初期資金
        side_mode: トレード方向 ("long" or "short")
    
    Returns:
        (results, best_strategy, best_interval, best_mp, best_er, best_margin)
    """
    logic = logicinstance()
    bt_instance = backtester()
    results = {}
    fixed_initial_equity = 100.0
    os.makedirs("backtest_data", exist_ok=True)
    
    strategy_types = ["range", "breakout"]
    intervals = [60, 120, 180]
    mp_periods = [12, 24, 36, 48, 60, 72, 96, 120, 144, 168]
    er_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # 生データから計算済みカラムを除去
    base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if 'turnover' in df_60m.columns:
        base_cols.append('turnover')
    df_raw = df_60m[base_cols].copy()
    
    for strat in strategy_types:
        discord.print_log(f"\n====== 戦略: {strat.upper()} の最適化を開始 ======")
        for interval in intervals:
            interval_label = f"{interval}m"
            
            # リサンプリング
            df_interval = resample_candles(df_raw, interval)
            
            if df_interval is None or df_interval.empty or len(df_interval) < 30:
                continue
            
            for mp_period in mp_periods:
                atr_multi = 1.5  # 建値ストップを発動させる利益幅（ATR倍率）
                
                # MP期間ごとにmake_logicを計算 (ERフィルターは0で無効化して取得)
                df_copy = df_interval.copy()
                df_copy = logic.make_logic(df_copy, market_profile_period=mp_period, er_threshold=0, strategy_type=strat)
                
                for er_th in er_thresholds:
                    # 効率比ER閾値を適用
                    df_run = df_copy.copy()
                    if strat == "breakout":
                        df_run['long'] = df_run['long_breakout'] & (df_run['er'] > er_th)
                    else:
                        df_run['long'] = df_run['long_range']
                        
                    label = f"{strat}_{interval_label}_MP{mp_period}_ER{er_th}"
                    
                    result_df = bt_instance.run_backtest(
                        df=df_run,
                        lot=lot,
                        data_equity=fixed_initial_equity,
                        side_mode=side_mode,
                        mp_period=mp_period,
                        atr_tp_multi=atr_multi,
                        er_threshold=er_th,
                        strategy_type=strat,
                        sl_margin_pct=1.0
                    )
                    
                    metrics = getattr(bt_instance, 'last_metrics', {})
                    csv_file = f"backtest_data/klines100_{label}.csv"
                    result_df.to_csv(csv_file, index=False)
                    
                    final_pnl = result_df['pnl'].iloc[-1] if not result_df.empty and 'pnl' in result_df.columns else 0
                    
                    results[label] = {
                        'strategy_type': strat,
                        'df': result_df,
                        'final_pnl': final_pnl,
                        'interval': interval,
                        'mp_period': mp_period,
                        'er_threshold': er_th,
                        'atr_multi': atr_multi,
                        'DD_max': metrics.get('DD_max', 0),
                        'DD_per': metrics.get('DD_per', 0),
                        'max_unrealized_loss': metrics.get('max_unrealized_loss', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'PF': metrics.get('PF', float('inf')),
                        'trade_count': metrics.get('trade_count', 0)
                    }
                    print(f"{label}: PnL={final_pnl:.2f} | DD={metrics.get('DD_max', 0):.2f} | 取引={metrics.get('trade_count', 0)}")
                    time.sleep(0.1)  # VPSのCPU負荷を抑えるための微小スリープ
        
    if not results:
        discord.print_log("全バックテスト結果なし")
        return results, "range", 60, 48, 0.3, 1.5
    
    # 全結果から1回以上トレードが行われたものを優先評価 pool
    active_results = {k: v for k, v in results.items() if v.get('trade_count', 0) > 0}
    eval_pool = active_results if active_results else results

    # 全体ベスト設定を選定
    best_pnl_key = max(eval_pool.keys(), key=lambda x: eval_pool[x]['final_pnl'])
    best_pnl_msg = f"★ PnL最大: {best_pnl_key} (PnL: {eval_pool[best_pnl_key]['final_pnl']:.2f} USDT, 取引: {eval_pool[best_pnl_key].get('trade_count', 0)}回)"
    best_risk_key = min(eval_pool.keys(), key=lambda x: eval_pool[x]['DD_max'] if eval_pool[x]['DD_max'] == eval_pool[x]['DD_max'] else float('inf'))
    best_risk_msg = f"★ リスク最小: {best_risk_key} (最大DD: {eval_pool[best_risk_key]['DD_max']:.2f})"
    
    # 各戦略（レンジ / ブレイクアウト）ごとに 1回以上トレードがあった中から最高成績（マイナス含む）のトップを選出
    range_active = {k: v for k, v in results.items() if v['strategy_type'] == 'range' and v.get('trade_count', 0) > 0}
    best_range_key = max(range_active.keys(), key=lambda x: range_active[x]['final_pnl']) if range_active else None

    breakout_active = {k: v for k, v in results.items() if v['strategy_type'] == 'breakout' and v.get('trade_count', 0) > 0}
    best_breakout_key = max(breakout_active.keys(), key=lambda x: breakout_active[x]['final_pnl']) if breakout_active else None
    
    discord.print_log("\n====== 戦略比較結果 ======")
    if best_range_key:
        discord.print_log(f"【レンジ戦略ベスト】: {best_range_key} -> PnL: {results[best_range_key]['final_pnl']:.2f} USDT (取引: {results[best_range_key]['trade_count']}回)")
    else:
        discord.print_log("【レンジ戦略ベスト】: 該当なし (期間中トレードなし)")

    if best_breakout_key:
        discord.print_log(f"【ブレイクアウト戦略ベスト】: {best_breakout_key} -> PnL: {results[best_breakout_key]['final_pnl']:.2f} USDT (取引: {results[best_breakout_key]['trade_count']}回)")
    else:
        discord.print_log("【ブレイクアウト戦略ベスト】: 該当なし (期間中トレードなし)")
    
    # ベストの画像のみDiscordに送信
    best_csv = f"backtest_data/klines100_{best_pnl_key}.csv"
    discord.plot_backtest(label=best_pnl_key, csv_file=best_csv)
    
    # Discord送信（結果テーブル - 上位10件のみ）
    discord.print_log("【時間足×戦略タイプ×MP期間×ER比較バックテスト結果 (Top 10)】")
    header = f"{'設定':<35} {'PnL':>8} {'DD_max':>8} {'含み損':>8} {'取引':>6}"
    separator = "-" * 70
    
    sorted_results = sorted(results.items(), key=lambda item: item[1]['final_pnl'], reverse=True)
    top_results = sorted_results[:10]
    
    all_rows = []
    for label, data in top_results:
        all_rows.append(f"{label:<35} {data['final_pnl']:>8.2f} {data['DD_max']:>8.2f} {data['max_unrealized_loss']:>8.2f} {data['trade_count']:>6}")
    
    table_lines = ["```", header, separator] + all_rows + ["```"]
    discord.print_log("\n".join(table_lines))
    
    time.sleep(1.0)
    discord.print_log(best_pnl_msg)
    time.sleep(1.0)
    discord.print_log(best_risk_msg)
    
    # ベスト設定を解析して返す
    best_strategy = results[best_pnl_key]['strategy_type']
    best_interval = results[best_pnl_key]['interval']
    best_mp = results[best_pnl_key]['mp_period']
    best_er = results[best_pnl_key]['er_threshold']
    best_atr = results[best_pnl_key]['atr_multi']
    
    discord.print_log(f"\n====== 第2段階: 最適ロジックでのストップロスマージン(%幅)の最適化 ======")
    margin_pcts = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    margin_results = {}
    
    df_interval = resample_candles(df_raw, best_interval)
    df_logic = logic.make_logic(df_interval, market_profile_period=best_mp, er_threshold=0, strategy_type=best_strategy)
    
    for margin_pct in margin_pcts:
        df_run = df_logic.copy()
        if best_strategy == "breakout":
            df_run['long'] = df_run['long_breakout'] & (df_run['er'] > best_er)
        else:
            df_run['long'] = df_run['long_range']
            
        label = f"Margin_{margin_pct}%"
        result_df = bt_instance.run_backtest(
            df=df_run,
            lot=lot,
            data_equity=fixed_initial_equity,
            side_mode=side_mode,
            mp_period=best_mp,
            atr_tp_multi=best_atr,
            er_threshold=best_er,
            strategy_type=best_strategy,
            sl_margin_pct=margin_pct
        )
        
        final_pnl = result_df['pnl'].iloc[-1] if not result_df.empty and 'pnl' in result_df.columns else 0
        metrics = getattr(bt_instance, 'last_metrics', {})
        margin_results[margin_pct] = {
            'final_pnl': final_pnl,
            'DD_max': metrics.get('DD_max', 0),
            'trade_count': metrics.get('trade_count', 0),
            'df': result_df
        }
        print(f"Margin {margin_pct}%: PnL={final_pnl:.2f} | DD={metrics.get('DD_max', 0):.2f} | 取引={metrics.get('trade_count', 0)}")
        time.sleep(0.1)

    if margin_results:
        best_margin = max(margin_results.keys(), key=lambda x: margin_results[x]['final_pnl'])
        discord.print_log(f"★ 最適マージン: {best_margin}% (PnL: {margin_results[best_margin]['final_pnl']:.2f})")
    else:
        best_margin = 1.0

    # ベスト画像の保存
    if best_margin in margin_results:
        best_csv = f"backtest_data/klines100_{best_pnl_key}_margin{best_margin}.csv"
        margin_results[best_margin]['df'].to_csv(best_csv, index=False)
        discord.plot_backtest(label=f"Margin{best_margin}%", csv_file=best_csv)

    discord.print_log(f"\n★ 採用設定: 戦略={best_strategy.upper()} {side_mode.upper()}, 時間足={best_interval}m, MP期間={best_mp}, ER閾値={best_er}, Margin={best_margin}%")
    return results, best_strategy, best_interval, best_mp, best_er, best_margin
