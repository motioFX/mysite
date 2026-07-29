#!/usr/bin/env python
# coding: utf-8

# import library
import time
import glob
import datetime
import copy
import pandas as pd
import numpy as np
import sys


# backtest用のクラス
class AirExchange:
    def __init__(self):
        # ポジション情報
        self.positions = []
        # 注文情報
        self.orders = []
        # 約定履歴
        self.exec_history = []
        # ポジション情報
        self.positions = {"avgEntry":0, "qty":0, "pos":0}

    def _rm_order(self, order, fill_timestamp):
        """
        - orderを削除
        - 履歴に追加
        - ポジションに追加
        """
        # 約定バーのtimestampを付与してポジション・履歴に反映
        executed = copy.copy(order)
        executed["timestamp"] = fill_timestamp
        self._add_order_to_position(executed)
        self.exec_history.append(executed)
        self.orders.remove(order)
    

    def check_order(self, high, low, timestamp=None):
        """
        high とlowと比べてオーダーの情報を更新する
        オーダーの情報例
        
        ord = {
            "id": id,
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": unix,
            "size": size,
            "side": side,
            "ord_type": "Limit",
            "price": price,
          
        }
        high :
            高値
        low :
            安値
        size : float
            オーダーの大きさ
        side : str
            Buy or Sell
        timestamp : int
            unixtime
        ord_type ; str
            Market Limit Stop
        price : float
            注文価格
        timestamp :
            約定したバーのunixtime
        """
        # orderが存在する場合True
        if len(self.orders):
            tmp_orders = copy.copy(self.orders)
            for i, order in enumerate(copy.copy(self.orders)):
                fill_timestamp = timestamp if timestamp is not None else order["timestamp"]
                if order["ord_type"] == "Limit":
                    # 買いの場合
                    if order["side"] == "Buy":
                        if order["price"] >= low:
                            self._rm_order(order, fill_timestamp)
                    # 売りの場合
                    else:
                        if order["price"] <= high:
                            self._rm_order(order, fill_timestamp)
                # stopの場合
                elif order["ord_type"] == "Stop":
                    if order["side"] == "Buy":
                        if order["price"] >= high:
                            self._rm_order(order, fill_timestamp)
                            
                    else:
                        if order["price"] <= low:
                            self._rm_order(order, fill_timestamp)
                            

    def _add_order_to_position(self, order):
        """オーダーをポジションに加える"""
        qty = self.positions["qty"]
        avg = self.positions["avgEntry"]
        pos = self.positions["pos"]
        size = order["size"]
        price = order["price"]
        new_qty = qty + size
        if new_qty == 0:
            new_avg = 0
            new_pos = 0
        else:
            if qty * size >= 0:     # 同方向のポジション
                new_avg = (qty * avg + size * price) / (qty + size)
                new_pos = pos + 1
            else:                   # 違う方向のポジション
                if qty >= 0:        # 元のポジションの方が大きい
                    if new_qty > 0: # 今のポジションの方が大きい
                        new_avg = avg
                    else:           # 元のポジションがマイナス
                        new_avg = price
                else:
                    if new_qty > 0: # 今のポジションの方が大きい
                        new_avg = price
                    else:
                        new_avg = avg
                new_pos = pos - 1
        self.positions["qty"] = new_qty
        self.positions["avgEntry"] = new_avg
        self.positions["pos"] = new_pos
        
    def set_order(self, id_, timestamp, size, side, ord_type, price, close):
        """
        close:
            closeの価格
        ord = {
            "id": id,
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": unix,
            "size": size,
            "side": side,
            "ord_type": "Limit",
            "price": price,
            
        """
        # 市場価格より低いとマーケットオーダーになる
        if side == "Buy":
            if ord_type == "Limit":
                if close > price:
                    self.orders.append({
                        "id":id_, 
                        "timestamp":timestamp, 
                        "size":size,
                        "side":side,
                        "ord_type":ord_type,
                        "price":price
                    })
                else:
                    # print(f"market executed side:{side} price:{price} 現在のclose{close}")
                    self.set_market_order(id_, timestamp, size, side, close)
            elif ord_type == "Stop":
                if close < price:
                    self.orders.append({
                        "id":id_, 
                        "timestamp":timestamp, 
                        "size":size,
                        "side":side,
                        "ord_type":ord_type,
                        "price":price
                    })
                else:
                    # print(f"market executed side:{side} price:{price} 現在のclose{close}")
                    self.set_market_order(id_, timestamp, size, side, close)
        # sell
        else:
            if ord_type == "Limit":
                if close < price:
                    self.orders.append({
                        "id":id_, 
                        "timestamp":timestamp, 
                        "size":size,
                        "side":side,
                        "ord_type":ord_type,
                        "price":price
                    })
                else:
                    # print(f"market executed side:{side} price:{price} 現在のclose{close}")
                    self.set_market_order(id_, timestamp, size, side, close)
            elif ord_type == "Stop":
                if close > price:
                    self.orders.append({
                        "id":id_, 
                        "timestamp":timestamp, 
                        "size":size,
                        "side":side,
                        "ord_type":ord_type,
                        "price":price
                    })
                else:
                    # print(f"market executed side:{side} price:{price} 現在のclose{close}")
                    self.set_market_order(id_, timestamp, size, side, close)
                
    def set_market_order(self,  id_, timestamp, size, side, close):
        """
        market order 
        priceには今のpriceがはいる
        market orderはcloseの時点で、発行されたものとする
        """
        order = {
            "id":id_, 
            "timestamp":timestamp, 
            "size":size,
            "side":side,
            "ord_type":"Market",
            "price":close
        }
        self._add_order_to_position(order)
        self.exec_history.append(order)
        
    def get_orders(self):
        """order取得"""
        return self.orders

    def cancel_order(self, idx):
        """あるオーダーをキャンセル"""
        del self.orders[idx]

    def cancel_all_orders(self):
        """全てのオーダーをキャンセル"""
        self.orders = []

    def get_position(self):
        """ポジションを取得"""
        return self.positions




class Backtest:
    
    def __init__(self, air_exchange, df, size=100, columns = None):
        """
        air_exchange : 取引所インスタンス
        df : 指標が格納されたデータフレーム
        size : float
            注文サイズ
        initial : float
            初期資金
        columns : list
            使用するカラム
        """
        self.air_exchange = air_exchange
        self.size = size
        # 必要な部分のみデータを抜き出す　カラムで抜き出すことで処理が軽くなる
        if columns:
            self.df = df[["timestamp"] + columns+ [ "close", "high", "low"]].dropna()
        else:
            self.df = df[["timestamp", "close", "high", "low"]].dropna()

        # 高速化するために変換
        self.arrays = self.df.to_numpy()
        self.previous_array = None
        self.idx = 0
        
        self.column_dic = {}
        self._init_column_idxs()
        self.t = 0
        self.c = None
        self.h = None
        self.l = None
        # timestamp index
        self.t_idx = self.column_dic["timestamp"]
        # close index
        self.c_idx = self.column_dic["close"]
        # high index
        self.h_idx = self.column_dic["high"]
        # low index
        self.l_idx = self.column_dic["low"]
        
    def _init_column_idxs(self):
        """カラム名に対応するインデックス番号を取得する"""
        for col in self.df.columns:
            # indexを辞書にセットする
            self.column_dic[col] = list(self.df.columns).index(col)
        # print(self.column_dic)
        
    def _get_position(self):
        """
        position情報取得

        Return
        {"avgEntry:float,"qty":float}
        avgEntry : 平均取得単価
        qty : ポジションサイズ
        """
        return self.air_exchange.positions
    
    def _get_orders(self):
        """order情報取得"""
        return self.air_exchange.orders
    
    def _cancel_all_orders(self):
        """注文キャンセル"""
        self.air_exchange.cancel_all_orders()
    
    def _limit_order(self, size, price):
        """指値注文"""
        side = "Buy" if size > 0 else "Sell"
        self.air_exchange.set_order(id_=self.idx, timestamp=self.t, size=size, side=side, ord_type="Limit", price=price, close=self.c)
    
    def _stop_order(self, size, price):
        """
        逆指値注文
        price : 
            stopのプライス
        """
        side = "Buy" if size > 0 else "Sell"
        self.air_exchange.set_order(id_=self.idx, timestamp=self.t, size=size, side=side, ord_type="Stop", price=price, close=self.c)
        
    def _market_order(self, size):
        """成り行き注文"""
        side = "Buy" if size > 0 else "Sell"
        self.air_exchange.set_market_order(id_= self.idx, timestamp=self.t, size=size, side=side, close=self.c)
        
    def _get_original_indi(self, name):
        """オリジナルのインジゲータの格納されているインデックスを取得"""
        return self.column_dic[name]

    def _cancel_entry_orders_after_stop(self):
        """
        stop_timestamp 到達後は新規エントリー方向の指値を捨てて、
        既存ポジションの縮小・決済だけを残す。フラットなら全注文をキャンセル。
        """
        pos_qty = self.air_exchange.positions["qty"]
        if pos_qty == 0:
            self.air_exchange.cancel_all_orders()
            return

        keep_orders = []
        for order in list(self.air_exchange.orders):
            size = order.get("size", 0)
            if pos_qty > 0 and size > 0:
                continue
            if pos_qty < 0 and size < 0:
                continue
            keep_orders.append(order)
        self.air_exchange.orders = keep_orders

    def run(self, stop_timestamp=None, continue_until_flat=False):
        """
        バックテストを行う。
        stop_timestamp が指定されている場合はその時刻以降は新規エントリーを行わず、
        continue_until_flat=True のときはポジションが解消されたら終了する。
        なお stop_timestamp 到達後は self.allow_entry が False に切り替わるので、
        ストラテジー側でこれを参照して新規発注を抑制する想定。
        """
        start = time.time()
        self.stop_timestamp = stop_timestamp
        self.allow_entry = True
        for i in range(len(self.arrays)):
            # indexを代入
            self.idx = i
            # 行動処理の前に previous_array を更新
            self.previous_array = self.array.copy() if i > 0 else None
            self.array = self.arrays[i]
            # 更新情報
            self.t, self.c, self.h, self.l = self.array[self.t_idx], self.array[self.c_idx], self.array[self.h_idx], self.array[self.l_idx]
            crossed_stop = stop_timestamp is not None and self.t >= stop_timestamp
            if crossed_stop:
                self.allow_entry = False
                if continue_until_flat:
                    self._cancel_entry_orders_after_stop()
            # order情報の判定
            self.air_exchange.check_order(self.h, self.l, self.t)            
            # 行動処理
            self.action()
            # exec_historyの最後の要素にlow/highの情報を追加
            if self.air_exchange.exec_history:
                self.air_exchange.exec_history[-1]['low'] = self.l
                self.air_exchange.exec_history[-1]['high'] = self.h
            if stop_timestamp is not None:
                if not continue_until_flat and self.t >= stop_timestamp:
                    break
                if continue_until_flat and not self.allow_entry:
                    pos_qty = self.air_exchange.positions["qty"]
                    if pos_qty == 0 and len(self.air_exchange.orders) == 0:
                        break
        elapsed_time = time.time() - start
        # print(f" 経過: {elapsed_time}  ")
            
            
    def action(self):

        """
        行動を選択する
        この部分に必要な処理を記載
        
        """
        # --------ここに条件を記載---------
        orders = self._get_orders()
        position = self._get_position()["qty"]  # avgEntryで平均価格を取得
        
        
        

def make_mm_pl(df, maker_fee=0, taker_fee=0, initial=100, has_ordertype = False):
    """
    df : pd.DataFrame
        size : float or int sellはマイナス buyはプラス
        time : unixtime or datetime
        price : float or int
    initial : 
        初期資金
    has_ordertype :
        手数料がある場合
    profit and loss を作成
    """

    # print(" ----   Make PL Graph   ----")
    start = time.time()

    # 必須列フォールバック (price必須、high/lowが無ければpriceで補完)
    if "price" not in df.columns:
        raise ValueError("make_mm_pl requires a 'price' column.")
    if "high" not in df.columns:
        df = df.copy()
        df["high"] = df["price"]
    if "low" not in df.columns:
        df = df.copy()
        df["low"] = df["price"]
    
    if df.empty:
        required_cols = [
            "PL",
            "low_PL",
            "comfee",
            "PLcomfee",
            "PL_graph",
            "low_PL_graph",
            "unrealized_loss",
            "_cumsum",
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = pd.Series(dtype=float)
        # print("Trades: 0 \nRealized PnL: 0 \nWin rate: 0 \nAverage PnL: 0 \nTotal Profit: 0 \nTotal Loss: 0\nPF: inf\nMax DD: 0(0%)\nMax Unrealized Loss: 0")
        # print(f" ----   elapsed time: {time.time() - start}   ---- ")
        empty_metrics = {'DD_max': 0, 'DD_per': 0, 'max_unrealized_loss': 0, 'win_rate': 0, 'PF': float('inf'), 'trade_count': 0}
        return df, 0.0, empty_metrics

    # それぞれの値をnumpyに変換
    size = df.sizes.values
    timestamp = df.time.values
    price = df.price.values
    pct_price = df.price.pct_change().values
    if has_ordertype:
        ord_type = df["ord_type"].values
    # buy sellの書き換え
    size = df.sizes.values
    # 計算用pl(空のarrayを作成)
    PLs = np.zeros(len(df))
    low_PLs = np.zeros(len(df))  # 追加：安値を使った損益
    comfees = np.zeros(len(df))
    # 積み上げposition
    cumsum_positon_size = np.cumsum(size)
    # 約定価格を保持するためのリスト (初期値は最初の価格に設定)
    entry_prices = [price[0] if len(price) > 0 else 0] * len(df)

    for i in range(1, len(df)):
        # 手数料がある場合
        if has_ordertype:
            if ord_type[i] == "Market" or ord_type[i] == "Stop" :
                comfee = taker_fee
            else:
                comfee = maker_fee
            com = comfee * abs(cumsum_positon_size[i - 1] - cumsum_positon_size[i])
            PLs[i] = pct_price[i] * cumsum_positon_size[i - 1]
            comfees[i] = com
        else:
            PLs[i] = pct_price[i] * cumsum_positon_size[i - 1]

        # 約定価格の更新 (注文があった場合)
        if size[i-1] != 0:
            entry_prices[i] = price[i-1]  # 直前の価格を約定価格とする
        else:
            entry_prices[i] = entry_prices[i-1] # 前回の約定価格を継続

        # 安値を使った損益計算（修正版）
        position_size = cumsum_positon_size[i - 1]
        if position_size != 0:
            # 安値での価格変化率を計算
            low_pct_change = (df["low"].values[i] - price[i-1]) / price[i-1] if price[i-1] != 0 else 0
            low_PLs[i] = low_pct_change * position_size
        else:
            low_PLs[i] = 0

    # 一回の損益
    df["PL"] = PLs
    df["low_PL"] = low_PLs  # 追加：安値を使った損益
    # 累積損益
    df["comfee"]= comfees
    df["comfee_graph"] = np.cumsum(comfees)
    if has_ordertype :
        PLcomfee = PLs - comfees
    else:
        PLcomfee = PLs
    df["PLcomfee"] = PLcomfee
    # 初期にinitial を足し合わせておく 
    tmp_PLcomfee = copy.copy(PLcomfee) 
    tmp_PLcomfee[0] += initial
    PL_graph = np.cumsum(tmp_PLcomfee)
    df["PL_graph"] = PL_graph
    
    # 安値ベースの累積損益も計算
    tmp_low_PLcomfee = copy.copy(low_PLs)
    tmp_low_PLcomfee[0] += initial
    low_PL_graph = np.cumsum(tmp_low_PLcomfee)
    df["low_PL_graph"] = low_PL_graph

    # 安値ベースの損益指標も計算
    low_pl = low_PL_graph[-1] - initial
    # ゼロ除算を防ぐ
    valid_trades = len(low_PLs) - (low_PLs == 0).sum()
    low_avg_pl = low_pl / valid_trades if valid_trades > 0 else 0
    low_profit_total = df.low_PL[df.low_PL > 0].sum()
    low_loss_total = df.low_PL[df.low_PL < 0].sum()
    low_PF = low_profit_total / abs(low_loss_total) if low_loss_total != 0 else float('inf')

    # 最大含み損の計算（ベクトル化版）
    position_sizes = np.roll(cumsum_positon_size, 1)
    position_sizes[0] = 0
    prev_prices = np.roll(price, 1)
    prev_prices[0] = price[0] if len(price) > 0 else 1
    
    low_values = df["low"].values
    high_values = df["high"].values
    
    # 価格変化率を計算（ゼロ除算防止）
    safe_prev_prices = np.where(prev_prices == 0, 1, prev_prices)
    low_pct_change = (low_values - prev_prices) / safe_prev_prices
    high_pct_change = (high_values - prev_prices) / safe_prev_prices
    
    # ロング/ショートポジションに応じて含み損を計算
    unrealized_loss_array = np.where(
        position_sizes > 0,
        low_pct_change * position_sizes,  # ロング: 安値での含み損
        np.where(
            position_sizes < 0,
            -high_pct_change * np.abs(position_sizes),  # ショート: 高値での含み損
            0  # ノーポジション
        )
    )
    unrealized_loss_array[0] = 0  # 最初の要素は0
    
    # 最大含み損
    max_unrealized_loss = unrealized_loss_array.min()

    # DataFrameに含み損の列を追加
    df["unrealized_loss"] = unrealized_loss_array

    # 安値ベースのDD計算（ベクトル化版）
    unrealized_loss_cumsum = np.cumsum(unrealized_loss_array)
    running_max = np.maximum.accumulate(unrealized_loss_cumsum)
    drawdowns = running_max - unrealized_loss_cumsum
    low_DD_max = drawdowns.max()
    
    # DD発生時のrunning_maxを使ってDD%を計算
    dd_max_idx = np.argmax(drawdowns)
    low_PL_max = running_max[dd_max_idx]
    low_DD_per = (low_DD_max / low_PL_max * 100) if low_PL_max != 0 else 0

    # if sys.platform == 'win32':
    #     print(f"安値ベースの損益指標:")
    #     print(f"実現損益: {low_pl}")
    #     print(f"平均損益: {low_avg_pl}")
    #     print(f"総利益: {low_profit_total}")
    #     print(f"総損失: {low_loss_total}")
    #     print(f"PF: {low_PF}")
    #     print(f"最大DD: {low_DD_max}({low_DD_per}%)")
    #     print(f"最大含み損: {max_unrealized_loss}")

    # 勝率
    win_cnt = PLs > 0
    none_cnt = PLs == 0
    none_cnt = none_cnt.sum()
    win_cnt = win_cnt.sum()
    win_rate = win_cnt / (len(PLs) - none_cnt) if (len(PLs) - none_cnt) > 0 else 0
    # 実現損益
    pl = PL_graph[-1] - initial
    # 平均損益
    avg_pl = pl / len(PLcomfee) if len(PLcomfee) > 0 else 0
    #総利益
    profit_total =  df.PLcomfee[df.PLcomfee > 0].sum()
    #総損失
    loss_total =  df.PLcomfee[ df.PLcomfee < 0].sum()
    #PF
    PF = profit_total / abs(loss_total) if loss_total != 0 else float('inf')
    #DD_max
    PL_max = 0.00
    DD_max = 0.00
    DD_per = 0.0000
    for i in df.PL_graph:
        if PL_max < i:
            PL_max = i
        DD = PL_max - i
        if DD_max < DD:
            DD_max = DD
            if PL_max:
                DD_per = (DD_max / PL_max)*100
            else:
                pl_per = None

    df["_cumsum"] = cumsum_positon_size
    # if sys.platform == 'win32':
    #     print(f"取引回数: {len(PLs)} \n実現損益: {pl} \n勝率: {win_rate} \n平均損益: {avg_pl} \n総利益: {profit_total} \n総損失: {loss_total}\nPF: {PF}\n最大DD: {DD_max}({DD_per}%)\n最大含み損: {max_unrealized_loss}")
    elapsed_time = time.time() - start
    # print(f" ----   elapsed time: {elapsed_time}   ---- ")
    # 追加のメトリクスを辞書で返す
    metrics = {
        'DD_max': DD_max,
        'DD_per': DD_per,
        'max_unrealized_loss': max_unrealized_loss,
        'win_rate': win_rate,
        'PF': PF,
        'trade_count': len(PLs)
    }
    return df, pl, metrics

