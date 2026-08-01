from bitget5_44_3logic import send_discord, backtester, PositionSizer, calc_add_pct
from decimal import Decimal, ROUND_HALF_UP
import time
import pandas as pd
import numpy as np
import pybotters
import asyncio
import math
import sys
import json
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

#====================〇二重化設定値〇====================
bitget_mode = 'demo' # 'live' または 'demo'
BITGET_TARGET_POSITION_VALUE_USDT = 50.0 # Bitgetの本番用発注ターゲット（Bybitと別設定可能）
LEVERAGE_FACTOR = 25.0
is_air = True

# Helper to read values
def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

#====================〇アカウント情報ロード〇====================
# Load Bitget credentials
bitget_config_path = Path(__file__).resolve().parent / "bitget_credentials.json"
bybit_config_path = Path(__file__).resolve().parent / "bybit_credentials.json"

try:
    with open(bitget_config_path, "r", encoding="utf-8") as fp:
        bitget_cfg = json.load(fp)
    apis_bitget_raw = bitget_cfg.get("apis", {})
    if "win32" in apis_bitget_raw or "default" in apis_bitget_raw or "linux" in apis_bitget_raw:
        apis_bitget_raw = apis_bitget_raw.get(sys.platform) or apis_bitget_raw.get("default") or {}
    
    apis_bitget = {
        "bitget": apis_bitget_raw.get("bitget"),
        "bitget_demo": apis_bitget_raw.get("bitget_demo")
    }
except Exception as e:
    raise RuntimeError(f"Failed to load Bitget credentials from {bitget_config_path}: {e}")

# Assign apis directly for Bitget
apis = apis_bitget 

# ポジションログの頻度制御（秒）。
POSITION_LOG_INTERVAL_SEC = 60 * 60
_last_position_log_ts = 0.0

RestAPI_url = {
    'bybit'          :'https://api.bybit.com',
    'bybit_demo'     :'https://api-demo.bybit.com',
    'bitget'         :'https://api.bitget.com',
}

discord = send_discord()

# Bitget symbol utilities
DEFAULT_PRODUCT_TYPE = "USDT-FUTURES"
PRODUCT_TYPE_MAP = {
    "UMCBL": "USDT-FUTURES",
    "CMCBL": "COIN-FUTURES",
    "DMCBL": "USDC-FUTURES",
}
PRODUCT_TYPE_SUFFIX_MAP = {value: key for key, value in PRODUCT_TYPE_MAP.items()}

def normalize_product_type(product_type: str) -> str:
    if not product_type:
        return DEFAULT_PRODUCT_TYPE
    key = str(product_type).upper()
    if key in PRODUCT_TYPE_MAP:
        return PRODUCT_TYPE_MAP[key]
    if key in PRODUCT_TYPE_SUFFIX_MAP:
        return key
    return key

def normalize_symbol(symbol: str) -> str:
    sym = str(symbol).upper()
    if "_" in sym:
        base, suffix = sym.rsplit("_", 1)
        if suffix in PRODUCT_TYPE_MAP:
            return base
    return sym

def split_symbol_and_product_type(symbol: str, product_type: str) -> tuple[str, str]:
    sym = str(symbol).upper()
    if "_" in sym:
        base, suffix = sym.rsplit("_", 1)
        mapped = PRODUCT_TYPE_MAP.get(suffix)
        if mapped:
            return base, mapped
        return base, normalize_product_type(product_type)
    return sym, normalize_product_type(product_type)

def build_contract_symbol(symbol: str, product_type: str) -> str:
    sym = str(symbol).upper()
    if "_" in sym:
        base, suffix = sym.rsplit("_", 1)
        if suffix in PRODUCT_TYPE_MAP:
            return sym
        sym = base
    suffix = PRODUCT_TYPE_SUFFIX_MAP.get(normalize_product_type(product_type))
    return f"{sym}_{suffix}" if suffix else sym


def quantize_quantity(quantity: float, step: float) -> float:
    if step <= 0:
        return quantity
    decimal_step = Decimal(str(step)).normalize()
    return float(Decimal(str(quantity)).quantize(decimal_step, rounding=ROUND_HALF_UP))

#====================〇Bitget API Helper〇====================
class api_bitget_helper:
    def __init__(self, symbol='BTCUSDT', product_type='umcbl', coin='USDT', mode='demo', instrument_spec=None):
        self.symbol, self.product_type = split_symbol_and_product_type(symbol, product_type)
        self.symbol_v1 = build_contract_symbol(self.symbol, self.product_type)
        self.coin = coin
        self.mode = mode  # 'demo' or 'live'
        self.instrument_spec = instrument_spec or {}
        self.base_url = RestAPI_url['bitget']
        self._demo_headers = {"paptrading": "1"} if self.mode in ('paper', 'demo') else {}
        api_key_name = 'bitget_demo' if self.mode in ('paper', 'demo') and 'bitget_demo' in apis_bitget else 'bitget'
        self._apis = {'bitget': apis_bitget.get(api_key_name) or apis_bitget.get('bitget') or apis_bitget}
        self._product_type_param = self.product_type

    def update_instrument_spec(self, spec=None):
        self.instrument_spec = spec or {}

    def _get_spec_value(self, key: str, default: float) -> float:
        spec = self.instrument_spec or {}
        value = spec.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _quantize_quantity(self, quantity: float) -> float:
        qty_step = self._get_spec_value("qty_step", 0.0)
        if qty_step <= 0:
            return float(quantity)
        decimal_step = Decimal(str(qty_step)).normalize()
        return float(Decimal(str(quantity)).quantize(decimal_step, rounding=ROUND_HALF_UP))

    def _quantize_price(self, price: float) -> float:
        price_place = self._get_spec_value("price_place", 4.0)
        try:
            return float(round(price, int(price_place)))
        except:
            return price

    def _headers(self) -> dict:
        return dict(self._demo_headers)

    async def get_account(self):
        async with pybotters.Client(apis=self._apis, base_url=self.base_url, headers=self._headers()) as client:
            try:
                resp = await asyncio.wait_for(
                    client.get(
                        '/api/v2/mix/account/account',
                        params={
                            'symbol': self.symbol,
                            'productType': self._product_type_param,
                            'marginCoin': self.coin,
                        },
                        headers=self._headers(),
                    ),
                    timeout=30
                )
                data = await resp.json()
            except Exception as e:
                discord.print_log(f"Bitget get account error: {e}")
                return 0.0

        result = data.get('data') if isinstance(data, dict) else None
        if isinstance(result, list):
            result = result[0] if result else {}
        if not isinstance(result, dict) or not result:
            return 0.0

        equity = float(
            result.get('equity')
            or result.get('usdtEquity')
            or result.get('available')
            or 0.0
        )
        discord.print_log(f"Bitget {self.coin} 残高 {equity} [{self.coin}] (mode={self.mode})")
        return equity

    async def get_positions(self, lot_size=None):
        def build_default_position() -> dict:
            return {
                'buy': 0.00,
                'sell': 0.00,
                'buy_pos': 0.00,
                'sell_pos': 0.00,
                'profit': 0.000,
                'raw_pnl': 0.000,
                'total_fee': 0.000,
                'pos_count': 0.0,
                'buy_count': 0,
                'sell_count': 0,
                'margin_mode': 'crossed',
            }

        def finalize_position(position: dict) -> dict:
            buy_qty = position["buy"]
            sell_qty = position["sell"]
            if lot_size and lot_size > 0:
                buy_size = max(round(buy_qty / lot_size), 0)
                sell_size = max(round(sell_qty / lot_size), 0)
                position['buy_count'] = buy_size
                position['sell_count'] = sell_size
                PositionSizer.build_cumulative_dict()
                stage = PositionSizer.get_stage_from_position_size(max(buy_size, sell_size))
                position['pos_count'] = float(stage)
            else:
                position['buy_count'] = 0 if buy_qty == 0 else 1
                position['sell_count'] = 0 if sell_qty == 0 else 1
                position['pos_count'] = float(max(position['buy_count'], position['sell_count']))
            return position

        async with pybotters.Client(apis=self._apis, base_url=self.base_url, headers=self._headers()) as client:
            try:
                res = await asyncio.wait_for(
                    client.get(
                        '/api/v2/mix/position/single-position',
                        params={
                            'symbol': self.symbol,
                            'productType': self._product_type_param,
                            'marginCoin': self.coin,
                        },
                        headers=self._headers(),
                    ),
                    timeout=30
                )
                data = await res.json()
            except Exception as e:
                discord.print_log(f'Bitget get_position error: {e}')
                return finalize_position(build_default_position())

        if not isinstance(data, dict) or data.get('code') != '00000':
            return finalize_position(build_default_position())

        payload = data.get('data') or []
        if isinstance(payload, dict):
            items = payload.get('positions')
            if not isinstance(items, list):
                items = [payload]
        elif isinstance(payload, list):
            items = payload
        else:
            items = []
        position = build_default_position()
        for p in items:
            side = str(p.get('holdSide') or p.get('posSide') or '').lower()
            qty = float(p.get('total') or p.get('available') or 0.0)
            if qty < 1e-8:
                qty = 0.0
            avg = float(p.get('averageOpenPrice') or p.get('avgOpenPrice') or 0.0)
            api_profit = float(p.get('unrealisedPL') or p.get('unrealisedPnl') or 0.0)
            mark_price = float(p.get('markPrice') or p.get('marketPrice') or p.get('lastPr') or p.get('lastPrice') or 0.0)
            
            # 手数料率 (デフォルト Taker 約 0.06%)
            commission_rate = getattr(self, 'commission', 0.0006)
            if not isinstance(commission_rate, (int, float)) or commission_rate <= 0:
                commission_rate = 0.0006
            
            # 自前損益計算（取引量、価格差、往復手数料を考慮）
            if qty > 0 and avg > 0 and mark_price > 0:
                entry_fee = qty * avg * commission_rate
                exit_fee = qty * mark_price * commission_rate
                total_fee = entry_fee + exit_fee
                if side == 'long':
                    raw_pnl = (mark_price - avg) * qty
                else:
                    raw_pnl = (avg - mark_price) * qty
                calc_profit = raw_pnl - total_fee
                profit = calc_profit
            else:
                raw_pnl = api_profit
                total_fee = 0.0
                profit = api_profit

            m_mode = str(p.get('marginMode') or '').lower()
            if m_mode:
                if m_mode == 'cross':
                    m_mode = 'crossed'
                position['margin_mode'] = m_mode
            
            if side == 'long':
                position['buy'] = qty
                position['buy_pos'] = avg
                position['profit'] = profit
                position['raw_pnl'] = raw_pnl
                position['total_fee'] = total_fee
            elif side == 'short':
                position['sell'] = qty
                position['sell_pos'] = avg
                position['profit'] = profit
                position['raw_pnl'] = raw_pnl
                position['total_fee'] = total_fee

        position = finalize_position(position)
        return position

    async def get_open_orders(self) -> list[dict]:
        async with pybotters.Client(apis=self._apis, base_url=self.base_url, headers=self._headers()) as client:
            try:
                res = await asyncio.wait_for(
                    client.get(
                        "/api/v2/mix/order/orders-pending",
                        params={
                            "symbol": self.symbol,
                            "productType": self._product_type_param,
                        },
                        headers=self._headers(),
                    ),
                    timeout=30
                )
                data = await res.json()
            except Exception as e:
                discord.print_log(f"Bitget get_open_orders error: {e}")
                return []

        if not isinstance(data, dict) or data.get("code") != "00000":
            return []

        result = data.get("data") or {}
        orders = result.get("entrustedList") or result.get("orderList") or result.get("orders") or []
        return orders

    async def get_open_plan_orders(self) -> list[dict]:
        orders = []
        async with pybotters.Client(apis=self._apis, base_url=self.base_url, headers=self._headers()) as client:
            for plan_type in ["normal_plan", "profit_loss"]:
                try:
                    r = await client.get(
                        "/api/v2/mix/order/orders-plan-pending",
                        params={
                            "symbol": self.symbol,
                            "productType": self._product_type_param,
                            "planType": plan_type,
                            "pageSize": 50,
                        },
                        headers=self._headers(),
                    )
                    data = await r.json()
                    if isinstance(data, dict) and data.get("code") == "00000":
                        result = data.get("data") or {}
                        lst = result.get("entrustedList") or result.get("orderList") or []
                        for o in lst:
                            o["planType"] = plan_type
                            orders.append(o)
                except Exception as e:
                    discord.print_log(f"Bitget get_open_plan_orders error for {plan_type}: {e}")
        return orders

    async def active_order_cancel(self):
        orders = await self.get_open_orders()
        normal_ok = True
        if orders:
            async with pybotters.Client(apis=self._apis, base_url=self.base_url, headers=self._headers()) as client:
                for order in orders:
                    order_id = order.get('orderId') or order.get('orderID')
                    client_oid = order.get('clientOid') or order.get('clientOrderId')
                    margin_coin = order.get('marginCoin') or self.coin
                    payload = {
                        'symbol': self.symbol,
                        'productType': self._product_type_param,
                        'marginCoin': margin_coin,
                    }
                    if order_id:
                        payload['orderId'] = order_id
                    if client_oid:
                        payload['clientOid'] = client_oid
                    if not order_id and not client_oid:
                        normal_ok = False
                        continue
                    try:
                        r = await client.post(
                            '/api/v2/mix/order/cancel-order',
                            data=payload,
                            headers=self._headers(),
                        )
                        data = await r.json()
                        if data.get('code') != '00000':
                            discord.print_log(f'Bitget Cancel order failed: {data}')
                            normal_ok = False
                    except Exception as e:
                        discord.print_log(f'Bitget active order cancel error: {e}')
                        normal_ok = False

        plan_orders = await self.get_open_plan_orders()
        plan_ok = True
        if plan_orders:
            async with pybotters.Client(apis=self._apis, base_url=self.base_url, headers=self._headers()) as client:
                for order in plan_orders:
                    order_id = order.get('orderId') or order.get('orderID')
                    if not order_id:
                        continue
                    margin_coin = order.get('marginCoin') or self.coin
                    payload = {
                        'symbol': self.symbol,
                        'productType': self._product_type_param,
                        'marginCoin': margin_coin,
                        'orderId': order_id,
                        'planType': order.get('planType', 'normal_plan')
                    }
                    try:
                        r = await client.post(
                            '/api/v2/mix/order/cancel-plan-order',
                            data=payload,
                            headers=self._headers(),
                        )
                        data = await r.json()
                        if data.get('code') != '00000':
                            discord.print_log(f"Bitget Failed to cancel plan order {order_id}: {data}")
                            plan_ok = False
                    except Exception as e:
                        discord.print_log(f"Bitget plan order cancel error: {e}")
                        plan_ok = False

        return normal_ok and plan_ok

    async def get_candle(self, df, interval, interval_int):
        max_retries = 3
        retry_delay = 5
        
        mapping = {
            '1': '1m', '3': '3m', '5': '5m', '15': '15m', '30': '30m',
            '60': '1H', '120': '2H', '240': '4H', '360': '6H', '720': '12H',
            'D': '1D',
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1H', '4h': '4H', '6h': '6H', '12h': '12H', '1d': '1D'
        }
        granularity = mapping.get(str(interval), '1H')

        def _build_df(rows) -> pd.DataFrame:
            if not rows:
                raise ValueError("Empty candle payload")
            tmp_df = pd.DataFrame(rows)
            tmp_df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                tmp_df[col] = pd.to_numeric(tmp_df[col], errors='coerce')
            if tmp_df[numeric_columns].isna().any().any():
                raise ValueError("Data contains NaN values")
            tmp_df['timestamp'] = pd.to_datetime(tmp_df['timestamp'].astype(float), unit='ms')
            tmp_df = tmp_df.sort_values('timestamp', ascending=True).reset_index(drop=True)
            return tmp_df

        async with pybotters.Client(base_url=self.base_url, headers=self._headers()) as client:
            for attempt in range(max_retries):
                try:
                    resps = await asyncio.wait_for(
                        client.get('/api/v2/mix/market/candles',
                            params={
                                'symbol': self.symbol,
                                'productType': self._product_type_param,
                                'granularity': granularity,
                                'limit': '500'
                            },
                            headers=self._headers()
                        ),
                        timeout=30
                    )
                    if resps.status != 200:
                        raise Exception(f"API request failed with status {resps.status}")
                    data = await resps.json()
                    if data and data.get("code") == "00000" and data.get("data"):
                        rows = data["data"]
                        return _build_df(rows)
                    else:
                        raise Exception(f"Invalid response code={data.get('code')} msg={data.get('msg')}")
                except Exception as e:
                    print(f"Bitget get_candle error - Attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt == max_retries - 1:
                        discord.print_log(f"Bitget get_candle error - Failed after {max_retries} attempts: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)

            # fallback
            if df is not None and not df.empty:
                return df
            return pd.DataFrame()


#====================〇二重化 API クラス (api_bybit / drop-in)〇====================
class api_bitget:
    def __init__(self, symbol='BTCUSDT', category='linear', coin='USDT', mode='demo', instrument_spec=None):
        self.symbol = symbol
        self.category = category
        self.coin = coin
        self.mode = mode
        self.instrument_spec = instrument_spec or {}

        # Bitget API Helper
        self.bitget_mode = bitget_mode
        self.bitget = api_bitget_helper(symbol, 'USDT-FUTURES', coin, bitget_mode)
        
        # Proactively load Bitget instrument spec at startup
        try:
            bg_spec = fetch_instrument_spec_bitget(symbol, 'USDT-FUTURES', bitget_mode)
            if bg_spec:
                self.bitget.update_instrument_spec(bg_spec)
                # print(f"[DUAL] Loaded Bitget contract spec for {symbol}: {bg_spec}")
        except Exception as e:
            discord.print_log(f"[DUAL] Error fetching Bitget spec for {symbol}: {e}")

    def update_instrument_spec(self, spec=None):
        self.instrument_spec = spec or {}
        self.bitget.update_instrument_spec(spec)

    def _get_spec_value(self, key: str, default: float) -> float:
        return self.bitget._get_spec_value(key, default)

    def _quantize_quantity(self, quantity: float) -> float:
        return self.bitget._quantize_quantity(quantity)

    async def get_account(self):
        return await self.bitget.get_account()

    async def get_candle(self, df, interval, interval_int):
        return await self.bitget.get_candle(df, interval, interval_int)

    async def get_positions(self, lot_size=None):
        return await self.bitget.get_positions(lot_size)

    async def active_order_cancel(self):
        return await self.bitget.active_order_cancel()

    async def get_open_orders(self, include_stop: bool = True) -> list[dict]:
        orders = await self.bitget.get_open_orders()
        if include_stop:
            plan_orders = await self.bitget.get_open_plan_orders()
            orders.extend(plan_orders)
        return orders

    async def long_entry(self, df, position, usdt_onhand_amount, lot_size, max_lot):
        if lot_size == 0:
            raise ValueError("lot_size cannot be zero")
        current_stage = int(position["pos_count"])
        if current_stage >= 1:
            return False
        if not df['long'].iloc[-1]:
            return False

        current_price = df['close'].iloc[-1]
        
        bitget_spec = self.bitget.instrument_spec or {}
        bitget_base_lot = compute_bitget_lot_size(current_price, BITGET_TARGET_POSITION_VALUE_USDT, bitget_spec)
        lot = self.bitget._quantize_quantity(bitget_base_lot)
        
        if lot <= 0:
            return False

        lotamount = current_price * lot / 100
        if usdt_onhand_amount < lotamount:
            discord.print_log(f"Insufficient funds for Bitget long entry: need {lotamount}, have {usdt_onhand_amount}")
            return False

        final_price = current_price
        try:
            async with pybotters.Client(apis=self.bitget._apis, base_url=self.bitget.base_url, headers=self.bitget._headers()) as client:
                ticker_res = await client.get(
                    "/api/v2/mix/market/ticker",
                    params={"symbol": self.bitget.symbol, "productType": self.bitget._product_type_param},
                    headers=self.bitget._headers(),
                )
                ticker_json = await ticker_res.json()
                tick = {}
                if isinstance(ticker_json, dict):
                    raw = ticker_json.get("data")
                    if isinstance(raw, list):
                        tick = raw[0] if raw else {}
                    elif isinstance(raw, dict):
                        tick = raw
                best_ask = float(tick.get("bestAsk") or tick.get("askPr") or 0.0)
                if best_ask > 0:
                    final_price = best_ask
        except Exception as e:
            discord.print_log(f"[BITGET] Failed to fetch ticker for entry price: {e}")
            final_price = current_price

        final_price = self.bitget._quantize_price(final_price)

        bitget_pos = await self.bitget.get_positions()
        margin_mode_param = bitget_pos.get('margin_mode', 'crossed')

        order_payload = {
            'symbol': self.bitget.symbol,
            'productType': self.bitget._product_type_param,
            'marginCoin': self.coin,
            'marginMode': margin_mode_param,
            'size': str(lot),
            'side': 'buy',
            'tradeSide': 'open',
            'orderType': 'limit',
            'price': str(final_price),
            'timeInForce': 'gtc',
        }
        await self.bitget.active_order_cancel()
        await asyncio.sleep(0.5)

        discord.print_log(f"[BITGET] Placing Stage 1 Long: Price {final_price}, Qty {lot}")
        try:
            async with pybotters.Client(apis=self.bitget._apis, base_url=self.bitget.base_url, headers=self.bitget._headers()) as client:
                res = await client.post("/api/v2/mix/order/place-order", data=order_payload, headers=self.bitget._headers())
                data = await res.json()
                if data.get('code') != '00000':
                    discord.print_log(f"Bitget Long entry failed: {data}")
                    return False
                return True
        except Exception as e:
            discord.print_log(f"Bitget Long entry exception: {e}")
            return False

    async def long_close(self, df, position, commission, sl_margin_pct=1.0, strategy_type="range", is_new_candle=False, entry_candle_time=None):
        pos_qty = float(position["buy"])
        if pos_qty <= 0:
            self.max_price_since_entry = None
            self.crossed_vah = False
            return False

        current_price = df['close'].iloc[-1]
        val_price = df['VAL'].iloc[-1]
        vah_price = df['VAH'].iloc[-1]
        avg_entry = float(position.get('buy_pos', 0))

        if not hasattr(self, 'max_price_since_entry') or self.max_price_since_entry is None or self.max_price_since_entry == 0.0:
            self.max_price_since_entry = max(avg_entry, current_price)
            self.crossed_vah = False
        else:
            self.max_price_since_entry = max(self.max_price_since_entry, current_price)

        if not hasattr(self, 'crossed_vah'):
            self.crossed_vah = False

        if strategy_type == "breakout":
            base_line = vah_price
        else:
            if current_price > vah_price:
                self.crossed_vah = True
            base_line = vah_price if self.crossed_vah else val_price

        sl_price = base_line * (1.0 - sl_margin_pct / 100.0)
        trailing_sl = self.max_price_since_entry * (1.0 - sl_margin_pct / 100.0)
        sl_price = max(sl_price, trailing_sl)

        discord.print_log(f"[BITGET-TRAIL] Price={current_price:.6f}, MaxPrice={self.max_price_since_entry:.6f}, BaseSL={base_line * (1.0 - sl_margin_pct / 100.0):.6f}, TrailSL={trailing_sl:.6f}, SL={sl_price:.6f}")

        if current_price < sl_price:
            discord.print_log(f"[BITGET] TrailingSL Triggered: Price {current_price} < SL {sl_price}")
            self.max_price_since_entry = None
            self.crossed_vah = False
            await self.bitget.active_order_cancel()
            await asyncio.sleep(0.5)
            await flatten_current_position_bitget(
                self.bitget.product_type, self.bitget.symbol, self.coin, self.bitget_mode,
                "TrailingSL", force_market=True
            )
            return True

        return False

    async def short_entry(self, df, position, usdt_onhand_amount, lot_size, max_lot):
        if lot_size == 0:
            raise ValueError("lot_size cannot be zero")
        current_stage = int(position["pos_count"])
        if current_stage >= 1:
            return False

        current_price = df['close'].iloc[-1]
        
        bitget_spec = self.bitget.instrument_spec or {}
        bitget_base_lot = compute_bitget_lot_size(current_price, BITGET_TARGET_POSITION_VALUE_USDT, bitget_spec)
        lot = self.bitget._quantize_quantity(bitget_base_lot)

        if lot <= 0:
            return False

        lotamount = current_price * lot / 100
        if usdt_onhand_amount < lotamount:
            discord.print_log(f"Insufficient funds for Bitget short entry: need {lotamount}, have {usdt_onhand_amount}")
            return False

        target_price = 0.0
        should_enter = False

        if current_stage == 0:
            if df['short'].iloc[-1]:
                target_price = 0
                should_enter = True
                discord.print_log(f"Signal: Short Breakout (Close {current_price} < VAL)")
        elif current_stage == 1:
            target_price = df['POC'].iloc[-1]
            should_enter = True
        elif current_stage == 2:
            target_price = df['VAH'].iloc[-1]
            should_enter = True

        if not should_enter:
            return False

        final_price = target_price
        if current_stage == 0 or final_price == 0:
            try:
                async with pybotters.Client(apis=self.bitget._apis, base_url=self.bitget.base_url, headers=self.bitget._headers()) as client:
                    ticker_res = await client.get(
                        "/api/v2/mix/market/ticker",
                        params={"symbol": self.bitget.symbol, "productType": self.bitget._product_type_param},
                        headers=self.bitget._headers(),
                    )
                    ticker_json = await ticker_res.json()
                    tick = {}
                    if isinstance(ticker_json, dict):
                        raw = ticker_json.get("data")
                        if isinstance(raw, list):
                            tick = raw[0] if raw else {}
                        elif isinstance(raw, dict):
                            tick = raw
                    best_bid = float(tick.get("bestBid") or tick.get("bidPr") or 0.0)
                    if best_bid > 0:
                        final_price = best_bid
            except Exception as e:
                discord.print_log(f"[BITGET] Failed to fetch ticker for entry price: {e}")
                final_price = current_price

        final_price = self.bitget._quantize_price(final_price)

        if current_stage > 0:
            open_orders = await self.get_open_orders(include_stop=False)
            for o in open_orders:
                o_price = float(o.get("price", 0) or o.get("triggerPrice", 0) or 0)
                o_qty = float(o.get("size") or o.get("qty") or 0)
                if o.get("side") == "sell" and abs(o_price - final_price) < 0.5 and abs(o_qty - lot) < 0.0001:
                    return False

        await self.bitget.active_order_cancel()
        await asyncio.sleep(0.5)

        bitget_pos = await self.bitget.get_positions()
        margin_mode_param = bitget_pos.get('margin_mode', 'crossed')

        order_payload = {
            'symbol': self.bitget.symbol,
            'productType': self.bitget._product_type_param,
            'marginCoin': self.coin,
            'marginMode': margin_mode_param,
            'size': str(lot),
            'side': 'sell',
            'tradeSide': 'open',
            'orderType': 'limit',
            'price': str(final_price),
            'timeInForce': 'gtc',
        }

        discord.print_log(f"[BITGET] Placing Short Entry (Stage {current_stage}): Price {final_price}, Qty {lot}")
        try:
            async with pybotters.Client(apis=self.bitget._apis, base_url=self.bitget.base_url, headers=self.bitget._headers()) as client:
                res = await client.post("/api/v2/mix/order/place-order", data=order_payload, headers=self.bitget._headers())
                data = await res.json()
                if data.get('code') != '00000':
                    discord.print_log(f"Bitget Short entry failed: {data}")
                    return False
                return True
        except Exception as e:
            discord.print_log(f"Bitget Short entry exception: {e}")
            return False

    async def short_close(self, df, position, commission, sl_margin_pct=1.0, strategy_type="range", is_new_candle=False, entry_candle_time=None):
        pos_qty = float(position["sell"])
        if pos_qty <= 0:
            self.max_price_since_entry = None
            self.crossed_vah = False
            return False

        current_price = df['close'].iloc[-1]
        avg_entry = float(position.get('sell_pos', 0))
        val_price = df['VAL'].iloc[-1]
        vah_price = df['VAH'].iloc[-1]

        if not hasattr(self, 'max_price_since_entry') or self.max_price_since_entry is None or self.max_price_since_entry == 0.0:
            self.max_price_since_entry = min(avg_entry, current_price)
            self.crossed_vah = False
        else:
            self.max_price_since_entry = min(self.max_price_since_entry, current_price)

        if not hasattr(self, 'crossed_vah'):
            self.crossed_vah = False

        if strategy_type == "breakout":
            base_line = val_price
        else:
            if current_price < val_price:
                self.crossed_vah = True
            base_line = val_price if self.crossed_vah else vah_price

        sl_price = base_line * (1.0 + sl_margin_pct / 100.0)
        trailing_sl = self.max_price_since_entry * (1.0 + sl_margin_pct / 100.0)
        sl_price = min(sl_price, trailing_sl)

        if current_price > sl_price:
            discord.print_log(f"[BITGET] TrailingSL Triggered (Short): Price {current_price} > SL {sl_price}")
            self.max_price_since_entry = None
            self.crossed_vah = False
            await self.bitget.active_order_cancel()
            await asyncio.sleep(0.5)
            await flatten_current_position_bitget(
                self.bitget.product_type, self.bitget.symbol, self.coin, self.bitget_mode,
                "TrailingSL", force_market=True
            )
            return True

        return False


#====================〇Bitget contract size calculations〇====================
def compute_bitget_lot_size(price: float, target_value: float, spec: dict) -> float:
    qty_step = spec.get("qty_step", 0.01)
    min_qty = spec.get("min_qty", 0.01)
    max_qty = spec.get("max_qty", float("inf"))
    min_notional = spec.get("min_notional", 0.0)

    base_qty = target_value / price if price > 0 else min_qty
    quantity = quantize_quantity(base_qty, qty_step)
    if quantity < min_qty:
        quantity = quantize_quantity(min_qty, qty_step)
    if min_notional > 0 and quantity * price < min_notional:
        required_qty = min_notional / price
        quantity = quantize_quantity(max(required_qty, quantity), qty_step)
    if math.isfinite(max_qty):
        quantity = min(quantity, max_qty)
    return quantity


#====================〇Global Spec Fetching Functions〇====================
def fetch_instrument_spec_bitget(symbol: str, product_type: str, mode: str) -> Optional[Dict[str, float]]:
    base_url = RestAPI_url['bitget']
    endpoint = f"{base_url}/api/v2/mix/market/contracts"
    params = {"productType": product_type, "symbol": symbol}
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        discord.print_log(f"Bitget 銘柄仕様の取得に失敗しました: {exc}")
        return None
    
    if not isinstance(payload, dict) or payload.get("code") != "00000":
        return None
    
    items = payload.get("data") or []
    if isinstance(items, dict):
        items = [items]
    if not items:
        return None
    
    item = None
    for it in items:
        if it.get("symbol", "").upper() == symbol.upper():
            item = it
            break
    if item is None:
        item = items[0]
    
    return {
        "qty_step": _to_float(item.get("sizeMultiplier") or item.get("volumePlace"), 0.01),
        "min_qty": _to_float(item.get("minTradeNum"), 0.01),
        "max_qty": _to_float(item.get("maxTradeNum") or item.get("maxOrderQty") or item.get("maxMarketOrderQty"), float("inf")),
        "min_notional": _to_float(item.get("minTradeUSDT"), 0.0),
        "max_leverage": _to_float(item.get("maxLever") or item.get("leverageRange"), LEVERAGE_FACTOR),
        "price_place": int(_to_float(item.get("pricePlace"), 4.0)),
    }


#====================〇Global Flatten / Symbol Fetching Functions〇====================
async def fetch_all_position_symbols_bitget(product_type: str, margin_coin: str, mode: str) -> list[str]:
    base_url = RestAPI_url["bitget"]
    normalized_product_type = normalize_product_type(product_type)
    demo_headers = {"paptrading": "1"} if mode in ("paper", "demo") else {}
    api_key_name = 'bitget_demo' if mode in ("paper", "demo") and 'bitget_demo' in apis_bitget else 'bitget'
    local_apis = {'bitget': apis_bitget.get(api_key_name) or apis_bitget.get('bitget') or apis_bitget}
    try:
        async with pybotters.Client(apis=local_apis, base_url=base_url, headers=demo_headers) as client:
            resp = await client.get(
                "/api/v2/mix/position/all-position",
                params={"productType": normalized_product_type, "marginCoin": margin_coin},
                headers=demo_headers,
            )
            data = await resp.json()
    except Exception as exc:
        discord.print_log(f"Bitget fetch_all_position_symbols error: {exc}")
        return []

    if not isinstance(data, dict) or data.get("code") != "00000":
        return []

    payload = data.get("data") or []
    if isinstance(payload, dict):
        items = payload.get("positions") or payload.get("list") or []
        if not isinstance(items, list):
            items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    symbols: list[str] = []
    for item in items:
        try:
            size = float(item.get("total", 0.0))
        except (TypeError, ValueError):
            size = 0.0
        if size > 0:
            sym = normalize_symbol(item.get("symbol", ""))
            if sym:
                symbols.append(sym)
    return list(dict.fromkeys(symbols))


async def fetch_all_position_symbols(category: str, settle_coin: str, mode: str) -> list[str]:
    try:
        return await fetch_all_position_symbols_bitget('USDT-FUTURES', settle_coin, bitget_mode)
    except Exception as exc:
        discord.print_log(f"[BITGET] fetch_all_position_symbols_bitget failed: {exc}")
        return []


async def flatten_current_position_bitget(
    product_type: str,
    symbol: str,
    coin: str,
    mode: str,
    reason: str = "",
    take_profit_pct: float = 0.0015,
    force_market: bool = False,
) -> bool:
    local_api = api_bitget_helper(symbol, product_type, coin, mode)
    position = await local_api.get_positions()
    buy_qty = float(position.get("buy", 0.0))
    sell_qty = float(position.get("sell", 0.0))
    margin_mode = position.get('margin_mode', 'crossed')
    
    best_ask: float = 0.0
    best_bid: float = 0.0
    if buy_qty <= 0 and sell_qty <= 0:
        return True
    discord.print_log(f"[BITGET] {reason}: starting flatten (buy={buy_qty}, sell={sell_qty}).")
    if is_air:
        discord.print_log(f"[AIR MODE] Bitget flatten execution skipped: Reason: {reason} (buy={buy_qty}, sell={sell_qty}) (Mock only)")
        return True

    try:
        async with pybotters.Client(apis=local_api._apis, base_url=local_api.base_url, headers=local_api._headers()) as client:
            ticker_res = await client.get(
                "/api/v2/mix/market/ticker",
                params={"symbol": local_api.symbol, "productType": local_api._product_type_param},
                headers=local_api._headers(),
            )
            ticker_json = await ticker_res.json()
            tick = {}
            if isinstance(ticker_json, dict):
                raw = ticker_json.get("data")
                if isinstance(raw, list):
                    tick = raw[0] if raw else {}
                elif isinstance(raw, dict):
                    tick = raw
            best_ask = float(tick.get("bestAsk") or tick.get("askPr") or 0.0)
            best_bid = float(tick.get("bestBid") or tick.get("bidPr") or 0.0)
    except Exception as exc:
        discord.print_log(f"[BITGET] ticker fetch error for flatten - {exc}")

    unrealised_profit = float(position.get("profit", 0.0))
    long_in_profit = buy_qty > 0 and unrealised_profit > 0
    short_in_profit = sell_qty > 0 and unrealised_profit > 0

    async def wait_flatten_completion(max_wait: float = 30.0, poll_interval: float = 2.0) -> bool:
        deadline = time.time() + max_wait
        while time.time() < deadline:
            await asyncio.sleep(poll_interval)
            pos_snapshot = await local_api.get_positions()
            open_orders = await local_api.get_open_orders()
            has_position = bool(pos_snapshot and (pos_snapshot.get("buy", 0) > 0 or pos_snapshot.get("sell", 0) > 0))
            has_orders = bool(open_orders)
            if not has_position and not has_orders:
                return True
        return False

    async def place_market_flatten(side: str, qty: float) -> bool:
        if qty <= 0:
            return True
        side_value = "buy" if side == "Sell" else "sell"
        modes_to_try = [margin_mode, "isolated" if margin_mode == "crossed" else "crossed"]

        for mode_attempt in modes_to_try:
            async with pybotters.Client(apis=local_api._apis, base_url=local_api.base_url, headers=local_api._headers()) as client:
                try:
                    order_payload = {
                        "symbol": local_api.symbol,
                        "productType": local_api._product_type_param,
                        "marginCoin": coin,
                        "marginMode": mode_attempt,
                        "size": qty,
                        "side": side_value,
                        "tradeSide": "close",
                        "orderType": "market",
                    }
                    res = await client.post("/api/v2/mix/order/place-order", data=order_payload, headers=local_api._headers())
                    data = await res.json()
                except Exception as exc:
                    continue
            code = data.get("code")
            if code == "00000":
                return True
            elif code == "22002":
                continue
        return False

    async def place_flatten_order(side: str, qty: float, entry_price: float) -> bool:
        if qty <= 0 or entry_price <= 0:
            return True
        tp_multiplier = 1 + take_profit_pct if side == "Sell" else 1 - take_profit_pct
        target_price = entry_price * tp_multiplier
        tif = "post_only"
        if side == "Sell" and long_in_profit and best_bid > 0:
            target_price = best_bid
            tif = "ioc"
        elif side == "Buy" and short_in_profit and best_ask > 0:
            target_price = best_ask
            tif = "ioc"
        side_value = "buy" if side == "Sell" else "sell"
        modes_to_try = [margin_mode, "isolated" if margin_mode == "crossed" else "crossed"]

        for mode_attempt in modes_to_try:
            async with pybotters.Client(apis=local_api._apis, base_url=local_api.base_url, headers=local_api._headers()) as client:
                try:
                    order_payload = {
                        "symbol": local_api.symbol,
                        "productType": local_api._product_type_param,
                        "marginCoin": coin,
                        "marginMode": mode_attempt,
                        "size": qty,
                        "side": side_value,
                        "tradeSide": "close",
                        "orderType": "limit",
                        "price": target_price,
                        "timeInForce": tif,
                    }
                    res = await client.post("/api/v2/mix/order/place-order", data=order_payload, headers=local_api._headers())
                    data = await res.json()
                except Exception as exc:
                    continue
            code = data.get("code")
            if code == "00000":
                return True
            elif code == "22002":
                continue
        return False

    await local_api.active_order_cancel()
    await asyncio.sleep(1)

    if force_market:
        position = await local_api.get_positions()
        buy_qty = float(position.get("buy", 0.0))
        sell_qty = float(position.get("sell", 0.0))
        if buy_qty <= 0 and sell_qty <= 0:
            return True
        long_closed = await place_market_flatten("Sell", buy_qty)
        short_closed = await place_market_flatten("Buy", sell_qty)
        if long_closed and short_closed:
            return await wait_flatten_completion(max_wait=15.0, poll_interval=2.0)
        return False

    long_closed = await place_flatten_order("Sell", buy_qty, float(position.get("buy_pos", 0.0)))
    short_closed = await place_flatten_order("Buy", sell_qty, float(position.get("sell_pos", 0.0)))
    if long_closed and short_closed:
        return await wait_flatten_completion()
    return False


async def flatten_current_position(
    category: str,
    symbol: str,
    coin: str,
    mode: str,
    reason: str = "",
    take_profit_pct: float = 0.0015,
    force_market: bool = False,
) -> bool:
    try:
        return await flatten_current_position_bitget('USDT-FUTURES', symbol, coin, bitget_mode, reason, take_profit_pct, force_market)
    except Exception as exc:
        discord.print_log(f"[BITGET] Bitget flatten error for {symbol}: {exc}")
        return False


async def flatten_all_positions(
    category: str,
    coin: str,
    mode: str,
    reason: str = "",
    take_profit_pct: float = 0.0015,
    force_market: bool = False,
) -> bool:
    try:
        return await flatten_all_positions_bitget('USDT-FUTURES', coin, bitget_mode, reason, take_profit_pct, force_market)
    except Exception as exc:
        discord.print_log(f"[BITGET] Bitget flatten_all failed: {exc}")
        return False


async def flatten_all_positions_bitget(
    product_type: str,
    coin: str,
    mode: str,
    reason: str = "",
    take_profit_pct: float = 0.0015,
    force_market: bool = False,
) -> bool:
    symbols = await fetch_all_position_symbols_bitget(product_type, coin, mode)
    if not symbols:
        return True
    all_ok = True
    for sym in symbols:
        ok = await flatten_current_position_bitget(product_type, sym, coin, mode, reason, take_profit_pct, force_market)
        all_ok = all_ok and ok
    return all_ok
