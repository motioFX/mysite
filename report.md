# Monday Overnight Anomaly Backtest Report

## Objective
Analyze the performance of the "Buy at Monday Close, Sell at Tuesday Open" strategy across 11 specified Japanese ETFs over the available historical period (up to 5 years).

## Methodology
- **Entry**: Buy at Market on Monday Close (15:00)
- **Exit**: Sell at Market on Tuesday Open (09:00)
- **Metrics**: Total Return, Win Rate, Profit Factor, Max Drawdown (MDD)

## Results Summary

| ETF Code   | ETF Name                        |   Total Trades |   Win Rate (%) |   Profit Factor |   Total Return (%) |   Max Drawdown (%) |
|:-----------|:--------------------------------|---------------:|---------------:|----------------:|-------------------:|-------------------:|
| 1579       | 日経平均レバレッジ・インデックス連動型上場投資信託       |            217 |          65.90 |            2.60 |             251.50 |              -6.37 |
| 1570       | NEXT FUNDS 日経平均レバレッジ・連動型上場投信    |            218 |          64.68 |            2.44 |             218.15 |              -6.50 |
| 2869       | iフリーETF ナスダック100レバレッジ           |            151 |          58.94 |            2.09 |             115.54 |             -11.60 |
| 213A       | 上場インデックスファンド日経半導体株              |             74 |          68.92 |            5.55 |             111.02 |              -4.88 |
| 221A       | MAXIS 日経半導体株上場投信                |             74 |          71.62 |            4.21 |              98.08 |              -3.69 |
| 2239       | 上場インデックスファンド米国株式（S&P500）レバレッジ2倍 |            136 |          58.09 |            2.67 |              89.73 |              -8.32 |
| 2516       | 東証グロース市場250指数連動型上場投信            |            218 |          59.63 |            2.24 |              87.09 |              -7.66 |
| 1475       | iシェアーズ・コア TOPIX ETF             |            218 |          69.72 |            2.68 |              72.54 |              -2.97 |
| 2237       | iフリーETF S&P500レバレッジ             |            142 |          61.27 |            2.01 |              63.05 |              -9.42 |
| 2238       | iフリーETF S&P500インバース             |            142 |          38.03 |            0.47 |             -27.36 |             -28.94 |
| 1357       | NEXT FUNDS 日経平均ダブルインバース・連動型上場投信 |            218 |          31.65 |            0.40 |             -73.53 |             -73.16 |

## Observations

Based on the 5-year backtest data of the "Monday Close Buy -> Tuesday Open Sell" strategy across the 11 targeted ETFs:

### 1. Best Performing ETFs (Most Suited)
*   **Nikkei Leveraged ETFs (1579, 1570):** These are by far the best performers. **1579** and **1570** show the highest total returns, excellent win rates (~65%), and very strong Profit Factors. Their Max Drawdowns remain relatively contained. The "Monday Anomaly" driven by Chicago futures overnight heavily favors these instruments.
*   **Semiconductor ETFs (213A, 221A):** Despite having fewer trades (shorter history), these exhibited the highest win rates (~69% - 71%) and extraordinary Profit Factors (5.55 and 4.21). They appear to capture the overnight gap strongly and cleanly, with very low Max Drawdowns.

### 2. Moderate Performers
*   **US Leveraged Tech/Broad Market (2869, 2239, 2237):** These US-focused leveraged ETFs also showed positive expectancy with good total returns, though their Max Drawdowns are notably higher than their domestic counterparts, suggesting they can suffer rougher overnight gaps against the position.
*   **Broad Japanese Markets (1475, 2516):** TOPIX (1475) shows a very high win rate (69.27%) and the lowest Max Drawdown, making it the safest, most consistent, but lower-yielding vehicle. Growth 250 (2516) also works reasonably well but with lower efficiency.

### 3. Worst Performing ETFs (Least Suited)
*   **Inverse ETFs (1357, 2238):** As expected, since the anomaly is heavily biased towards an overnight *upward* gap, taking a long position on *inverse* ETFs (effectively shorting the market) over this window performs terribly. **1357** (Nikkei Double Inverse) suffered a massive negative return and a 31% win rate. **This strategy should explicitly avoid inverse ETFs.**

**Conclusion:** The strategy is highly robust for leveraged long Japanese equities (specifically Nikkei 225 Leveraged like 1579/1570 and Semiconductor ETFs). US leveraged and broad Japanese indices also work but offer different risk/reward profiles. Inverse ETFs are strictly contraindicated.
