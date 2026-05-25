import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_last_monday(date_obj):
    # If the date + 7 days is in a different month, it is the last occurrence of that weekday in the month
    next_week = date_obj + pd.Timedelta(days=7)
    return date_obj.month != next_week.month

def calculate_mdd(equity_curve):
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

def main():
    tickers = ["1579.T", "1357.T"]

    # Download data from 2024-12-01 to ensure we have data around 2025-01-01
    print("Downloading historical data...")
    data = yf.download(tickers, start="2024-12-01", group_by='ticker', auto_adjust=True)

    # Process 1579 (Bull) and 1357 (Bear)
    bull_df = data["1579.T"].copy().dropna()
    bear_df = data["1357.T"].copy().dropna()

    # Filter dates from 2025-01-01
    bull_df = bull_df[bull_df.index >= '2025-01-01']
    bear_df = bear_df[bear_df.index >= '2025-01-01']

    # Get common dates to align the calendars
    common_dates = sorted(list(set(bull_df.index) | set(bear_df.index)))
    market_dates = pd.DatetimeIndex(common_dates)

    trades = []

    for i, current_date in enumerate(market_dates):
        # We only trade on Mondays
        if current_date.dayofweek != 0:
            continue

        # Check if tomorrow is a market day (Tuesday)
        if i + 1 >= len(market_dates):
            # No next day in the dataset, skip
            continue

        next_date = market_dates[i+1]

        # If the next market day is not Tuesday (e.g. Tuesday is a holiday), skip the trade
        if next_date.dayofweek != 1:
            continue

        # Determine if it's the last Monday
        last_mon = is_last_monday(current_date)

        # Determine ticker and direction
        if last_mon:
            ticker = "1357.T"
            df = bear_df
            trade_type = "Bear (1357)"
        else:
            ticker = "1579.T"
            df = bull_df
            trade_type = "Bull (1579)"

        # Verify we have data for both days for the selected ticker
        if current_date not in df.index or next_date not in df.index:
            continue

        entry_price = df.loc[current_date, 'Close']
        exit_price = df.loc[next_date, 'Open']

        # Some yf data can be single-value or series depending on exact download shape
        if isinstance(entry_price, pd.Series): entry_price = entry_price.iloc[0]
        if isinstance(exit_price, pd.Series): exit_price = exit_price.iloc[0]

        ret = (exit_price - entry_price) / entry_price

        trades.append({
            'Entry_Date': current_date,
            'Exit_Date': next_date,
            'Type': trade_type,
            'Return': ret
        })

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print("No trades executed.")
        return

    # Calculate performance metrics
    trades_df['Equity'] = (1 + trades_df['Return']).cumprod()

    total_trades = len(trades_df)
    wins = trades_df[trades_df['Return'] > 0]
    losses = trades_df[trades_df['Return'] <= 0]

    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    gross_profit = wins['Return'].sum()
    gross_loss = abs(losses['Return'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    total_return = trades_df['Equity'].iloc[-1] - 1
    mdd = calculate_mdd(trades_df['Equity'])

    # Text Output
    print("\n--- Backtest Results (2025-01-01 to Present) ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate * 100:.2f}%")
    print(f"Total Return (Cumulative): {total_return * 100:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: {mdd * 100:.2f}%")
    print("--------------------------------------------------\n")

    # Plotting
    plt.figure(figsize=(12, 6))

    # Reindex to Entry Date for plotting
    trades_df.set_index('Entry_Date', inplace=True)

    plt.plot(trades_df.index, trades_df['Equity'], label='Total Equity Curve', color='blue', linewidth=2)

    # Mark Bull and Bear trades
    bull_trades = trades_df[trades_df['Type'] == "Bull (1579)"]
    bear_trades = trades_df[trades_df['Type'] == "Bear (1357)"]

    plt.scatter(bull_trades.index, bull_trades['Equity'], color='green', marker='^', label='Bull Trade (1579)')
    plt.scatter(bear_trades.index, bear_trades['Equity'], color='red', marker='v', label='Bear Trade (1357)')

    plt.title('Equity Curve: Bull (Normal Mon) / Bear (Last Mon)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Multiplier)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('monthly_bear_equity.png')
    print("Saved plot to monthly_bear_equity.png")

if __name__ == "__main__":
    main()
