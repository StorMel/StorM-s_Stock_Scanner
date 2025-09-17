import yfinance as yf
import pandas as pd
import json
import numpy as np


def find_nearest_sr_levels(hist: pd.DataFrame, channel_width_pct: int = 1, loopback: int = 250,
                           min_strength: int = 1) -> list:
    """
    Finds support and resistance levels based on pivot points.

    Args:
        hist (pd.DataFrame): Historical price data.
        channel_width_pct (int): Percentage of total range for channel width.
        loopback (int): Number of historical bars to look back.
        min_strength (int): Minimum number of pivots required for a level.

    Returns:
        list: A list of tuples, where each tuple contains (min_price, max_price) of a S/R level.
    """
    pivot_lookback = 5

    # Identify pivot points using a rolling window
    hist['pivot_high'] = hist['High'].rolling(window=pivot_lookback * 2 + 1, center=True).max() == hist['High']
    hist['pivot_low'] = hist['Low'].rolling(window=pivot_lookback * 2 + 1, center=True).min() == hist['Low']

    pivot_vals = []

    # Extract unique pivot values from the lookback period
    for i in range(len(hist) - 1, len(hist) - loopback - 1, -1):
        if i >= 0 and i < len(hist):
            if hist['pivot_high'].iloc[i] and hist['High'].iloc[i] not in pivot_vals:
                pivot_vals.append(hist['High'].iloc[i])
            if hist['pivot_low'].iloc[i] and hist['Low'].iloc[i] not in pivot_vals:
                pivot_vals.append(hist['Low'].iloc[i])

    if not pivot_vals:
        return []

    # Group pivots into S/R channels
    sr_levels = []
    pivot_vals.sort()
    processed_pivots = set()

    for pivot in pivot_vals:
        if pivot in processed_pivots:
            continue

        channel_width = (hist['High'].tail(300).max() - hist['Low'].tail(300).min()) * channel_width_pct / 100
        channel_pivots = [p for p in pivot_vals if abs(p - pivot) <= channel_width]

        if len(channel_pivots) >= min_strength:
            channel_min = min(channel_pivots)
            channel_max = max(channel_pivots)
            sr_levels.append((channel_min, channel_max))
            processed_pivots.update(channel_pivots)

    return sr_levels


def run_stock_scan() -> None:
    """
    Main function to perform stock screening based on proximity to key levels.
    """
    THRESHHOLD_VAL = 2.5

    with open("filtered_nyse_nasdaq_stocks.json", "r") as f:
        data = json.load(f)

    TICKERS = data#["EVGO", "UBER", "JDCMF", "IBRX", "CRF", "CAMT"]  #
    results = []

    for symbol in TICKERS:
        print(f"Processing {symbol}...")
        try:
            stock = yf.Ticker(symbol)

            # Retrieve the stock's information, which contains the exchange
            info = stock.get_info()
            exchange = info.get('exchange', 'N/A')

            # Convert Yahoo's exchange codes to TradingView's format
            # This is a crucial step for the hyperlink to work
            if exchange == 'NMS':
                tv_exchange = 'NASDAQ'
            elif exchange == 'NYQ':
                tv_exchange = 'NYSE'
            else:
                tv_exchange = exchange  # Use the original name if not a common US exchange
                # ASE, NGM, OTC

            hist = stock.history(period="1y")
            info = stock.get_info()
            market_cap = info.get('marketCap', 0)

            if not market_cap or market_cap < 1e9 or hist.empty or len(hist) < 200:
                print(f"Skipping {symbol}: Market Cap missing or under $1B or insufficient data.")
                continue

            current_price = hist["Close"].iloc[-1]
            current_volume = hist["Volume"].iloc[-1]

            # --- Start: Added Volume Logic ---
            hist["avg_volume"] = hist["Volume"].rolling(window=14).mean()

            # Get the average volume for the last two days
            avg_volume_today = hist["avg_volume"].iloc[-1]
            avg_volume_yesterday = hist["avg_volume"].iloc[-2]

            # Check if volume on the last two days was greater than their own historical average
            rising_volume_1 = hist["Volume"].iloc[-1] > avg_volume_today
            rising_volume_2 = hist["Volume"].iloc[-2] > avg_volume_yesterday

            # Combine the two checks into a single boolean
            rising_volume_2d = rising_volume_1 and rising_volume_2
            # --- End: Added Volume Logic ---

            # Calculate True Range (TR)
            hist['tr'] = np.maximum(
                hist['High'] - hist['Low'],
                np.maximum(
                    abs(hist['High'] - hist['Close'].shift(1)),
                    abs(hist['Low'] - hist['Close'].shift(1))
                )
            )

            # Calculate Average True Range (ATR) using Wilder's smoothing
            alpha = 1 / 14
            hist['atr'] = hist['tr'].ewm(alpha=alpha, adjust=False).mean()
            atr_value = hist['atr'].iloc[-1]

            if pd.isna(atr_value):
                continue

            # Calculate key moving averages and VWAP
            sma200 = hist["Close"].rolling(window=200).mean().iloc[-1]
            sma150 = hist["Close"].rolling(window=150).mean().iloc[-1]
            hist["TPV"] = (hist["High"] + hist["Low"] + hist["Close"]) * hist["Volume"] / 3
            vwap_150 = hist["TPV"].tail(150).sum() / hist["Volume"].tail(150).sum()

            # Check if current price is near and BELOW key levels
            is_near_sma150 = False
            is_near_vwap = False
            is_near_sma200 = False
            is_near_sr = False

            reasons = []

            # SMA150 check
            if current_price < sma150 and abs(current_price - sma150) * 100 / current_price <= THRESHHOLD_VAL:
                is_near_sma150 = True
                reasons.append("Near SMA150")

            # VWAP check
            if current_price < vwap_150 and abs(current_price - vwap_150) * 100 / current_price <= THRESHHOLD_VAL:
                is_near_vwap = True
                reasons.append("Near VWAP")

            # SMA200 check
            if current_price < sma200 and abs(current_price - sma200) * 100 / current_price <= THRESHHOLD_VAL:
                is_near_sma200 = True
                reasons.append("Near SMA200")

            # S/R proximity check
            sr_levels = find_nearest_sr_levels(hist)
            sr_range_str, sr_proximity_pct = "0", np.nan

            # Check if price is within an S/R range and below it
            for sr_min, sr_max in sr_levels:
                if sr_min <= current_price <= sr_max:
                    # Check if the level is acting as resistance
                    if current_price < (sr_min + sr_max) / 2:
                        is_near_sr = True
                        reasons.append("Near S/R")
                        sr_range_str = f"{sr_min:.2f}-{sr_max:.2f}"
                        sr_proximity_pct = (abs(current_price - sr_max) / current_price) * 100
                        break

            # If not in a range, find the closest level and check if it's below
            if not is_near_sr:
                closest_level, min_distance = None, float('inf')

                for sr_min, sr_max in sr_levels:
                    # Only consider levels that are above the current price
                    if current_price < sr_min:
                        dist = abs(sr_min - current_price)
                        if dist < min_distance:
                            min_distance = dist
                            closest_level = sr_min

                if closest_level is not None:
                    sr_proximity_pct = (min_distance / current_price) * 100

                    if sr_proximity_pct <= THRESHHOLD_VAL and current_price > sma150 and closest_level > sma150:
                        is_near_sr = True
                        reasons.append("Near S/R")
                        sr_range_str = f"{closest_level:.2f}"

            # Log results if any condition is met
            if is_near_sma150 or is_near_vwap or is_near_sma200 or is_near_sr and avg_volume_today >= 300000:
                reason_str = ", ".join(reasons)

                lookback = 2
                bullish_streak = (hist["Close"].tail(lookback) > hist["Open"].tail(lookback)).all()
                rising_volume = (hist["Volume"].tail(lookback).diff().dropna() > 0).all()

                results.append({
                    "Ticker": f'=HYPERLINK("https://www.tradingview.com/chart/?symbol={tv_exchange}:{symbol}", "{symbol}")',
                    "Exchange": tv_exchange,
                    "Reason": reason_str,
                    "Close": current_price,
                    "VWAP150": vwap_150,
                    "VWAP_distance": abs(current_price - vwap_150) / current_price * 100,
                    "SMA150": sma150,
                    "SMA150_distance": abs(current_price - sma150) / current_price * 100,
                    "SMA200": sma200,
                    "SMA200_distance": abs(current_price - sma200) / current_price * 100,
                    "ATR_Pct": (atr_value / current_price) * 100,
                    "SR_Range": sr_range_str,
                    "SR_distance": sr_proximity_pct,
                    "Bull Streak": bullish_streak,
                    "Rising Vol": rising_volume,
                    "Vol > Avg": rising_volume_2d,
                    "Avg_Volume": avg_volume_today
                })

        except Exception as e:
            print(f"Skipping {symbol}: An error occurred: {e}")

    # Output results
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv("scan_results.csv", index=False)
        print("Results saved to scan_results.csv")
    else:
        print("No tickers met the criteria.")


if __name__ == "__main__":
    run_stock_scan()
