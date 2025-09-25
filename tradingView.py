import yfinance as yf
import pandas as pd
import json
import numpy as np
from typing import List, Tuple, Dict, Any


def is_pivot_high(data: pd.DataFrame, index: int, period: int) -> bool:
    """Checks for a pivot high with a given lookback period."""
    if index < period or index >= len(data) - period:
        return False
    return data['High'].iloc[index] == data['High'].iloc[index - period: index + period + 1].max()


def is_pivot_low(data: pd.DataFrame, index: int, period: int) -> bool:
    """Checks for a pivot low with a given lookback period."""
    if index < period or index >= len(data) - period:
        return False
    return data['Low'].iloc[index] == data['Low'].iloc[index - period: index + period + 1].min()


def calculate_pivots(hist: pd.DataFrame, pivot_period: int = 10) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Calculates and returns a list of pivot high and pivot low points.

    Args:
        hist (pd.DataFrame): The historical price data.
        pivot_period (int): The lookback/lookahead period for pivots.

    Returns:
        tuple[list, list]: A tuple containing two lists:
                           - The first list is of dictionaries for pivot highs.
                           - The second list is of dictionaries for pivot lows.
    """
    pivot_highs = []
    pivot_lows = []

    # Loop through the entire DataFrame and let is_pivot_* functions handle boundaries.
    for i in range(len(hist)):
        if is_pivot_high(hist, i, pivot_period):
            pivot_highs.append({'price': hist['High'].iloc[i], 'index': i})
        elif is_pivot_low(hist, i, pivot_period):
            pivot_lows.append({'price': hist['Low'].iloc[i], 'index': i})

    return pivot_highs, pivot_lows

def find_nearest_sr_levels(hist: pd.DataFrame,pivot_points: list, channel_width_pct: int = 1, loopback: int = 490,
                           min_strength: int = 2,pivot_period: int = 10) -> List[Tuple[float, float, int]]:
    """
    Finds support and resistance levels based on pivot points,
    and returns their location for debugging.

    Args:
        hist (pd.DataFrame): Historical price data.
        channel_width_pct (int): Percentage of total range for channel width.
        loopback (int): Number of historical bars to look back.
        min_strength (int): Minimum number of pivots required for a level.

    Returns:
        list: A list of tuples, where each tuple contains
              (min_price, max_price, most_recent_pivot_index).
    """

    #get pivot points
    end_index = max(pivot_period, len(hist) - loopback - 1)
    # 2. Group pivots into S/R channels and calculate their strength
    potential_sr_channels = []

    unique_pivot_prices = sorted(list(set(p['price'] for p in pivot_points)))

    historical_high = hist['High'].tail(500).max()
    historical_low = hist['Low'].tail(500).min()
    channel_width = (historical_high - historical_low) * channel_width_pct / 100

    for pivot_price in unique_pivot_prices:
        channel_pivots_prices = [p for p in unique_pivot_prices if abs(p - pivot_price) <= channel_width]

        if len(channel_pivots_prices) >= min_strength:
            channel_min = min(channel_pivots_prices)
            channel_max = max(channel_pivots_prices)

            pivot_strength = len(channel_pivots_prices) * 20

            bar_interaction_strength = 0
            for i in range(len(hist) - 1, end_index, -1):
                high_price = hist['High'].iloc[i]
                low_price = hist['Low'].iloc[i]

                if (high_price <= channel_max and high_price >= channel_min) or \
                        (low_price <= channel_max and low_price >= channel_min):
                    bar_interaction_strength += 1

            total_strength = pivot_strength + bar_interaction_strength

            channel_indices = [p['index'] for p in pivot_points if p['price'] in channel_pivots_prices]
            most_recent_index = min(channel_indices)

            potential_sr_channels.append({
                'min': channel_min,
                'max': channel_max,
                'strength': total_strength,
                'pivots': set(channel_pivots_prices),
                'index': most_recent_index
            })

    # 3. Sort channels by strength and filter for non-overlapping levels
    potential_sr_channels.sort(key=lambda x: x['strength'], reverse=True)

    final_sr_levels = []
    used_pivots_prices = set()

    for channel in potential_sr_channels:
        if not channel['pivots'].intersection(used_pivots_prices):
            final_sr_levels.append((channel['min'], channel['max'], channel['index']))
            used_pivots_prices.update(channel['pivots'])

    return final_sr_levels


def find_unbroken_trendlines(hist: pd.DataFrame, pivot_highs: list[dict]) -> list[dict]:
    """
    Finds all valid, unbroken resistance trendlines from a list of pivot highs.

    Args:
        hist (pd.DataFrame): Historical price data.
        pivot_highs (list): List of pivot high dictionaries.

    Returns:
        list: A list of dictionaries, each representing a valid trendline with its slope and intercept.
    """
    valid_trendlines = []

    for i in range(len(pivot_highs) - 1):
        for j in range(i + 1, len(pivot_highs)):
            pivot1 = pivot_highs[i]
            pivot2 = pivot_highs[j]

            # Rule 1: Must be a downtrend (resistance line)
            if pivot2['price'] >= pivot1['price']:
                continue

            # Calculate the line's equation (y = mx + b)
            slope = (pivot2['price'] - pivot1['price']) / (pivot2['index'] - pivot1['index'])
            intercept = pivot1['price'] - slope * pivot1['index']

            # Rule 2: Check for a breach between the two pivots
            is_valid = True
            for k in range(pivot1['index'] + 1, pivot2['index']):
                projected_price = slope * k + intercept
                if hist['High'].iloc[k] > projected_price:
                    is_valid = False
                    break

            if is_valid:
                valid_trendlines.append({
                    'slope': slope,
                    'intercept': intercept,
                    'end_index': pivot2['index'],
                    'start_price':pivot1['price'],
                    'end_price': pivot2['price']
                })

    return valid_trendlines


def run_stock_scan() -> None:
    """
    Main function to perform stock screening based on proximity to key levels.
    """
    THRESHHOLD_VAL = 2.5

    with open("filtered_nyse_nasdaq_stocks.json", "r") as f:
        data = json.load(f)

    TICKERS = data
    results = []

    for symbol in TICKERS:
        print(f"Processing {symbol}...")
        try:
            stock = yf.Ticker(symbol)

            # --- Data Retrieval and Initial Checks ---
            info = stock.get_info()
            exchange = info.get('exchange', 'N/A')
            if exchange == 'NMS':
                tv_exchange = 'NASDAQ'
            elif exchange == 'NYQ':
                tv_exchange = 'NYSE'
            else:
                tv_exchange = exchange

            hist = stock.history(period="4y",auto_adjust=False) #to ensure both your Python script and your TradingView chart are using the same type of data.

            if hist.empty or len(hist) < 200:
                print(f"Skipping {symbol}: Not enough data.")
                continue

            market_cap = info.get('marketCap', 0)
            if not market_cap or market_cap < 1e9:
                print(f"Skipping {symbol}: Market Cap missing or under $1B.")
                continue

            current_price = hist["Close"].iloc[-1]
            last_index = len(hist) - 1

            # --- Technical Indicator Calculations ---
            hist["avg_volume_14"] = hist["Volume"].rolling(window=14).mean()
            current_volume = hist["Volume"].iloc[-1]
            avg_volume_today = hist["avg_volume_14"].iloc[-1]
            rising_volume_2d = False
            if len(hist) >= 2:
                avg_volume_yesterday = hist["avg_volume_14"].iloc[-2]
                rising_volume_1 = current_volume > avg_volume_today
                rising_volume_2 = hist["Volume"].iloc[-2] > avg_volume_yesterday
                rising_volume_2d = rising_volume_1 and rising_volume_2

            hist['tr'] = np.maximum(
                hist['High'] - hist['Low'],
                np.maximum(
                    abs(hist['High'] - hist['Close'].shift(1)),
                    abs(hist['Low'] - hist['Close'].shift(1))
                )
            )
            alpha = 1 / 14
            hist['atr'] = hist['tr'].ewm(alpha=alpha, adjust=False).mean()
            atr_value = hist['atr'].iloc[-1]
            if pd.isna(atr_value):
                continue

            sma200 = hist["Close"].rolling(window=200).mean().iloc[-1]
            sma150 = hist["Close"].rolling(window=150).mean().iloc[-1]
            hist["TPV"] = (hist["High"] + hist["Low"] + hist["Close"]) * hist["Volume"] / 3
            vwap_150 = hist["TPV"].tail(150).sum() / hist["Volume"].tail(150).sum()

            # --- Proximity Checks & Condition Evaluation ---
            is_near_sma150 = current_price < sma150 and abs(
                current_price - sma150) * 100 / current_price <= THRESHHOLD_VAL and abs(
                current_price - sma150)  <= atr_value
            is_near_vwap = current_price < vwap_150 and abs(
                current_price - vwap_150) * 100 / current_price <= THRESHHOLD_VAL and abs(
                current_price - vwap_150)  <= atr_value
            is_near_sma200 = current_price < sma200 and abs(
                current_price - sma200) * 100 / current_price <= THRESHHOLD_VAL and abs(
                current_price - sma200)  <= atr_value

            reasons = []
            if is_near_sma150: reasons.append("Near SMA150")
            if is_near_vwap: reasons.append("Near VWAP")
            if is_near_sma200: reasons.append("Near SMA200")

            pivot_highs, pivot_lows = calculate_pivots(hist)
            all_pivots =[]
            all_pivots.extend(pivot_lows)
            all_pivots.extend(pivot_highs)
            #resistance trendline check
            unbroken_trendlines = find_unbroken_trendlines(hist, pivot_highs)

            if unbroken_trendlines:
                # Sort by end_index to check the most recent lines first
                unbroken_trendlines.sort(key=lambda x: x['end_index'], reverse=True)

                current_bar_index = len(hist) - 1
                is_near_resistance = 0
                distance=0
                for line in unbroken_trendlines:
                    # Project the line to the current bar
                    projected_price = line['slope'] * current_bar_index + line['intercept']
                    Trend_dist=abs(projected_price - current_price)*100/current_price
                    # Check if the price is below the line AND within one ATR distance
                    if current_price < projected_price and abs(projected_price - current_price) < atr_value*2:
                        is_near_resistance = projected_price
                        distance=Trend_dist
                        reasons.append("Near Trendline")
                        break

            # S/R Check
            sr_levels = find_nearest_sr_levels(hist,all_pivots)
            is_near_sr = False
            sr_range_str, sr_proximity_pct, sr_candles_ago = "0", np.nan, np.nan

            for sr_min, sr_max, sr_index in sr_levels:
                if sr_min <= current_price <= sr_max:
                    if current_price < (sr_min + sr_max) / 2 and abs(current_price - (sr_min + sr_max) / 2)  <= atr_value and current_price > sma150:
                        is_near_sr = True
                        reasons.append("Near S/R")
                        sr_range_str = f"{sr_min:.2f}-{sr_max:.2f}"
                        sr_proximity_pct = (abs(current_price - sr_max) / current_price) * 100
                        sr_candles_ago = last_index - sr_index
                        break

            if not is_near_sr:
                closest_level, min_distance, sr_index = None, float('inf'), None
                for sr_min, sr_max, idx in sr_levels:
                    if current_price < sr_min:
                        dist = abs(sr_min - current_price)
                        if dist < min_distance:
                            min_distance = dist
                            closest_level = sr_min
                            sr_index = idx
                if closest_level is not None:
                    sr_proximity_pct = (min_distance / current_price) * 100
                    if sr_proximity_pct <= THRESHHOLD_VAL and current_price > sma150 and closest_level > sma150 and sr_proximity_pct <=(atr_value / current_price) * 100:
                        is_near_sr = True
                        reasons.append("Near S/R")
                        sr_range_str = f"{closest_level:.2f}"
                        sr_candles_ago = last_index - sr_index

            # --- Final Result Aggregation ---
            if (is_near_sma150 or is_near_vwap or is_near_sma200 or is_near_sr or is_near_resistance>0) and avg_volume_today >= 300000 :

                VWAP_distance=abs(current_price - vwap_150) / current_price * 100
                SMA150_distance=abs(current_price - sma150) / current_price * 100
                SMA200_distance = abs(current_price - sma200) / current_price * 100

                reason_str = ", ".join(reasons)
                lookback = 2
                bullish_streak = (hist["Close"].tail(lookback) > hist["Open"].tail(lookback)).all()
                rising_volume = (hist["Volume"].tail(lookback).diff().dropna() > 0).all()


                results.append({
                    "Ticker": f'=HYPERLINK("https://www.tradingview.com/chart/?symbol={tv_exchange}:{symbol}", "{symbol}")',
                    "Exchange": tv_exchange,
                    "Reason": reason_str,
                    "Close": current_price,
                    "ATR_Pct": (atr_value / current_price) * 100,
                    "VWAP150": vwap_150,
                    "VWAP_distance": VWAP_distance,
                    "SMA150": sma150,
                    "SMA150_distance": SMA150_distance,
                    "SMA200": sma200,
                    "SMA200_distance": SMA200_distance,
                    "SR_Range": sr_range_str,
                    "SR_distance": sr_proximity_pct,
                    "Trendline": is_near_resistance,
                    "Trendline_distance": distance,
                    "Bull Streak": bullish_streak,
                    "Rising Vol": rising_volume,
                    "Vol > Avg (2D)": rising_volume_2d,
                    "Avg_Volume": avg_volume_today,
                })

        except Exception as e:
            print(f"Skipping {symbol}: An error occurred: {e}")

    # Save results
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv("scan_results.csv", index=False)
        print("Results saved to scan_results.csv")
    else:
        print("No tickers met the criteria.")


if __name__ == "__main__":
    run_stock_scan()
