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

    TICKERS = data#["EVGO","UBER","JDCMF","IBRX","CRF","CAMT"]#
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
                #ASE, NGM, OTC

            hist = stock.history(period="1y")
            info = stock.get_info()
            market_cap = info.get('marketCap', 0)

            if not market_cap or market_cap < 1e9 or hist.empty or len(hist) < 200:
                print(f"Skipping {symbol}: Market Cap missing or under $1B or insufficient data.")
                continue

            current_price = hist["Close"].iloc[-1]

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
                    if sr_proximity_pct <= THRESHHOLD_VAL:
                        is_near_sr = True
                        reasons.append("Near S/R")
                        sr_range_str = f"{closest_level:.2f}"

            # Log results if any condition is met
            if is_near_sma150 or is_near_vwap or is_near_sma200 or is_near_sr:
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
                    "VWAP_Proximity_ATR_Pct": abs(current_price - vwap_150) / current_price * 100,
                    "SMA150": sma150,
                    "SMA150_Proximity_ATR_Pct": abs(current_price - sma150) / current_price * 100,
                    "SMA200": sma200,
                    "SMA200_Proximity_ATR_Pct": abs(current_price - sma200) / current_price * 100,
                    "ATR_Pct": (atr_value / current_price) * 100,
                    "SR_Range": sr_range_str,
                    "SR_Proximity_Pct": sr_proximity_pct,
                    "Bull Streak": bullish_streak,
                    "Rising Vol": rising_volume
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

#pine script
''' 

// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © LonesomeTheBlue

//@version=6
indicator('StorMs all in one', 'SRchannel', overlay = true, max_bars_back = 501)
prd = input.int(defval = 5, title = 'Pivot Period', minval = 4, maxval = 30, group = 'Settings 🔨', tooltip = 'Used while calculating Pivot Points, checks left&right bars')
ppsrc = input.string(defval = 'High/Low', title = 'Source', options = ['High/Low', 'Close/Open'], group = 'Settings 🔨', tooltip = 'Source for Pivot Points')
ChannelW = input.int(defval = 1, title = 'Maximum Channel Width %', minval = 1, maxval = 8, group = 'Settings 🔨', tooltip = 'Calculated using Highest/Lowest levels in 300 bars')
minstrength = input.int(defval = 1, title = 'Minimum Strength', minval = 1, group = 'Settings 🔨', tooltip = 'Channel must contain at least 2 Pivot Points')
maxnumsr = input.int(defval = 10, title = 'Maximum Number of S/R', minval = 1, maxval = 10, group = 'Settings 🔨', tooltip = 'Maximum number of Support/Resistance Channels to Show') - 1
loopback = input.int(defval = 250, title = 'Loopback Period', minval = 100, maxval = 400, group = 'Settings 🔨', tooltip = 'While calculating S/R levels it checks Pivots in Loopback Period')
res_col = input.color(defval = color.new(#fbff00, 40), title = 'Resistance Color', group = 'Colors 🟡🟢🟣')
sup_col = input.color(defval = color.new(#fbff00, 40), title = 'Support Color', group = 'Colors 🟡🟢🟣')
inch_col = input.color(defval = color.new(#fbff00, 63), title = 'Color When Price in Channel', group = 'Colors 🟡🟢🟣')
showpp = input.bool(defval = false, title = 'Show Pivot Points', group = 'Extras ⏶⏷')
showsrbroken = input.bool(defval = false, title = 'Show Broken Support/Resistance', group = 'Extras ⏶⏷')
showthema1en = input.bool(defval = false, title = 'MA 1', inline = 'ma1')
showthema1len = input.int(defval = 50, title = '', inline = 'ma1')
showthema1type = input.string(defval = 'SMA', title = '', options = ['SMA', 'EMA'], inline = 'ma1')
showthema2en = input.bool(defval = false, title = 'MA 2', inline = 'ma2')
showthema2len = input.int(defval = 200, title = '', inline = 'ma2')
showthema2type = input.string(defval = 'SMA', title = '', options = ['SMA', 'EMA'], inline = 'ma2')

ma1 = showthema1en ? showthema1type == 'SMA' ? ta.sma(close, showthema1len) : ta.ema(close, showthema1len) : na
ma2 = showthema2en ? showthema2type == 'SMA' ? ta.sma(close, showthema2len) : ta.ema(close, showthema2len) : na

plot(ma1, color = not na(ma1) ? color.blue : na)
plot(ma2, color = not na(ma2) ? color.red : na)

// get Pivot High/low
float src1 = ppsrc == 'High/Low' ? high : math.max(close, open)
float src2 = ppsrc == 'High/Low' ? low : math.min(close, open)
float ph = ta.pivothigh(src1, prd, prd)
float pl = ta.pivotlow(src2, prd, prd)

// draw Pivot points
plotshape(bool(ph) and showpp, text = 'H', style = shape.labeldown, color = na, textcolor = color.new(color.red, 0), location = location.abovebar, offset = -prd)
plotshape(bool(pl) and showpp, text = 'L', style = shape.labelup, color = na, textcolor = color.new(color.lime, 0), location = location.belowbar, offset = -prd)

//calculate maximum S/R channel width
prdhighest = ta.highest(300)
prdlowest = ta.lowest(300)
cwidth = (prdhighest - prdlowest) * ChannelW / 100

// get/keep Pivot levels
var pivotvals = array.new_float(0)
var pivotlocs = array.new_float(0)
if bool(ph) or bool(pl)
    array.unshift(pivotvals, bool(ph) ? ph : pl)
    array.unshift(pivotlocs, bar_index)
    for x = array.size(pivotvals) - 1 to 0 by 1
        if bar_index - array.get(pivotlocs, x) > loopback // remove old pivot points
            array.pop(pivotvals)
            array.pop(pivotlocs)
            continue
        break

//find/create SR channel of a pivot point
get_sr_vals(ind) =>
    float lo = array.get(pivotvals, ind)
    float hi = lo
    int numpp = 0
    for y = 0 to array.size(pivotvals) - 1 by 1
        float cpp = array.get(pivotvals, y)
        float wdth = cpp <= hi ? hi - cpp : cpp - lo
        if wdth <= cwidth // fits the max channel width?
            if cpp <= hi
                lo := math.min(lo, cpp)
                lo
            else
                hi := math.max(hi, cpp)
                hi

            numpp := numpp + 20 // each pivot point added as 20
            numpp
    [hi, lo, numpp]

// keep old SR channels and calculate/sort new channels if we met new pivot point
var suportresistance = array.new_float(20, 0) // min/max levels
changeit(x, y) =>
    tmp = array.get(suportresistance, y * 2)
    array.set(suportresistance, y * 2, array.get(suportresistance, x * 2))
    array.set(suportresistance, x * 2, tmp)
    tmp := array.get(suportresistance, y * 2 + 1)
    array.set(suportresistance, y * 2 + 1, array.get(suportresistance, x * 2 + 1))
    array.set(suportresistance, x * 2 + 1, tmp)

if bool(ph) or bool(pl)
    supres = array.new_float(0) // number of pivot, strength, min/max levels
    stren = array.new_float(10, 0)
    // get levels and strengs
    for x = 0 to array.size(pivotvals) - 1 by 1
        [hi, lo, strength] = get_sr_vals(x)
        array.push(supres, strength)
        array.push(supres, hi)
        array.push(supres, lo)

    // add each HL to strengh
    for x = 0 to array.size(pivotvals) - 1 by 1
        h = array.get(supres, x * 3 + 1)
        l = array.get(supres, x * 3 + 2)
        s = 0
        for y = 0 to loopback by 1
            if high[y] <= h and high[y] >= l or low[y] <= h and low[y] >= l
                s := s + 1
                s
        array.set(supres, x * 3, array.get(supres, x * 3) + s)

    //reset SR levels
    array.fill(suportresistance, 0)
    // get strongest SRs
    src = 0
    for x = 0 to array.size(pivotvals) - 1 by 1
        stv = -1. // value
        stl = -1 // location
        for y = 0 to array.size(pivotvals) - 1 by 1
            if array.get(supres, y * 3) > stv and array.get(supres, y * 3) >= minstrength * 20
                stv := array.get(supres, y * 3)
                stl := y
                stl
        if stl >= 0
            //get sr level
            hh = array.get(supres, stl * 3 + 1)
            ll = array.get(supres, stl * 3 + 2)
            array.set(suportresistance, src * 2, hh)
            array.set(suportresistance, src * 2 + 1, ll)
            array.set(stren, src, array.get(supres, stl * 3))

            // make included pivot points' strength zero 
            for y = 0 to array.size(pivotvals) - 1 by 1
                if array.get(supres, y * 3 + 1) <= hh and array.get(supres, y * 3 + 1) >= ll or array.get(supres, y * 3 + 2) <= hh and array.get(supres, y * 3 + 2) >= ll
                    array.set(supres, y * 3, -1)

            src := src + 1
            if src >= 10
                break

    for x = 0 to 8 by 1
        for y = x + 1 to 9 by 1
            if array.get(stren, y) > array.get(stren, x)
                tmp = array.get(stren, y)
                array.set(stren, y, array.get(stren, x))
                changeit(x, y)


get_level(ind) =>
    float ret = na
    if ind < array.size(suportresistance)
        if array.get(suportresistance, ind) != 0
            ret := array.get(suportresistance, ind)
            ret
    ret

get_color(ind) =>
    color ret = na
    if ind < array.size(suportresistance)
        if array.get(suportresistance, ind) != 0
            ret := array.get(suportresistance, ind) > close and array.get(suportresistance, ind + 1) > close ? res_col : array.get(suportresistance, ind) < close and array.get(suportresistance, ind + 1) < close ? sup_col : inch_col
            ret
    ret

var srchannels = array.new_box(10)
for x = 0 to math.min(9, maxnumsr) by 1
    box.delete(array.get(srchannels, x))
    srcol = get_color(x * 2)
    if not na(srcol)
        array.set(srchannels, x, box.new(left = bar_index, top = get_level(x * 2), right = bar_index + 1, bottom = get_level(x * 2 + 1), border_color = srcol, border_width = 1, extend = extend.both, bgcolor = srcol))

resistancebroken = false
supportbroken = false

// check if it's not in a channel
not_in_a_channel = true
for x = 0 to math.min(9, maxnumsr) by 1
    if close <= array.get(suportresistance, x * 2) and close >= array.get(suportresistance, x * 2 + 1)
        not_in_a_channel := false
        not_in_a_channel

// if price is not in a channel then check broken ones
if not_in_a_channel
    for x = 0 to math.min(9, maxnumsr) by 1
        if close[1] <= array.get(suportresistance, x * 2) and close > array.get(suportresistance, x * 2)
            resistancebroken := true
            resistancebroken
        if close[1] >= array.get(suportresistance, x * 2 + 1) and close < array.get(suportresistance, x * 2 + 1)
            supportbroken := true
            supportbroken

alertcondition(resistancebroken, title = 'Resistance Broken', message = 'Resistance Broken')
alertcondition(supportbroken, title = 'Support Broken', message = 'Support Broken')
plotshape(showsrbroken and resistancebroken, style = shape.triangleup, location = location.belowbar, color = color.new(color.lime, 0), size = size.tiny)
plotshape(showsrbroken and supportbroken, style = shape.triangledown, location = location.abovebar, color = color.new(color.red, 0), size = size.tiny)


// Typical Price = (High + Low + Close)/3
atr_length = input.int(14, "ATR Length")  // ATR period

// User input
lookback_vwap = input.int(150, "VWAP Lookback Days")

// Typical Price for VWAP
tp = (high + low + close) / 3

// Cumulative sums for rolling VWAP
tpv = ta.cum(tp * volume)
vol_cum = ta.cum(volume)

// Rolling VWAP
vwap_rolling = (tpv - ta.cum(tp * volume)[lookback_vwap]) / (vol_cum - ta.cum(volume)[lookback_vwap])
label_size_input = input.string("Normal", "Label Size", options=["Tiny", "Small", "Normal", "Large", "Huge"])
x_offset_input = input.int(2, "Label X Offset", minval=0) // Bars to the right
// Convert string input to Pine Script size type
label_size = switch label_size_input
    "Tiny"   => size.tiny
    "Small"  => size.small
    "Normal" => size.normal
    "Large"  => size.large
    "Huge"   => size.huge
// Plot VWAP
plot(vwap_rolling, color=color.white, title="VWAP Rolling")

// --- Simple Moving Averages ---
sma20  = ta.sma(close, 20)
sma50  = ta.sma(close, 50)
sma100 = ta.sma(close, 100)
sma150 = ta.sma(close, 150)
sma200 = ta.sma(close, 200)

// Plot SMAs
plot(sma20,  color=color.rgb(59, 143, 211),   title="SMA20")
plot(sma50,  color=color.rgb(69, 224, 74),  title="SMA50")
plot(sma100, color=color.orange, title="SMA100")
plot(sma150, color=color.purple, title="SMA150")
plot(sma200, color=color.rgb(238, 45, 77),    title="SMA200")

// --- ATR Indicator ---
atrValue = ta.atr(atr_length)
atr=(atrValue / close) * 100

atr_color=color.rgb(36, 224, 42)
if atr>3
    atr_color:=color.rgb(233, 200, 14)
if atr>6
    atr_color:=color.rgb(233, 14, 14)
// Persistent label
var label atr_label = na

if barstate.islast
    // Delete previous label
    if not na(atr_label)
        label.delete(atr_label)
    
    // Offsets to float label to the right
    
    y_offset = 0  // adjust vertically if needed
    
    // Create new label
    atr_label := label.new(x = bar_index + x_offset_input,y = close + y_offset,xloc = xloc.bar_index,yloc = yloc.price,text = "ATR: " + str.tostring(atrValue, format.mintick) + " (" + str.tostring(atr, format.mintick) + "%)",color = atr_color,textcolor = color.white,style = label.style_label_left,size = label_size)

'''

