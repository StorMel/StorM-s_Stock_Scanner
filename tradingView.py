import yfinance as yf
import pandas as pd
import json

# Load tickers from your JSON file
#with open("flat-ui__data-Wed Aug 20 2025.json", "r") as f:
#    data = json.load(f)

with open("filtered_large_cap_stocks.json", "r") as f:
    data = json.load(f)

THRESHOLD_PCT = 3

TICKERS = [item["Symbol"] for item in data]#["NVDA","AAPL","AMD","ACGL","FRME","FOLD","RIGL"]#

results = []

for symbol in TICKERS:
    print(symbol)
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")  # 1 year of daily data
        info=stock.info
        market_cap = info.get('marketCap', 0)

        # Skip stocks with market cap under $1B
        if not market_cap or market_cap < 1e9:
            print(f"Skipping {symbol}: Market Cap missing or under $1B")
            continue
        if hist.empty or len(hist)<200:
            continue

        # Current price = last close
        current = hist["Close"].iloc[-1]

        # SMA200
        sma200 = hist["Close"].rolling(window=200).mean().iloc[-1]
        # Skip if SMA200 is empty

        # SMA150
        sma150 = hist["Close"].rolling(window=150).mean().iloc[-1]


        # Volume (latest)
        #vol = hist["Volume"].iloc[-1]

        # ✅ VWAP 150 days rolling indicator
        hist["TPV"] = (hist["High"] + hist["Low"] + hist["Close"])* hist["Volume"] / 3   # Typical price * volume
        vwap_150 = hist["TPV"].tail(150).sum() / hist["Volume"].tail(150).sum()


        # check if in bullish streak and check for rising volume
        lookback = 2  # 👈 choose how many days back
        bullish_streak = (hist["Close"].tail(lookback) > hist["Open"].tail(lookback)).all()
        volumes = hist["Volume"].tail(lookback)
        rising_volume = (volumes.diff().dropna() > 0).all()


        if pd.isna(sma150):
            continue

        pct150_diff = (current - sma150) / current * 100 # from the price to the moving average
        pct200_diff = (current - sma200) / current * 100
        pctVwap_diff=-(current - vwap_150) / current * 100
        if abs(pct150_diff) <= THRESHOLD_PCT:
            link_formula = f'=HYPERLINK("https://www.tradingview.com/chart/?symbol=NASDAQ:{symbol}", "{symbol}")'
            #link_formula = '=HYPERLINK("https://www.tradingview.com/chart/?symbol=NASDAQ:" & A2, A2)'
            results.append((link_formula, current,vwap_150, abs(pctVwap_diff), sma150, abs(pct150_diff), sma200,abs(pct200_diff),bullish_streak,rising_volume))
    except Exception as e:
        print(f"Skipping {symbol}: {e}")

df = pd.DataFrame(results, columns=["Ticker", "Close","VWAP150","% Diff VWAP","SMA150" , "% Diff 150", "SMA200", "% Diff 200","Bull Streak","Rising Vol"])
print(df)

# Optionally save to CSV
df.to_csv("scan_results.csv", index=False)
# =$H1>0 excel condotional formatting for above 0 values


#pine script for tradingview:
"""
// This Pine Script® code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © StorMonster

//@version=6
indicator("Custom Rolling VWAP + SMAs", overlay=true, max_lines_count=500)


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

"""
