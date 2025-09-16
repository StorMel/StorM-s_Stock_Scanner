import yfinance as yf
import json
import os


def create_filtered_stock_list(input_filename: str, output_filename: str, min_market_cap: int = 1_000_000_000) -> None:
    """
    Reads a JSON file with a specific format, filters by market cap, and saves the list.
    """
    if not os.path.exists(input_filename):
        print(f"Error: The file '{input_filename}' was not found.")
        return

    try:
        with open(input_filename, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file: {e}")
        return

    # Your JSON is a list containing a single dictionary with numbered string keys.
    # We must access the first element (the dictionary) and then its values.
    # The `data[0]` part is the key fix here.
    tickers_to_check = [item['ticker'] for item in data[0].values() if 'ticker' in item]

    filtered_tickers = []
    print(f"Filtering {len(tickers_to_check)} stocks with market cap > ${min_market_cap / 1e9:.0f}B...")

    for symbol in tickers_to_check:
        try:
            stock = yf.Ticker(symbol)
            info = stock.get_info()
            market_cap = info.get('marketCap', 0)

            if market_cap and market_cap > min_market_cap:
                filtered_tickers.append(symbol)
                print(f"  ✅ Added {symbol} (Market Cap: ${market_cap / 1e9:.2f}B)")
            else:
                print(f"  ❌ Skipping {symbol} (Market Cap: ${market_cap / 1e9:.2f}B)")

        except Exception as e:
            print(f"  ❌ Skipping {symbol} due to an error: {e}")
            continue

    with open(output_filename, "w") as f:
        json.dump(filtered_tickers, f, indent=4)

    print(f"\nFiltered list saved to '{output_filename}'")
    print(f"Total stocks in the filtered list: {len(filtered_tickers)}")


if __name__ == "__main__":
    input_file = "flat-ui__data-Wed Aug 20 2025.json"
    output_file = "filtered_large_cap_stocks.json"
    create_filtered_stock_list(input_file, output_file)