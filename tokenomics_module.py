import requests
import numpy as np
from fuzzywuzzy import process

# Fetch historical prices
def fetch_historical_prices(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=365"
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        return [p[1] for p in data["prices"]]
    except:
        return None

# Calculate CAGR and volatility
def calculate_cagr_and_volatility(prices):
    try:
        returns = [np.log(prices[i+1] / prices[i]) for i in range(len(prices)-1)]
        avg_daily_return = np.mean(returns)
        daily_volatility = np.std(returns)
        trading_days = 365
        annual_return = np.exp(avg_daily_return * trading_days) - 1
        annual_volatility = daily_volatility * np.sqrt(trading_days)
        conservative_return = annual_return * 0.5
        return annual_return, annual_volatility, conservative_return
    except:
        return None, None, None

# Suggest similar tokens
def suggest_similar_tokens(user_input):
    try:
        res = requests.get("https://api.coingecko.com/api/v3/coins/list")
        coin_list = res.json()
        coin_ids = [coin['id'] for coin in coin_list]
        best_matches = process.extract(user_input.lower(), coin_ids, limit=5)
        return [match[0] for match in best_matches if match[1] > 60]
    except:
        return []

# Main function to fetch token data
def fetch_token_data(coin_id, investment_amount):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id.lower().strip()}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        market = data["market_data"]

        circulating = market.get("circulating_supply", 0)
        total = market.get("total_supply", 0)
        price = market.get("current_price", {}).get("usd", 0)
        mcap = market.get("market_cap", {}).get("usd", 0)
        fdv = total * price if total else 0
        circ_percent = (circulating / total) * 100 if total else None
        fdv_mcap_ratio = (fdv / mcap) if mcap else None

        healthy = "✅ This coin seems healthy!" if circ_percent and circ_percent > 50 and fdv_mcap_ratio and fdv_mcap_ratio < 2 else "⚠️ Warning: This coin might be risky or inflated."

        prices = fetch_historical_prices(coin_id)
        if not prices:
            return None
        cagr, volatility, conservative_cagr = calculate_cagr_and_volatility(prices)

        expected_yearly_return = investment_amount * conservative_cagr
        expected_monthly_return = expected_yearly_return / 12

        return {
            "Coin Name & Symbol": f"{data['name']} ({data['symbol'].upper()})",
            "Current Price ($)": f"${price:,.6f}",
            "Market Cap (B)": f"${mcap / 1e9:,.2f}B — The value of all coins in the market",
            "Total Supply (M)": f"{total / 1e6:,.2f}M — Maximum possible number of coins",
            "Circulating Supply (M)": f"{circulating / 1e6:,.2f}M — Coins that are currently in circulation",
            "Circulating Supply %": f"{circ_percent:,.2f}%" if circ_percent else "N/A",
            "FDV (B)": f"${fdv / 1e9:,.2f}B — What the coin could be worth if all coins were unlocked",
            "FDV/Market Cap Ratio": f"{fdv_mcap_ratio:,.2f} — The lower this ratio, the better" if fdv_mcap_ratio else "N/A",
            "Historical Annual Return (CAGR)": f"{cagr * 100:,.2f}% — How much the coin has grown in a year" if cagr else "N/A",
            "Annual Volatility": f"{volatility * 100:,.2f}% — Price fluctuation level" if volatility else "N/A",
            "Realistic Yearly Return (50% CAGR)": f"{conservative_cagr * 100:,.2f}%" if conservative_cagr else "N/A",
            "Expected Monthly Return ($)": f"${expected_monthly_return:,.2f}" if expected_monthly_return else "N/A",
            "Expected Yearly Return ($)": f"${expected_yearly_return:,.2f}" if expected_yearly_return else "N/A",
            "Should I Invest?": healthy
        }
    except requests.exceptions.RequestException:
        return None

# Optional CLI usage
def main(coin_id=None, investment=None):
    if coin_id is None:
        coin_id = input("Enter CoinGecko Coin ID (e.g. 'solana', 'ethereum', 'bitcoin'): ").strip()
    if investment is None:
        try:
            investment = float(input("Enter your investment amount in USD: "))
            if investment <= 0:
                raise ValueError("Investment must be greater than 0.")
        except ValueError as e:
            print(f"[Error] Invalid input: {e}")
            return

    data = fetch_token_data(coin_id, investment)

    if data:
        print("\n📊 Tokenomics & Forecast:")
        for key, value in data.items():
            print(f"{key}: {value}")
    else:
        suggestions = suggest_similar_tokens(coin_id)
        if suggestions:
            print("\nCoin not found. Did you mean:")
            for suggestion in suggestions:
                print(f"- {suggestion}")
        else:
            print("\nCoin not found, and no similar coins were detected. Please check the spelling.")

# Only runs when executed directly
if __name__ == "__main__":
    main()
