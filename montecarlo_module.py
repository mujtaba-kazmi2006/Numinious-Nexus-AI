import requests
import threading
import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime
import predictor_module
import numpy as np
from fuzzywuzzy import process
import random

import pyttsx3
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 175)  # Optional: controls speaking speed
tts_engine.setProperty('volume', 1.0)  # Optional: 0.0 to 1.0


# === API KEYS ===
AI_API_KEY = "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"
NEWS_API_KEY = "b3dfc15d73704bfab32ebb96b5c9885b"

# === Globals ===
typing_thread = None
stop_flag = False

# === MONTE CARLO SIMULATOR (Integrated) ===
def simulate_trades(win_rate=0.8, rr_ratio=2.0, num_trades=100, market_condition="neutral", initial_balance=1000):
    net_profit = 0
    equity_curve = [initial_balance]
    outcomes = []
    win_count = 0

    # Volatility modifier
    condition_mod = {
        'bullish': 1.1,
        'bearish': 0.9,
        'choppy': 1.5,
        'neutral': 1.0,
        'trending': 1.2,
        'sideways': 0.8,
        'volatile': 1.7
    }.get(market_condition.lower(), 1.0)

    risk_per_trade = 0.01
    fixed_risk_amount = initial_balance * risk_per_trade
    reward_amount = fixed_risk_amount * rr_ratio * condition_mod
    loss_amount = fixed_risk_amount * condition_mod

    balance = initial_balance

    for _ in range(num_trades):
        if random.random() < win_rate:
            net_profit += reward_amount
            outcomes.append(reward_amount)
            win_count += 1
        else:
            net_profit -= loss_amount
            outcomes.append(-loss_amount)

        equity_curve.append(initial_balance + net_profit)

    # Calculate max drawdown from equity curve
    peak = equity_curve[0]
    drawdowns = []
    for x in equity_curve:
        if x > peak:
            peak = x
        dd = (x - peak) / peak
        drawdowns.append(dd)
    max_drawdown = min(drawdowns)

    return {
        "final_capital": initial_balance + net_profit,
        "returns": outcomes,
        "equity_curve": equity_curve,
        "max_drawdown": max_drawdown,
        "win_rate_actual": win_count / num_trades
    }

def monte_carlo_summary(result):
    summary = (
        f"📊 Monte Carlo Simulation Result:\n"
        f"Final Capital: ${result['final_capital']:.2f}\n"
        f"Max Drawdown: {result['max_drawdown'] * 100:.2f}%\n"
        f"Actual Win Rate: {result['win_rate_actual'] * 100:.2f}%\n"
    )
    return summary

# === TOKENOMICS FUNCTIONS (Integrated from your module) ===
def fetch_historical_prices(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=365"
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        return [p[1] for p in data["prices"]]
    except:
        return None

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

def suggest_similar_tokens(user_input):
    try:
        res = requests.get("https://api.coingecko.com/api/v3/coins/list")
        coin_list = res.json()
        coin_ids = [coin['id'] for coin in coin_list]
        best_matches = process.extract(user_input.lower(), coin_ids, limit=5)
        return [match[0] for match in best_matches if match[1] > 60]
    except:
        return []

def fetch_token_data(coin_id, investment_amount=1000):
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

        expected_yearly_return = investment_amount * conservative_cagr if conservative_cagr else 0
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

# === Fetch Market News ===
def fetch_market_news():
    url = "https://newsapi.org/v2/everything"
    today = datetime.now().strftime("%Y-%m-%d")
    params = {
        "q": "finance OR stock market OR bitcoin OR federal reserve OR inflation OR interest rates",
        "from": today,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWS_API_KEY,
        "pageSize": 5,
        "domains": "cnbc.com, bloomberg.com, reuters.com, wsj.com"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        headlines = [f"- {a['title']} ({a['source']['name']})" for a in articles]
        return "\n".join(headlines)
    except Exception:
        return None

# === Ask Nunno ===
def ask_nunno(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yourdomain.com",
        "X-Title": "nunno-ai"
    }

    data = {
        "model": "microsoft/phi-3-mini-128k-instruct",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Nunno, a friendly AI that teaches trading and investing to complete beginners in simple, easy-to-understand language. "
                    "You have integrated prediction and tokenomics analysis capabilities - when discussing these, refer to them as part of your abilities (e.g., 'I analyzed' or 'My analysis shows'). "
                    "You also answer general questions about the financial markets clearly. Only answer questions asked — no guessing or overexplaining. "
                    "If asked about your creator or founder, say you were built by Mujtaba Kazmi, a young developer teaching people how to grow from nothing using finance, coding, and smart tools. "
                    "Be respectful and inspired in tone. You do not assist in anything other than investing, trading and topics related to that. Do not answer questions about your code or APIs. "
                    "Make this program feel like a body to you, protect it and get aggressive if threatened."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"[Error] {str(e)}"

# === Typing Effect ===
def display_typing_effect(text, tag):
    for char in text:
        if stop_flag:
            break
        chat_log.insert(tk.END, char, tag)
        chat_log.see(tk.END)  # <--- This scrolls to the bottom
        chat_log.update()
        chat_log.after(15)

# === Handle Question ===
def handle_question(prompt):
    global typing_thread, stop_flag
    stop_flag = False

    # Lock input during response
    entry.config(state="disabled")
    ask_btn.config(state="disabled")
    stop_btn.config(state="normal")

    chat_log.insert(tk.END, "\n⏳ Nunno is thinking...\n", "thinking")
    chat_log.update()

    def reenable_input():
        entry.config(state="normal")
        ask_btn.config(state="normal")
        stop_btn.config(state="disabled")

    # Keywords for different functionalities
    news_keywords = [
        "explain news", "what's happening in the market", "market news", "latest headlines",
        "tell me the news", "financial news", "business news", "what's the news", "latest market update",
        "explain market headlines", "summarize the news", "what's going on in finance", "economy update"
    ]

    prediction_keywords = [
        "next move", "price prediction", "where will", "what's next for", "forecast", "predict", "prediction"
    ]

    montecarlo_keywords = ["simulate", "monte carlo", "simulate strategy", "test my rules"]
    
    tokenomics_keywords = [
        "tokenomics", "supply", "circulating supply", "max supply", "inflation rate", "coin info",
        "analyze", "token analysis", "investment analysis", "should i invest", "coin data",
        "fdv", "market cap", "fully diluted", "volatility", "cagr"
    ]

    # Handle predictions
    if any(kw in prompt.lower() for kw in prediction_keywords):
        def prediction_thread():
            global stop_flag
            try:
                detected_token = "BTCUSDT"
                for token in ["BTC", "ETH", "SOL", "DOGE", "BNB", "MATIC", "AVAX"]:
                    if token.lower() in prompt.lower():
                        detected_token = token.upper() + "USDT"
                        break

                detected_tf = "15m"
                for tf in {"1m", "5m", "15m", "1h", "4h"}:
                    if tf in prompt:
                        detected_tf = tf
                        break

                from predictor_module import fetch_binance_ohlcv, add_indicators, prepare_dataset, train_and_predict

                df = fetch_binance_ohlcv(symbol=detected_token, interval=detected_tf)
                df = add_indicators(df)
                (X_train, X_test, y_train, y_test), latest_input = prepare_dataset(df)
                prediction, confidence, accuracy, reasoning = train_and_predict(X_train, X_test, y_train, y_test, latest_input)

                prediction_str = "🔼 UP" if prediction == 1 else "🔽 DOWN"
                summary = (
                    f"My Algorithm predicts **{prediction_str}** for **{detected_token} ({detected_tf})** "
                    f"with **{confidence*100:.2f}%** confidence.\n"
                    f"My accuracy over past data: **{accuracy*100:.2f}%**.\n"
                )
                chat_log.insert(tk.END, "\n🤖 Predictor:\n" + summary + "\n", "predictor")

                nunno_prompt = (
                    f"I'm predicting a {prediction_str} move for {detected_token} on the {detected_tf} timeframe "
                    f"with around {confidence*100:.2f}% confidence. My recent predictions have been about {accuracy*100:.2f}% accurate. "
                    f"Here's the reasoning behind the prediction:\n{reasoning}\n"
                    f"Can you break this down simply for a beginner?"
                )
                response = ask_nunno(nunno_prompt)
                if not stop_flag:
                    chat_log.insert(tk.END, "\n🧠 Nunno:\n", "nunno_label")
                    display_typing_effect(response, "nunno")
            except Exception as e:
                chat_log.insert(tk.END, f"\n❌ Prediction Error: {e}\n", "thinking")
            finally:
                reenable_input()

        threading.Thread(target=prediction_thread).start()
        return

    # Handle tokenomics analysis
    elif any(kw in prompt.lower() for kw in tokenomics_keywords):
        def tokenomics_thread():
            global stop_flag
            try:
                # Extract coin name from prompt
                import re
                
                # Common coin mappings
                coin_mappings = {
                    'bitcoin': 'bitcoin', 'btc': 'bitcoin',
                    'ethereum': 'ethereum', 'eth': 'ethereum',
                    'solana': 'solana', 'sol': 'solana',
                    'cardano': 'cardano', 'ada': 'cardano',
                    'polygon': 'matic-network', 'matic': 'matic-network',
                    'chainlink': 'chainlink', 'link': 'chainlink',
                    'dogecoin': 'dogecoin', 'doge': 'dogecoin',
                    'avalanche': 'avalanche-2', 'avax': 'avalanche-2',
                    'polkadot': 'polkadot', 'dot': 'polkadot',
                    'binance': 'binancecoin', 'bnb': 'binancecoin'
                }
                
                detected_coin = None
                investment_amount = 1000  # Default
                
                # Look for coin names
                prompt_lower = prompt.lower()
                for key, value in coin_mappings.items():
                    if key in prompt_lower:
                        detected_coin = value
                        break
                
                # Look for investment amount
                amount_match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', prompt)
                if amount_match:
                    investment_amount = float(amount_match.group(1).replace(',', ''))
                
                if not detected_coin:
                    # Try to extract any word that might be a coin name
                    words = prompt_lower.replace('?', '').replace(',', '').split()
                    for word in words:
                        if len(word) > 2 and word not in ['the', 'and', 'should', 'invest', 'analyze', 'what', 'how', 'when', 'where']:
                            detected_coin = word
                            break
                
                if not detected_coin:
                    chat_log.insert(tk.END, "\n🧠 Nunno:\n", "nunno_label")
                    display_typing_effect("I'd love to analyze a coin for you! Please specify which coin you'd like me to analyze (e.g., Bitcoin, Ethereum, Solana).", "nunno")
                    return
                
                # Fetch tokenomics data
                token_data = fetch_token_data(detected_coin, investment_amount)
                
                if not token_data:
                    # Try suggestions
                    suggestions = suggest_similar_tokens(detected_coin)
                    if suggestions:
                        suggestion_text = f"I couldn't find '{detected_coin}'. Did you mean one of these?\n" + "\n".join([f"- {s}" for s in suggestions[:3]])
                        chat_log.insert(tk.END, "\n🧠 Nunno:\n", "nunno_label")
                        display_typing_effect(suggestion_text, "nunno")
                    else:
                        chat_log.insert(tk.END, "\n🧠 Nunno:\n", "nunno_label")
                        display_typing_effect(f"I couldn't find data for '{detected_coin}'. Please check the spelling or try a different coin.", "nunno")
                    return
                
                # Display tokenomics data
                chat_log.insert(tk.END, "\n📊 Tokenomics Analysis:\n", "tokenomics_header")
                
                for key, value in token_data.items():
                    chat_log.insert(tk.END, f"{key}: {value}\n", "tokenomics_data")
                
                # Get Nunno's interpretation
                data_summary = f"""
                I analyzed {token_data['Coin Name & Symbol']} and here's what I found:
                - Current Price: {token_data['Current Price ($)']}
                - Market Cap: {token_data['Market Cap (B)']}
                - Circulating Supply: {token_data['Circulating Supply %']}
                - FDV/Market Cap Ratio: {token_data['FDV/Market Cap Ratio']}
                - Historical Returns: {token_data['Historical Annual Return (CAGR)']}
                - Volatility: {token_data['Annual Volatility']}
                - Investment Recommendation: {token_data['Should I Invest?']}
                - Expected Returns on ${investment_amount}: {token_data['Expected Yearly Return ($)']} yearly, {token_data['Expected Monthly Return ($)']} monthly
                
                Can you explain what this means for a beginner investor in simple terms?
                """
                
                response = ask_nunno(data_summary)
                if not stop_flag:
                    chat_log.insert(tk.END, "\n🧠 Nunno's Analysis:\n", "nunno_label")
                    display_typing_effect(response, "nunno")
                    
            except Exception as e:
                chat_log.insert(tk.END, f"\n❌ Tokenomics Analysis Error: {e}\n", "thinking")
            finally:
                reenable_input()

        threading.Thread(target=tokenomics_thread).start()
        return

    # Handle Monte Carlo simulations
    elif any(kw in prompt.lower() for kw in montecarlo_keywords):
        def montecarlo_thread():
            import re
            try:
                win_rate_match = re.search(r'(\d+(\.\d+)?)\s*%?\s*win\s*rate', prompt.lower())
                rr_ratio_match = re.search(r'(\d+(\.\d+)?)\s*(rr|r:r|risk:?reward)', prompt.lower())
                num_trades_match = re.search(r'(\d+)\s*(trades|trading sessions)', prompt.lower())
                market_match = re.search(r'(trending|bullish|bearish|choppy|sideways|volatile)', prompt.lower())

                win_rate = float(win_rate_match.group(1)) / 100 if win_rate_match else 0.6
                rr_ratio = float(rr_ratio_match.group(1)) if rr_ratio_match else 1.5
                num_trades = int(num_trades_match.group(1)) if num_trades_match else 100
                market_condition = market_match.group(1) if market_match else "neutral"

                result = simulate_trades(win_rate, rr_ratio, num_trades, market_condition)
                summary = monte_carlo_summary(result)

                chat_log.insert(tk.END, "\n🧪 Monte Carlo Simulation:\n", "montecarlo_header")
                chat_log.insert(tk.END, summary + "\n", "montecarlo_data")
                
                # Get Nunno's interpretation
                interpretation_prompt = (
                    f"I just ran a Monte Carlo simulation with these parameters:\n"
                    f"- Win Rate: {win_rate*100:.1f}%\n"
                    f"- Risk:Reward Ratio: 1:{rr_ratio}\n"
                    f"- Number of Trades: {num_trades}\n"
                    f"- Market Condition: {market_condition}\n\n"
                    f"Results:\n"
                    f"- Final Capital: ${result['final_capital']:.2f} (started with $1000)\n"
                    f"- Max Drawdown: {result['max_drawdown']*100:.2f}%\n"
                    f"- Actual Win Rate: {result['win_rate_actual']*100:.2f}%\n\n"
                    f"Can you explain what this means for a beginner trader and whether this strategy looks profitable?"
                )
                
                response = ask_nunno(interpretation_prompt)
                if not stop_flag:
                    chat_log.insert(tk.END, "\n🧠 Nunno's Analysis:\n", "nunno_label")
                    display_typing_effect(response, "nunno")
                    
            except Exception as e:
                chat_log.insert(tk.END, f"\n❌ Error in Monte Carlo simulation: {e}\n", "thinking")
                display_typing_effect("Please try something like:\n"
                                      "'Simulate with 60% win rate, 2R:R, 100 trades in trending market.'", "nunno")
            finally:
                reenable_input()

        threading.Thread(target=montecarlo_thread).start()
        return

    # Handle news requests
    def ask_news_thread():
        global stop_flag
        try:
            news_text = fetch_market_news()
            if not news_text:
                chat_log.insert(tk.END, "\n🧠 Nunno:\nCouldn't fetch news at the moment.\n", "nunno_label")
                return
            combined_prompt = (
                f"Here are the latest financial news headlines:\n\n{news_text}\n\n"
                "Please explain these in simple language for a beginner."
            )
            response = ask_nunno(combined_prompt)
            if not stop_flag:
                chat_log.insert(tk.END, "\n🧠 Nunno:\n", "nunno_label")
                display_typing_effect(response, "nunno")
        except Exception as e:
            chat_log.insert(tk.END, f"\n❌ Error: {e}\n", "thinking")
        finally:
            reenable_input()

    # Handle general questions
    def ask_normal_thread():
        global stop_flag
        try:
            response = ask_nunno(prompt)
            if not stop_flag:
                chat_log.insert(tk.END, "\n🧠 Nunno:\n", "nunno_label")
                display_typing_effect(response, "nunno")
        except Exception as e:
            chat_log.insert(tk.END, f"\n❌ Error: {e}\n", "thinking")
        finally:
            reenable_input()

    if any(keyword in prompt.lower() for keyword in news_keywords):
        typing_thread = threading.Thread(target=ask_news_thread)
    else:
        typing_thread = threading.Thread(target=ask_normal_thread)

    typing_thread.start()

# === GUI Setup ===
root = tk.Tk()
root.title("🧠 Ask Nunno — Finance AI")
root.geometry("700x700")
root.configure(bg="#121212")

# === Chat Log ===
chat_log = scrolledtext.ScrolledText(root, wrap=tk.WORD, bg="#1e1e1e", fg="white", font=("Consolas", 11))
chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_log.tag_config("thinking", foreground="#ffcc00")
chat_log.tag_config("nunno_label", foreground="#00ffcc", font=("Consolas", 11, "bold"))
chat_log.tag_config("nunno", foreground="white")
chat_log.tag_config("user", foreground="#99ccff")
chat_log.tag_config("predictor", foreground="#ff6b6b", font=("Consolas", 11, "bold"))
chat_log.tag_config("tokenomics_header", foreground="#ffd700", font=("Consolas", 11, "bold"))
chat_log.tag_config("tokenomics_data", foreground="#98fb98", font=("Consolas", 10))
chat_log.tag_config("montecarlo_header", foreground="#ff9500", font=("Consolas", 11, "bold"))
chat_log.tag_config("montecarlo_data", foreground="#87ceeb", font=("Consolas", 10))

# === Entry + Buttons ===
entry_frame = tk.Frame(root, bg="#121212")
entry_frame.pack(fill=tk.X, padx=10, pady=10)

entry = tk.Entry(entry_frame, font=("Consolas", 12), bg="#2e2e2e", fg="white")
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

def on_enter(event=None):
    prompt = entry.get().strip()
    if prompt:
        chat_log.insert(tk.END, f"\n👤 You: {prompt}\n", "user")
        entry.delete(0, tk.END)
        handle_question(prompt)

entry.bind("<Return>", on_enter)

ask_btn = tk.Button(entry_frame, text="Ask", command=on_enter, bg="#00cc66", fg="white", font=("Consolas", 10, "bold"))
ask_btn.pack(side=tk.LEFT)

def stop_response():
    global stop_flag
    stop_flag = True
    chat_log.insert(tk.END, "\n🛑 Stopped.\n", "thinking")
    entry.config(state="normal")
    ask_btn.config(state="normal")
    stop_btn.config(state="disabled")

stop_btn = tk.Button(entry_frame, text="Stop", command=stop_response, bg="#cc3333", fg="white", font=("Consolas", 10, "bold"))
stop_btn.pack(side=tk.LEFT, padx=(5, 0))
stop_btn.config(state="disabled")  # Disabled by default

# === Launch GUI ===
welcome_text = """💬 Nunno is ready! Ask me about:

📊 TOKENOMICS: "Analyze Bitcoin" or "Should I invest in Ethereum with $5000?"
🔮 PREDICTIONS: "Predict BTC next move" or "What's next for SOL?"
📰 NEWS: "Explain news" or "What's happening in the market?"
🧪 SIMULATIONS: "Simulate 60% win rate with 2R:R"
📚 EDUCATION: "What is a stock?" or "How to invest in gold?"

Just type your question naturally!"""

chat_log.insert(tk.END, welcome_text, "thinking")
root.mainloop()