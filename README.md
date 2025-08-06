# 🧠 Numinous Nexus AI – Nunno

“**Nunno**” (short for **Numinous Nexus AI**) is your personal trading and finance learning assistant — designed to simplify markets, charts, and crypto for beginners using intelligent, human-style conversation and predictions.

This repo includes **two complete versions** of Nunno’s prediction engine:

---

## 🔀 Versions

### 🧠 V1 – Basic Prediction Engine (`predictor_module.py`)
- Simple up/down directional forecasts
- Probability & confidence scores
- Good for quick, straightforward predictions
- Includes mock indicator logic and placeholders
- Best for testing or beginner exploration

### 🧠 V2 – Confluence-Based Trend Analysis (`betterpridictormodule.py`)
- Uses real indicators: RSI, MACD, Stoch RSI, EMA 14/200, ADX, ATR, BB Width
- No binary UP/DOWN guesses — just **real trend commentary**
- Simulates how a smart trader would explain the chart
- Much more reliable for education and deeper insight

---

## ✨ Key Features

- 📚 **Beginner-Friendly Explanations**  
  Explains everything in human terms, no jargon

- 🧠 **LLM-Powered Reasoning (via OpenRouter)**  
  Natural answers, not robotic scripts

- 📰 **Live Market News Summaries**  
  Breaks down news and explains their potential impact

- 📊 **Tokenomics Reports**  
  Full breakdown: market cap, FDV, volatility, CAGR, etc.

- 🎲 **Monte Carlo Simulator**  
  Simulates trade outcomes with win rates and R:R ratios

- 🖼 **(GUI Version)** Upload chart images and get visual candle reasoning

- 🖥 **Tkinter Chat GUI**  
  Stop responses anytime, start new chats, use example buttons

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/your-username/numinous-nexus-ai.git
cd numinous-nexus-ai
pip install -r requirements.txt
You can also run V2 standalone using only betterpridictormodule.py.

🧪 Sample Usage for V2
python
Copy
Edit
from betterpridictormodule import fetch_binance_ohlcv, add_indicators, generate_reasoning

df = fetch_binance_ohlcv("BTCUSDT", "15m")
df = add_indicators(df)
latest = df.iloc[-1]

for line in generate_reasoning(latest):
    print("•", line)
🔐 API Keys Setup
This app uses:

🧠 OpenRouter API for LLM responses

📰 NewsAPI.org for financial headlines


🛣️ Roadmap
 Build GUI with interactive chat

 Add two predictor versions

 Market news summarizer

 Token data analyzer

 Monte Carlo risk simulation

 Web version (in progress)

 Strategy backtester (planned)

👨‍💻 Created By
Mujtaba Kazmi
A self-taught dev on a mission to help people grow through finance, coding, and real tools that work.

📄 License
This project is open-source under the MIT License.

yaml
Copy
Edit

---

Let me know if you want:
- This as a `.md` file
- A nice cover image/banner
- Or a `requirements.txt` file with all dependencies ready
