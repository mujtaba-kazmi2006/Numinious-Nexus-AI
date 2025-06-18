🧠 Numinous Nexus AI (Nunno)
"Nunno" is the friendly short form for Numinous Nexus AI – your personal finance and trading learning companion. It's designed to demystify the complex world of investing and trading for beginners, providing insights, predictions, and explanations in simple, easy-to-understand language.

Nunno serves as a central hub (nexus) of profound financial knowledge (numinous), connecting users to essential market data, analytical tools, and expert guidance.

✨ Features
Personalized Interactions: Nunno greets you by your name and tailors responses based on your age (if provided), making the learning experience more engaging.

Intelligent Conversational AI: Built on a powerful Large Language Model, Nunno can answer a wide range of financial questions naturally.

Conversation Memory: Nunno remembers previous questions and answers in a session, allowing for more coherent and context-aware follow-up discussions.

Crypto Tokenomics Analysis: Get detailed breakdowns of cryptocurrency metrics like Market Cap, Circulating Supply, FDV, Historical Returns (CAGR), and Volatility.

Crypto Price Prediction: Ask for "next move" predictions for major cryptocurrencies (BTC, ETH, SOL, etc.) across different timeframes.

Real-time Market News: Fetches and summarizes the latest financial headlines from reputable sources, explaining their potential market impact.

Monte Carlo Simulation: Test and simulate trading strategies with customizable win rates, risk-reward ratios, and market conditions.

Beginner-Friendly Explanations: All responses are crafted to be easy for novices to grasp, simplifying complex financial jargon.

Intuitive Tkinter GUI: A clean and responsive graphical user interface for easy interaction.

"Stop Response" Feature: Allows you to halt an ongoing AI response at any time.

"New Chat" Option: Easily clear the conversation history and start a fresh discussion.

🚀 Getting Started
Follow these steps to set up and run Numinous Nexus AI on your local machine.

Prerequisites
Before you begin, ensure you have:

Python 3.8+ installed on your system.

pip (Python package installer) up to date.

Installation
Clone the Repository:

git clone https://github.com/mujtaba.kazmi2006/numinous-nexus-ai.git
cd numinous-nexus-ai

(Note: Replace your-username with your actual GitHub username)

Install Python Dependencies:

pip install requests numpy fuzzywuzzy python-Levenshtein pyttsx3

Required Modules:
This project relies on two custom modules: predictor_module.py and montecarlo_module.py. Ensure these files are present in the same directory as your main application script. If you don't have them, you'll need to create placeholder files or implement their logic. (These modules would typically contain the specialized prediction and simulation algorithms.)

Example predictor_module.py (placeholder structure):

# predictor_module.py
import pandas as pd
import numpy as np

def fetch_binance_ohlcv(symbol="BTCUSDT", interval="15m"):
    print(f"Fetching mock OHLCV data for {symbol} {interval}...")
    # Replace with actual API call to Binance or other exchange
    data = {
        'timestamp': pd.to_datetime([pd.Timestamp.now() - pd.Timedelta(minutes=i) for i in range(100)]),
        'open': np.random.rand(100) * 1000 + 30000,
        'high': np.random.rand(100) * 1000 + 30500,
        'low': np.random.rand(100) * 1000 + 29500,
        'close': np.random.rand(100) * 1000 + 30000,
        'volume': np.random.rand(100) * 100000
    }
    return pd.DataFrame(data).set_index('timestamp')

def add_indicators(df):
    print("Adding mock indicators...")
    # Add simple mock indicators
    df['SMA'] = df['close'].rolling(window=10).mean()
    return df.dropna()

def prepare_dataset(df):
    print("Preparing dataset...")
    if df.empty:
        return (None, None, None, None), None
    # Mock dataset preparation
    X = df[['close', 'SMA']].values
    y = (df['close'].shift(-1) > df['close']).astype(int).values # Simple up/down
    latest_input = X[-1].reshape(1, -1)
    return (X[:-1], X[:-1], y[:-1], y[:-1]), latest_input # Mock train/test split

def train_and_predict(X_train, X_test, y_train, y_test, latest_input):
    print("Training and predicting (mock)...")
    if X_train is None or latest_input is None:
        return 0, 0.5, 0.5, "Insufficient data for prediction."

    # Mock prediction logic
    prediction = np.random.choice([0, 1]) # Randomly predict up (1) or down (0)
    confidence = np.random.uniform(0.6, 0.9) # Random confidence
    accuracy = np.random.uniform(0.55, 0.85) # Random accuracy
    reasoning = "Based on mock analysis of historical patterns and simulated indicator trends, a probabilistic directional shift is observed."
    return prediction, confidence, accuracy, reasoning

Example montecarlo_module.py (placeholder structure):

# montecarlo_module.py
import numpy as np

def simulate_trades(win_rate, rr_ratio, num_trades, market_condition):
    print(f"Running Monte Carlo simulation for Win Rate: {win_rate*100}%, RR: {rr_ratio}, Trades: {num_trades}, Market: {market_condition} (mock)...")
    results = []
    for _ in range(num_trades):
        if np.random.rand() < win_rate:
            # Win: Profit = Risk * RR Ratio (assume risk per trade is 1 unit)
            results.append(rr_ratio)
        else:
            # Loss: Loss = Risk (assume risk per trade is 1 unit)
            results.append(-1.0)
    return results

def monte_carlo_summary(results):
    if not results:
        return "No simulation results available."

    total_profit = sum(results)
    num_wins = sum(1 for r in results if r > 0)
    num_losses = len(results) - num_wins
    win_rate = num_wins / len(results) if len(results) > 0 else 0

    summary = (
        f"Simulation Results (over {len(results)} trades):\n"
        f"  Total Profit/Loss: {total_profit:,.2f} units\n"
        f"  Win Rate: {win_rate*100:.2f}%\n"
        f"  Number of Wins: {num_wins}\n"
        f"  Number of Losses: {num_losses}\n"
        "  This simulation provides a hypothetical outcome based on the given parameters. Past performance is not indicative of future results."
    )
    return summary

API Keys Setup
Numinous Nexus AI uses external APIs for its functionality. You will need to obtain your own API keys and update the AI_API_KEY and NEWS_API_KEY variables in the main.py file.

OpenRouter AI (for LLM):

Go to OpenRouter.ai.

Sign up or log in.

Navigate to your "API Keys" section.

Create a new API key.

Copy the key and paste it into the AI_API_KEY variable in main.py.

NewsAPI (for Market News):

Go to NewsAPI.org.

Sign up for a free developer account.

You'll find your API key on your dashboard.

Copy the key and paste it into the NEWS_API_KEY variable in main.py.

# main.py
# ...
AI_API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"
NEWS_API_KEY = "YOUR_NEWSAPI_KEY_HERE"
# ...

Security Note: For production applications, it's best practice to load API keys from environment variables or a secure configuration system, rather than hardcoding them directly in the script.

Running the Application
Once you have installed the dependencies and set up your API keys, you can run the application:

python main.py

💡 Usage
Welcome Screen: Upon launching, you'll be greeted by a welcome screen asking for your name and age. Enter your details to personalize your experience.

Start Chat: Click "Start Chat with Nunno" to proceed to the main chat interface.

Ask Questions: Type your finance or trading-related questions into the input field at the bottom and press Enter or click "Ask Nunno."

Quick Start Examples: Use the provided example buttons for quick queries.

Stop Response: If Nunno is taking too long or you want to interrupt, click "Stop Response."

New Chat: Click "New Chat" to clear the conversation history and start a fresh session with Nunno.

🛠️ Customization and Development
Modify AI Behavior: Adjust the SYSTEM_PROMPT_FORMAT string in main.py to change Nunno's persona, rules, or core instructions.

Extend Functionality:

New API Integrations: Add more data sources (e.g., stock data, economic indicators) by creating new functions similar to fetch_token_data or fetch_market_news.

Advanced Analytics: Implement more sophisticated financial models or machine learning algorithms within predictor_module.py or montecarlo_module.py.

UI Enhancements: Customize the Tkinter GUI's appearance, add new widgets, or improve responsiveness.

Conversation History Depth: Adjust MAX_HISTORY_MESSAGES in main.py to control how much past conversation Nunno remembers.

Voice Output: The pyttsx3 library is initialized for text-to-speech, though not actively used for output in the current chat display. You can integrate tts_engine.say(response) where response is Nunno's text to enable voice output.

🛣️ Future Enhancements
Persistent Chat History: Implement saving/loading chat history to a file or database so conversations can be resumed across sessions.

User Authentication: For multi-user scenarios or storing sensitive data.

Advanced Plotting: Integrate libraries like matplotlib or plotly to visualize financial data and predictions directly within the GUI.

Strategy Backtesting: Expand the Monte Carlo module to allow loading real historical data for backtesting.

News Filtering/Preferences: Allow users to specify preferred news topics or sources.

More LLM Models: Experiment with different models available on OpenRouter to find the best fit for specific tasks.

🤝 Contributing
Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, please:

Fork the repository.

Create a new branch (git checkout -b feature/your-feature-name).

Make your changes.

Commit your changes (git commit -m 'Add new feature').

Push to the branch (git push origin feature/your-feature-name).

Open a Pull Request.

📄 License
This project is open-source and available under the MIT License.

👨‍💻 Creator
Numinous Nexus AI (Nunno) was built by Mujtaba Kazmi, a young developer passionate about teaching people how to grow from nothing using finance, coding, and smart tools.
