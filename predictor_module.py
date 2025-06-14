import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# STEP 1: Fetch OHLCV data from Binance
def fetch_binance_ohlcv(symbol="BTCUSDT", interval="15m", limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.text}")
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base", "Taker Buy Quote", "Ignore"
    ])
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    return df

# STEP 2: Add indicators
def add_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']

    df['RSI'] = RSIIndicator(close, window=14).rsi()
    df['EMA'] = EMAIndicator(close, window=14).ema_indicator()
    df['MACD'] = MACD(close).macd_diff()
    df['Stoch_RSI'] = StochasticOscillator(close=close, high=high, low=low).stoch()
    bb = BollingerBands(close)
    df['BB_width'] = bb.bollinger_hband() - bb.bollinger_lband()
    df['ATR'] = AverageTrueRange(high=high, low=low, close=close).average_true_range()
    df['ADX'] = ADXIndicator(high=high, low=low, close=close).adx()
    df['Body'] = abs(df['Close'] - df['Open'])

    df.dropna(inplace=True)
    return df

# STEP 3: Prepare dataset
def prepare_dataset(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    features = [
        'Open', 'High', 'Low', 'Close',
        'RSI', 'EMA', 'MACD',
        'Stoch_RSI', 'BB_width', 'ATR', 'ADX',
        'Body'
    ]

    X = df[features]
    y = df['Target']

    return train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42), X.iloc[[-1]]

# STEP 4: Train model and predict
def train_and_predict(X_train, X_test, y_train, y_test, latest_input):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    prediction = model.predict(latest_input)[0]
    confidence = max(model.predict_proba(latest_input)[0])
    accuracy = model.score(X_test, y_test)

    # Generate reasoning based on feature importance
    importances = model.feature_importances_
    top_features = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)[:3]
    reasoning = "This decision was based mostly on:\n"
    for name, weight in top_features:
        reasoning += f"- {name}: importance {weight:.2f}\n"

    return prediction, confidence, accuracy, reasoning


# STEP 5: User selects token
def user_input_token():
    options = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT"]
    print("\nSelect a token to analyze:")
    for i, token in enumerate(options, start=1):
        print(f"{i}. {token}")
    print(f"{len(options)+1}. Enter custom token (e.g., MATICUSDT)")
    
    choice = input("Your choice: ")
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        return options[int(choice)-1]
    elif choice == str(len(options)+1):
        custom = input("Enter custom token (e.g., AVAXUSDT): ").upper()
        return custom
    else:
        print("Invalid choice. Defaulting to BTCUSDT.")
        return "BTCUSDT"

# STEP 6: User selects timeframe
def user_input_timeframe():
    tf_options = {"1": "1m", "2": "5m", "3": "15m", "4": "1h", "5": "4h"}
    print("\nSelect a timeframe:")
    for key, value in tf_options.items():
        print(f"{key}. {value}")
    choice = input("Your choice: ")

    return tf_options.get(choice, "15m")
def generate_reasoning(row):
    reasons = []

    if row['RSI'] > 70:
        reasons.append("RSI is overbought → possible reversal or strength continuation")
    elif row['RSI'] < 30:
        reasons.append("RSI is oversold → potential bounce or rally")

    if row['MACD'] > 0:
        reasons.append("MACD is bullish → trend momentum is up")
    elif row['MACD'] < 0:
        reasons.append("MACD is bearish → downward momentum")

    if row['Stoch_RSI'] > 80:
        reasons.append("Stoch RSI is overbought → possible short-term dip")
    elif row['Stoch_RSI'] < 20:
        reasons.append("Stoch RSI is oversold → short-term bounce possible")

    if row['Close'] > row['EMA']:
        reasons.append("Price is above EMA → uptrend bias")
    else:
        reasons.append("Price is below EMA → downtrend bias")

    if row['ADX'] > 25:
        reasons.append("ADX is strong → market is trending")
    else:
        reasons.append("ADX is weak → market might be ranging")

    if row['BB_width'] < row['ATR']:
        reasons.append("Bollinger Band width is low relative to ATR → potential breakout setup")

    if row['Body'] > row['ATR']:
        reasons.append("Large candle body → strong momentum candle")
    
    if not reasons:
        reasons.append("Mixed signals → market indecision")

    return reasons


    return prediction, confidence, accuracy, reasoning

# MAIN PROGRAM
if __name__ == "__main__":
    try:
        token = user_input_token()
        timeframe = user_input_timeframe()
        print(f"\nFetching data for {token} ({timeframe})...")
        
        df = fetch_binance_ohlcv(symbol=token, interval=timeframe)
        df = add_indicators(df)
        (X_train, X_test, y_train, y_test), latest_input = prepare_dataset(df)
        prediction, confidence, accuracy, reasoning = train_and_predict(X_train, X_test, y_train, y_test, latest_input)

        print(f"\nModel Accuracy: {accuracy*100:.2f}%")
        print(f"Prediction for Next Candle: {'🔼 UP' if prediction == 1 else '🔽 DOWN'}")
        print(f"Confidence: {confidence*100:.2f}%")

        latest_row = df.iloc[-1]
        reasons = generate_reasoning(latest_row)
        print("\n📊 Nunno's Reasoning:")
        for reason in reasons:
            print(f"• {reason}")

    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        
        

        

