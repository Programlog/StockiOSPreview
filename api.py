import requests
from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import ta
from newsapi import NewsApiClient
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


app = Flask(__name__)

ALPACA_API_KEY = "PK53866Z83921QJD7MEN"
ALPACA_SECRET_KEY = "mEwSIZ8raaouXPLAHIg0dKdc8KiwGNxSHmPV4bLU"

BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"

@app.route("/stock_data", methods=["GET"])
def get_stock_data():
    """
    Example endpoint: 
      GET /stock_data?symbol=AAPL&timeframe=5Min&start=2024-01-03T00:00:00Z&end=2024-01-04T01:02:03.123456789Z&limit=3
    """
    symbol = request.args.get("symbol", "AAPL")
    timeframe = request.args.get("timeframe", "1Day")  # default 1 day
    start = request.args.get("start", "2024-01-01T00:00:00Z")
    end = request.args.get("end", "2024-01-05T00:00:00Z")
    limit = request.args.get("limit", "20")

    # Compose the Alpaca query
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "limit": limit,
        "adjustment": "all",
        "feed": "iex",
        "sort": "asc"
    }
    
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    
    # Make request to Alpaca
    response = requests.get(BASE_URL, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return jsonify(data)
    else:
        return jsonify({"error": "Failed to fetch stock data", "status_code": response.status_code}), 400


@app.route('/latest_bars', methods=['GET'])
def get_latest_bars():
    # Get query parameters
    symbols = request.args.get('symbols')
    feed = request.args.get('feed')  # Optional
    currency = request.args.get('currency', 'USD')  # Default to 'USD'
    
    if not symbols:
        return jsonify({"error": "The 'symbols' query parameter is required."}), 400
    
    # Clean and validate symbols
    symbols_clean = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    if not symbols_clean:
        return jsonify({"error": "No valid symbols provided."}), 400
    symbols_str = ",".join(symbols_clean)

    
    # Construct the Alpaca API URL
    alpaca_url = "https://data.alpaca.markets/v2/stocks/bars/latest"
    params = {
        "symbols": symbols_str,
        "feed": "iex",
        "currency": currency
    }
    
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    
    try:
        response = requests.get(alpaca_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Optionally, process the response data before returning
        data = response.json()
        
        return jsonify(data), response.status_code
    except requests.exceptions.HTTPError as http_err:
        return jsonify({"error": f"HTTP error occurred: {http_err}"}), response.status_code
    except requests.exceptions.RequestException as req_err:
        return jsonify({"error": f"Request error occurred: {req_err}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


"""STOCK PREDICTION API BELOW"""

class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM Model to capture forward and backward dependencies.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Set bidirectional=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        
        # Because it's bidirectional, the final output size is hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, output_size)
        

    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # Take the last timestep output from both directions
        out = self.fc(out[:, -1, :])
        return out

###############################################################################
# LSTM Model (Original or GRU/Transformer can be placed here as well)
###############################################################################
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last time step's output
        return out

###############################################################################
# GRU Model (Optional)
###############################################################################
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

###############################################################################
# StockPredictor Class with Extended Feature Engineering, Data Augmentation,
# Hyperparameter Tuning Hooks, and Backtesting/Evaluation
###############################################################################
class StockPredictor:
    def __init__(
        self, 
        symbol, 
        lookback_days=60,
        model_arch="bilstm",
        hidden_size=100,
        num_layers=2
    ):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_arch = model_arch
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Create a pipeline for text classification
        self.sentiment_pipeline = pipeline(
            task="text-classification",
            model=finbert_model,
            tokenizer=tokenizer,
            return_all_scores=True,  # We'll get probabilities for all classes
            device=0 if torch.cuda.is_available() else -1
        )

    ###########################################################################
    # 1) Data Fetch + Additional Features
    ###########################################################################
    def fetch_data(self, start_date="2y"):
        df = self._download_stock_history(self.symbol, start_date)
        
        if df.empty:
            raise ValueError(f"No data found for symbol: {self.symbol}")
        df.index = df.index.tz_localize(None)

        df = self._add_technical_indicators(df)
        df = self._add_sector_correlation(df, sector_symbol="XLK")

        sentiment_score = self._fetch_sentiment(self.symbol)

        df['sentiment'] = sentiment_score
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        df = df.dropna()
        return df

    def _fetch_sentiment(self, symbol):
        """
        Fetch recent news headlines for the given symbol using NewsAPI
        and compute an advanced sentiment score with FinBERT (multi-class).
        """
        newsapi = NewsApiClient(api_key="8e03339e59534393ad9af39d7c176b79")

        try:
            response = newsapi.get_everything(
                q=symbol,
                language='en',
                sort_by='relevancy',
                page_size=20
            )
        except Exception as e:
            print(f"NewsAPI request failed: {e}")
            return 0.0  # fallback

        articles = response.get('articles', [])
        if not articles:
            return 0.0

        # Gather titles
        headlines = [article.get('title', '') for article in articles if article.get('title')]
        if not headlines:
            return 0.0

        # FinBERT pipeline returns a list-of-lists with return_all_scores=True
        # Each element is e.g.:
        # [
        #   {"label": "positive", "score": 0.70},
        #   {"label": "neutral",  "score": 0.20},
        #   {"label": "negative", "score": 0.10}
        # ]
        results = self.sentiment_pipeline(headlines, truncation=True)

        scores = []
        for row in results:
            # row is a list of dicts like:
            # [{"label":"positive","score":0.70}, {"label":"neutral","score":0.20}, ...]
            p_pos = 0.0
            p_neg = 0.0
            p_neu = 0.0

            for d in row:
                label = d["label"].lower()
                score = d["score"]
                if label == "positive":
                    p_pos = score
                elif label == "negative":
                    p_neg = score
                elif label == "neutral":
                    p_neu = score

            # Example advanced scoring: p_pos - p_neg
            # ignoring neutral. Or you can do something else:
            # e.g., final_score = p_pos + 0.5 * p_neu - p_neg
            final_score = p_pos - p_neg
            scores.append(final_score)

        # Average across all headlines
        avg_score = float(np.mean(scores))
        return avg_score
    def _download_stock_history(self, symbol, period):
        stock = yf.Ticker(symbol)
        return stock.history(period=period)

    def _add_technical_indicators(self, df):
        # =============================================
        # Existing Indicators (RSI, MACD, Bollinger, etc.)
        # =============================================
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['Stoch_K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'] - df['Open']

        # =============================================
        # Additional Momentum Indicators
        # =============================================
        df['Williams_%R'] = ta.momentum.WilliamsRIndicator(
            df['High'], df['Low'], df['Close']
        ).williams_r()
        df['CCI'] = ta.trend.CCIIndicator(
            df['High'], df['Low'], df['Close']
        ).cci()

        # Example fallback if MomentumIndicator is missing:
        df['Momentum'] = ta.momentum.ROCIndicator(df['Close'], window=10).roc()

        # =============================================
        # NEW: Awesome Oscillator (AO)
        # =============================================
        # The AO default is window1=5, window2=34, which matches the standard formula.
        ao = ta.momentum.AwesomeOscillatorIndicator(
            high=df['High'], 
            low=df['Low'], 
            window1=5, 
            window2=34, 
            fillna=False  # or True, if you wish to fill NaNs
        )
        df['AO'] = ao.awesome_oscillator()

        # =============================================
        # Example: S&P 500 Correlation
        # =============================================
        sp500 = yf.download('^GSPC', start=df.index[0], end=df.index[-1])
        sp500.index = sp500.index.tz_localize(None)
        df['SP500_corr'] = df['Close'].rolling(window=20).corr(sp500['Close'])

        return df

    def _add_sector_correlation(self, df, sector_symbol="XLK"):
        
        """Add correlation feature with a sector ETF (e.g., XLK)."""
        sector_data = yf.download(sector_symbol, start=df.index[0], end=df.index[-1])
        if not sector_data.empty:
            sector_data.index = sector_data.index.tz_localize(None)
            df['Sector_corr'] = df['Close'].rolling(window=20).corr(sector_data['Close'])
        else:
            df['Sector_corr'] = 0.0  # fallback if sector data unavailable
        return df

    # Placeholder for sentiment retrieval + scoring
    def _fetch_and_compute_sentiment(self, symbol):
        """
        Pseudo-code for fetching news and computing sentiment.
        In practice, integrate with a library like 'textblob', HuggingFace transformers, or others.
        """
        # news_data = requests.get(...).json()  # from NewsAPI, Alpha Vantage, etc.
        # sentiment_score = compute_sentiment(news_data)  # custom method
        # return sentiment_score
        return 0.0  # placeholder

    ###########################################################################
    # 4) Data Augmentation & Preprocessing
    ###########################################################################
    def prepare_data(self, df, apply_log_transform=True, add_noise=True):
        # Example: Log transform to reduce extreme values
        if apply_log_transform:
            df['Close'] = np.log1p(df['Close'])  # log(1 + x)
            df['Volume'] = np.log1p(df['Volume'])

        features = [
            'Close', 'Volume', 'RSI', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_lower', 'ATR', 'Stoch_K', 'Volume_MA',
            'Returns', 'Price_Change', 'SP500_corr', 
            'Williams_%R', 'CCI', 'Momentum',
            'Sector_corr', 
            'sentiment',
            'day_of_week', 'month'
        ]

        df = df.dropna(subset=features)

        data = df[features].values
        
        # Optional: Add noise for augmentation
        if add_noise:
            noise_factor = 0.001  # adjust as needed
            data = data + noise_factor * np.random.randn(*data.shape)

        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(self.lookback_days, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_days:i])
            y.append(scaled_data[i, 0])

        X = np.array(X) 
        y = np.array(y) 

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)


        return X_tensor, y_tensor, df

    ###########################################################################
    # 2) Hyperparameter Tuning Hook (Pseudo-code with Optuna or Ray Tune)
    ###########################################################################
    def train(
        self, 
        epochs=50, 
        batch_size=32, 
        lr=0.001, 
        apply_log_transform=True, 
        add_noise=True
    ):
        df = self.fetch_data()
        X, y, df_processed = self.prepare_data(
            df, 
            apply_log_transform=apply_log_transform, 
            add_noise=add_noise
        )

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        input_size = X.shape[2]

        # Choose model architecture
        if self.model_arch == "bilstm":
            self.model = BiLSTMModel(
                input_size=input_size, 
                hidden_size=self.hidden_size, 
                num_layers=self.num_layers, 
                output_size=1
            )
        elif self.model_arch == "gru":
            self.model = GRUModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1
            )
        else:
            # default LSTM
            self.model = LSTMModel(
                input_size=input_size, 
                hidden_size=self.hidden_size, 
                num_layers=self.num_layers, 
                output_size=1
            )

        self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size].to(self.device)
                y_batch = y_train[i:i+batch_size].to(self.device)

                outputs = self.model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

        # Evaluate
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test.to(self.device)).squeeze()
            test_loss = criterion(y_pred, y_test.to(self.device)).item()
            print(f"Test Loss: {test_loss:.4f}")

        return test_loss

    ###########################################################################
    # Predict Next Day
    ###########################################################################
    def predict_next_day(self):
        # 1) Fetch unmodified data
        df_original = self.fetch_data()
        if df_original.empty:
            raise ValueError(f"No data found for symbol: {self.symbol}")

        # 2) Keep the last 'Close' from the raw DataFrame for actual price
        #    (i.e., the real last close, unlogged)
        raw_actual_price = df_original['Close'].iloc[-1]

        # 3) Create a copy for data transformations
        df_transformed = df_original.copy()

        # 4) Prepare data on the copy only
        X, _, df_processed = self.prepare_data(
            df_transformed,  # pass the copy 
            apply_log_transform=True,
            add_noise=False
        )

        # 5) Get the last sequence for prediction
        last_sequence = X[-1:].to(self.device)

        # 6) Run prediction in scale space
        self.model.eval()
        with torch.inference_mode():
            predicted_scaled_price = self.model(last_sequence).item()

        # 7) Reconstruct a row with the correct number of features for inverse scaling
        num_features = self.scaler.n_features_in_
        reconstructed = np.zeros((1, num_features))
        reconstructed[0, 0] = predicted_scaled_price  # The predicted 'Close' at index=0

        inverse_row = self.scaler.inverse_transform(reconstructed)[0]
        predicted_price_log = inverse_row[0]  # This is log1p(Close)

        # 8) Exponentiate to get predicted Close in normal space
        predicted_price = np.expm1(predicted_price_log)

        # 9) Now we have the real last close from df_original
        actual_price = raw_actual_price

        # 10) Price change and signal
        price_change = ((predicted_price - actual_price) / actual_price) * 100
        signal = self.generate_signal(
            price_change,
            df_processed['RSI'].iloc[-1],
            df_processed['MACD'].iloc[-1]
        )

        return {
            'symbol': self.symbol,
            'current_price': actual_price,
            'predicted_price': predicted_price,
            'predicted_change': price_change,
            'signal': signal
        }
    def generate_signal(self, price_change, rsi, macd):
        RSI_UPPER = 70
        RSI_LOWER = 30
        PRICE_CHANGE_THRESHOLD = 2.0

        if price_change > PRICE_CHANGE_THRESHOLD and rsi < RSI_UPPER and macd > 0:
            return 'BUY'
        elif price_change < -PRICE_CHANGE_THRESHOLD and rsi > RSI_LOWER and macd < 0:
            return 'SELL'
        else:
            return 'HOLD'

    ###########################################################################
    # 5) Model Evaluation and Backtesting
    ###########################################################################
    def evaluate_model(self, X_test, y_test):
        """
        Calculate financial-specific metrics like Sharpe Ratio, Max Drawdown, and
        Cumulative Returns based on the model’s predictions. 
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test.to(self.device)).squeeze().cpu().numpy()
        y_test_np = y_test.cpu().numpy()

        # Convert from scaled log prices to actual
        # (Pseudo approach: you'd track predictions day-by-day and simulate trades)
        # Here we just return placeholders for demonstration
        sharpe_ratio = self._compute_sharpe_ratio(y_pred, y_test_np)
        max_drawdown = self._compute_max_drawdown(y_pred)
        cumulative_returns = self._compute_cumulative_returns(y_pred, y_test_np)

        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns
        }

    def _compute_sharpe_ratio(self, y_pred, y_true, risk_free_rate=0.01):
        # Placeholder
        returns = y_pred - y_true
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0.0
        return (mean_ret - risk_free_rate) / std_ret

    def _compute_max_drawdown(self, series):
        # Placeholder
        return np.max(np.maximum.accumulate(series) - series)

    def _compute_cumulative_returns(self, y_pred, y_true):
        # Placeholder
        returns = (y_pred - y_true) / y_true
        cumulative = np.cumprod(1 + returns)[-1] - 1
        return cumulative

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', '').upper()
    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400

    # Optional parameters for hyperparam tuning or architecture selection
    model_arch = request.args.get('model_arch', 'bilstm')  # 'lstm', 'gru', or 'bilstm'
    hidden_size = int(request.args.get('hidden_size', '100'))
    num_layers = int(request.args.get('num_layers', '2'))
    epochs = int(request.args.get('epochs', '10'))

    try:
        predictor = StockPredictor(symbol, model_arch=model_arch, hidden_size=hidden_size, num_layers=num_layers)
        predictor.train(epochs=epochs)  # Could do more advanced tuning
        prediction = predictor.predict_next_day()
        return jsonify({
            'symbol': symbol,
            'current_price': prediction['current_price'],
            'predicted_price': prediction['predicted_price'],
            'predicted_change': prediction['predicted_change'],
            'signal': prediction['signal']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True, port=5000)

