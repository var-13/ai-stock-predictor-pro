"""
Real-time Data Collection and Processing Pipeline

This module handles live data feeds, streaming analytics, and real-time predictions.
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import logging
from typing import Dict, List, Optional
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import redis
from kafka import KafkaProducer, KafkaConsumer
import yfinance as yf

logger = logging.getLogger(__name__)


class RealTimeDataCollector:
    """Real-time stock data collection using multiple sources."""
    
    def __init__(self, config):
        self.config = config
        self.symbols = config['data']['symbols']
        self.cache = redis.Redis(host='localhost', port=6379, db=0) if self._redis_available() else None
        self.producer = self._setup_kafka_producer() if self._kafka_available() else None
        
    def _redis_available(self):
        """Check if Redis is available."""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            return True
        except:
            return False
    
    def _kafka_available(self):
        """Check if Kafka is available."""
        try:
            from kafka import KafkaProducer
            producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
            producer.close()
            return True
        except:
            return False
    
    def _setup_kafka_producer(self):
        """Setup Kafka producer for streaming data."""
        try:
            return KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
        except:
            return None
    
    async def collect_websocket_data(self, symbol):
        """Collect real-time data via WebSocket (Alpaca, Polygon, etc.)."""
        # This is a template - replace with actual WebSocket endpoints
        websocket_url = f"wss://socket.polygon.io/stocks"
        
        try:
            async with websockets.connect(websocket_url) as websocket:
                # Subscribe to symbol
                subscribe_msg = {
                    "action": "subscribe",
                    "params": f"T.{symbol}"
                }
                await websocket.send(json.dumps(subscribe_msg))
                
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if data.get('ev') == 'T':  # Trade event
                        trade_data = {
                            'symbol': data.get('sym'),
                            'price': data.get('p'),
                            'volume': data.get('s'),
                            'timestamp': datetime.now(),
                            'exchange': data.get('x')
                        }
                        
                        # Cache data
                        if self.cache:
                            self.cache.setex(
                                f"trade:{symbol}:{int(time.time())}",
                                300,  # 5 minute expiry
                                json.dumps(trade_data, default=str)
                            )
                        
                        # Stream to Kafka
                        if self.producer:
                            self.producer.send('stock_trades', trade_data)
                        
                        yield trade_data
                        
        except Exception as e:
            logger.error(f"WebSocket error for {symbol}: {e}")
    
    def collect_api_data(self, symbol):
        """Collect data via REST API (fallback method)."""
        try:
            # Use yfinance for real-time quotes
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            current_data = {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'volume': info.get('volume', info.get('regularMarketVolume')),
                'change': info.get('regularMarketChangePercent', 0),
                'timestamp': datetime.now(),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE')
            }
            
            return current_data
            
        except Exception as e:
            logger.error(f"API error for {symbol}: {e}")
            return None
    
    def get_news_sentiment(self, symbol, hours_back=24):
        """Get recent news sentiment for symbol."""
        try:
            # News API call
            api_key = self.config['data'].get('news_api_key')
            if not api_key:
                return None
            
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{symbol} stock",
                'from': start_time.isoformat(),
                'to': end_time.isoformat(),
                'sortBy': 'publishedAt',
                'apiKey': api_key,
                'language': 'en',
                'pageSize': 20
            }
            
            response = requests.get(url, params=params)
            news_data = response.json()
            
            if news_data.get('status') == 'ok':
                articles = news_data.get('articles', [])
                
                # Calculate sentiment
                from textblob import TextBlob
                sentiments = []
                
                for article in articles:
                    title = article.get('title', '')
                    description = article.get('description', '')
                    text = f"{title} {description}"
                    
                    blob = TextBlob(text)
                    sentiments.append(blob.sentiment.polarity)
                
                if sentiments:
                    return {
                        'avg_sentiment': np.mean(sentiments),
                        'sentiment_std': np.std(sentiments),
                        'news_count': len(articles),
                        'timestamp': datetime.now()
                    }
            
        except Exception as e:
            logger.error(f"News sentiment error for {symbol}: {e}")
        
        return None


class StreamingPredictor:
    """Real-time prediction engine."""
    
    def __init__(self, model_path, scaler_path, feature_columns):
        self.model = self._load_model(model_path)
        self.scaler = self._load_scaler(scaler_path)
        self.feature_columns = feature_columns
        self.data_buffer = {}
        self.predictions = []
        
    def _load_model(self, model_path):
        """Load trained model."""
        try:
            import joblib
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def _load_scaler(self, scaler_path):
        """Load feature scaler."""
        try:
            import joblib
            return joblib.load(scaler_path)
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            return None
    
    def update_buffer(self, symbol, data):
        """Update data buffer with new market data."""
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []
        
        self.data_buffer[symbol].append(data)
        
        # Keep only last 100 data points
        if len(self.data_buffer[symbol]) > 100:
            self.data_buffer[symbol] = self.data_buffer[symbol][-100:]
    
    def calculate_features(self, symbol):
        """Calculate features from buffered data."""
        if symbol not in self.data_buffer or len(self.data_buffer[symbol]) < 20:
            return None
        
        df = pd.DataFrame(self.data_buffer[symbol])
        df = df.sort_values('timestamp')
        
        # Calculate technical indicators
        features = {}
        
        # Price features
        features['current_price'] = df['price'].iloc[-1]
        features['price_change'] = df['price'].pct_change().iloc[-1]
        
        # Moving averages
        features['sma_5'] = df['price'].rolling(5).mean().iloc[-1]
        features['sma_20'] = df['price'].rolling(20).mean().iloc[-1]
        
        # RSI
        if len(df) >= 14:
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
        else:
            features['rsi'] = 50  # Neutral
        
        # Volume features
        if 'volume' in df.columns:
            features['volume_ratio'] = (df['volume'].iloc[-1] / 
                                       df['volume'].rolling(20).mean().iloc[-1])
        else:
            features['volume_ratio'] = 1.0
        
        # Volatility
        features['volatility'] = df['price'].rolling(20).std().iloc[-1]
        
        return features
    
    def make_prediction(self, symbol):
        """Make real-time prediction for symbol."""
        if not self.model or not self.scaler:
            return None
        
        features = self.calculate_features(symbol)
        if not features:
            return None
        
        try:
            # Create feature vector
            feature_vector = np.array([features.get(col, 0) for col in self.feature_columns])
            feature_vector = feature_vector.reshape(1, -1)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Make prediction
            prediction = self.model.predict(feature_vector_scaled)[0]
            
            prediction_result = {
                'symbol': symbol,
                'predicted_return': prediction,
                'current_price': features['current_price'],
                'timestamp': datetime.now(),
                'features': features
            }
            
            self.predictions.append(prediction_result)
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return None


class RealTimeMonitor:
    """Real-time monitoring and alerting system."""
    
    def __init__(self, config):
        self.config = config
        self.alert_thresholds = {
            'high_confidence_buy': 0.02,  # 2% predicted return
            'high_confidence_sell': -0.02,
            'high_volatility': 0.05,  # 5% volatility
            'unusual_volume': 3.0  # 3x average volume
        }
        self.alerts = []
    
    def check_alerts(self, prediction_data):
        """Check for trading alerts."""
        alerts = []
        
        symbol = prediction_data['symbol']
        predicted_return = prediction_data['predicted_return']
        features = prediction_data['features']
        
        # High confidence buy signal
        if predicted_return > self.alert_thresholds['high_confidence_buy']:
            alerts.append({
                'type': 'BUY_SIGNAL',
                'symbol': symbol,
                'predicted_return': predicted_return,
                'confidence': 'HIGH',
                'timestamp': datetime.now()
            })
        
        # High confidence sell signal
        elif predicted_return < self.alert_thresholds['high_confidence_sell']:
            alerts.append({
                'type': 'SELL_SIGNAL',
                'symbol': symbol,
                'predicted_return': predicted_return,
                'confidence': 'HIGH',
                'timestamp': datetime.now()
            })
        
        # High volatility alert
        if features.get('volatility', 0) > self.alert_thresholds['high_volatility']:
            alerts.append({
                'type': 'HIGH_VOLATILITY',
                'symbol': symbol,
                'volatility': features['volatility'],
                'timestamp': datetime.now()
            })
        
        # Unusual volume alert
        if features.get('volume_ratio', 1) > self.alert_thresholds['unusual_volume']:
            alerts.append({
                'type': 'UNUSUAL_VOLUME',
                'symbol': symbol,
                'volume_ratio': features['volume_ratio'],
                'timestamp': datetime.now()
            })
        
        for alert in alerts:
            self.send_alert(alert)
        
        return alerts
    
    def send_alert(self, alert):
        """Send alert notification."""
        logger.info(f"ALERT: {alert}")
        
        # Save to database
        self.save_alert_to_db(alert)
        
        # Send email/SMS (implement as needed)
        # self.send_email_alert(alert)
        # self.send_slack_alert(alert)
    
    def save_alert_to_db(self, alert):
        """Save alert to database."""
        try:
            with sqlite3.connect('data/alerts.db') as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        type TEXT,
                        symbol TEXT,
                        data TEXT,
                        timestamp TEXT
                    )
                ''')
                
                conn.execute('''
                    INSERT INTO alerts (type, symbol, data, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (
                    alert['type'],
                    alert['symbol'],
                    json.dumps(alert),
                    alert['timestamp'].isoformat()
                ))
                
        except Exception as e:
            logger.error(f"Error saving alert: {e}")


class RealTimePipeline:
    """Main real-time processing pipeline."""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            import yaml
            self.config = yaml.safe_load(f)
        
        self.data_collector = RealTimeDataCollector(self.config)
        self.predictor = StreamingPredictor(
            'models/ensemble.pkl',
            'models/scaler.pkl',
            ['current_price', 'price_change', 'sma_5', 'sma_20', 'rsi', 'volume_ratio', 'volatility']
        )
        self.monitor = RealTimeMonitor(self.config)
        
    async def process_symbol_stream(self, symbol):
        """Process real-time data stream for a symbol."""
        logger.info(f"Starting real-time processing for {symbol}")
        
        while True:
            try:
                # Collect current data
                current_data = self.data_collector.collect_api_data(symbol)
                
                if current_data:
                    # Update predictor buffer
                    self.predictor.update_buffer(symbol, current_data)
                    
                    # Make prediction
                    prediction = self.predictor.make_prediction(symbol)
                    
                    if prediction:
                        # Check for alerts
                        alerts = self.monitor.check_alerts(prediction)
                        
                        # Log prediction
                        logger.info(f"{symbol}: Predicted return = {prediction['predicted_return']:.4f}")
                
                # Wait before next update
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def run_pipeline(self):
        """Run the complete real-time pipeline."""
        logger.info("Starting real-time ML pipeline...")
        
        # Create tasks for each symbol
        tasks = []
        for symbol in self.config['data']['symbols']:
            task = asyncio.create_task(self.process_symbol_stream(symbol))
            tasks.append(task)
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)
    
    def start(self):
        """Start the real-time pipeline."""
        try:
            asyncio.run(self.run_pipeline())
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")


def main():
    """Start the real-time ML pipeline."""
    pipeline = RealTimePipeline()
    pipeline.start()


if __name__ == "__main__":
    main()
