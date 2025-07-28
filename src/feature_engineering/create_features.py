"""
Feature Engineering Module for Stock Market Prediction

This module creates technical indicators, sentiment features, and other
engineered features for machine learning models.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yaml
import sqlite3
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Class for calculating technical indicators."""
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages."""
        # Simple Moving Averages
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()
        
        # Moving Average Convergence Divergence (MACD)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        return df
    
    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators."""
        # Relative Strength Index (RSI)
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        
        # Rate of Change
        df['ROC_5'] = df['close'].pct_change(periods=5) * 100
        df['ROC_10'] = df['close'].pct_change(periods=10) * 100
        
        # Williams %R
        df['Williams_R'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        # Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None and not stoch.empty:
            df['Stoch_K'] = stoch.iloc[:, 0]
            df['Stoch_D'] = stoch.iloc[:, 1]
        
        return df
    
    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators."""
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None and not bb.empty:
            df['BB_upper'] = bb.iloc[:, 0]
            df['BB_middle'] = bb.iloc[:, 1]
            df['BB_lower'] = bb.iloc[:, 2]
            df['BB_width'] = df['BB_upper'] - df['BB_lower']
            df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Average True Range (ATR)
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Historical Volatility
        df['volatility_20'] = df['close'].rolling(window=20).std()
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume Weighted Average Price (VWAP)
        df['VWAP'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # On Balance Volume (OBV)
        df['OBV'] = ta.obv(df['close'], df['volume'])
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(periods=5)
        
        # Price Volume Trend
        df['PVT'] = ta.pvt(df['close'], df['volume'])
        
        return df
    
    @staticmethod
    def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based indicators."""
        # Average Directional Index (ADX)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None and not adx.empty:
            df['ADX'] = adx.iloc[:, 0]
            df['DI_plus'] = adx.iloc[:, 1]
            df['DI_minus'] = adx.iloc[:, 2]
        
        # Parabolic SAR
        df['PSAR'] = ta.psar(df['high'], df['low'], df['close'])
        
        # Commodity Channel Index (CCI)
        df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=20)
        
        return df


class SentimentAnalyzer:
    """Class for analyzing sentiment from news articles."""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a text using multiple methods.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if pd.isna(text) or not text.strip():
            return {'vader_compound': 0, 'textblob_polarity': 0, 'textblob_subjectivity': 0}
        
        # VADER Sentiment
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # TextBlob Sentiment
        blob = TextBlob(text)
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity
        }
    
    def aggregate_daily_sentiment(self, news_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Aggregate news sentiment by date for a given symbol.
        
        Args:
            news_df: DataFrame with news articles
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with daily sentiment aggregates
        """
        # Filter news for the symbol
        symbol_news = news_df[news_df['symbol'] == symbol].copy()
        
        if symbol_news.empty:
            return pd.DataFrame()
        
        # Convert published date to datetime
        symbol_news['publishedAt'] = pd.to_datetime(symbol_news['publishedAt'])
        symbol_news['date'] = symbol_news['publishedAt'].dt.date
        
        # Analyze sentiment for each article
        sentiment_data = []
        for _, row in symbol_news.iterrows():
            text = f"{row.get('title', '')} {row.get('description', '')}"
            sentiment = self.analyze_text_sentiment(text)
            sentiment['date'] = row['date']
            sentiment['symbol'] = symbol
            sentiment_data.append(sentiment)
        
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Aggregate by date
        daily_sentiment = sentiment_df.groupby(['date', 'symbol']).agg({
            'vader_compound': ['mean', 'std', 'count'],
            'vader_positive': 'mean',
            'vader_negative': 'mean',
            'textblob_polarity': ['mean', 'std'],
            'textblob_subjectivity': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = [
            'date', 'symbol', 'sentiment_mean', 'sentiment_std', 'news_count',
            'positive_sentiment', 'negative_sentiment', 'polarity_mean', 
            'polarity_std', 'subjectivity_mean'
        ]
        
        # Fill NaN std with 0 (when only one article)
        daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)
        daily_sentiment['polarity_std'] = daily_sentiment['polarity_std'].fillna(0)
        
        return daily_sentiment


class FeatureEngineer:
    """Main class for feature engineering."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the feature engineer with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.technical_indicators = TechnicalIndicators()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def load_data(self) -> tuple:
        """Load data from database or CSV files."""
        try:
            # Try loading from database first
            db_path = self.config['database']['path']
            with sqlite3.connect(db_path) as conn:
                stock_data = pd.read_sql_query("SELECT * FROM stock_prices", conn)
                news_data = pd.read_sql_query("SELECT * FROM news_articles", conn)
                
            logger.info("Loaded data from database")
            
        except Exception as e:
            # Fallback to CSV files
            logger.warning(f"Database loading failed: {e}. Loading from CSV files.")
            stock_data = pd.read_csv('data/raw/stock_data.csv')
            try:
                news_data = pd.read_csv('data/raw/news_data.csv')
            except FileNotFoundError:
                logger.warning("No news data found")
                news_data = pd.DataFrame()
        
        # Convert date columns
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        if not news_data.empty and 'publishedAt' in news_data.columns:
            news_data['publishedAt'] = pd.to_datetime(news_data['publishedAt'])
        
        return stock_data, news_data
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all technical indicator features."""
        logger.info("Creating technical indicators...")
        
        # Sort by symbol and date
        df = df.sort_values(['Symbol', 'date']).reset_index(drop=True)
        
        # Process each symbol separately
        processed_data = []
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol].copy()
            
            # Add technical indicators
            symbol_data = self.technical_indicators.add_moving_averages(symbol_data)
            symbol_data = self.technical_indicators.add_momentum_indicators(symbol_data)
            symbol_data = self.technical_indicators.add_volatility_indicators(symbol_data)
            symbol_data = self.technical_indicators.add_volume_indicators(symbol_data)
            symbol_data = self.technical_indicators.add_trend_indicators(symbol_data)
            
            processed_data.append(symbol_data)
        
        result = pd.concat(processed_data, ignore_index=True)
        logger.info(f"Created technical features for {len(result)} records")
        
        return result
    
    def create_sentiment_features(self, stock_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment features from news data."""
        if news_df.empty:
            logger.warning("No news data available for sentiment analysis")
            # Add empty sentiment columns
            sentiment_columns = [
                'sentiment_mean', 'sentiment_std', 'news_count',
                'positive_sentiment', 'negative_sentiment', 'polarity_mean',
                'polarity_std', 'subjectivity_mean'
            ]
            for col in sentiment_columns:
                stock_df[col] = 0
            return stock_df
        
        logger.info("Creating sentiment features...")
        
        # Aggregate sentiment for all symbols
        all_sentiment = []
        for symbol in stock_df['Symbol'].unique():
            daily_sentiment = self.sentiment_analyzer.aggregate_daily_sentiment(news_df, symbol)
            if not daily_sentiment.empty:
                all_sentiment.append(daily_sentiment)
        
        if not all_sentiment:
            logger.warning("No sentiment data created")
            return stock_df
        
        sentiment_df = pd.concat(all_sentiment, ignore_index=True)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Merge with stock data
        stock_df = stock_df.merge(
            sentiment_df, 
            left_on=['date', 'Symbol'], 
            right_on=['date', 'symbol'], 
            how='left'
        )
        
        # Fill missing sentiment values with neutral sentiment
        sentiment_columns = [
            'sentiment_mean', 'sentiment_std', 'news_count',
            'positive_sentiment', 'negative_sentiment', 'polarity_mean',
            'polarity_std', 'subjectivity_mean'
        ]
        
        for col in sentiment_columns:
            if col in stock_df.columns:
                stock_df[col] = stock_df[col].fillna(0)
        
        # Drop duplicate symbol column
        if 'symbol' in stock_df.columns:
            stock_df = stock_df.drop('symbol', axis=1)
        
        logger.info("Sentiment features created and merged")
        return stock_df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features."""
        logger.info("Creating lag features...")
        
        lag_periods = self.config['features']['lag_periods']
        
        # Features to create lags for
        lag_features = ['close', 'volume', 'price_change', 'RSI_14', 'MACD']
        
        processed_data = []
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')
            
            # Create lag features
            for feature in lag_features:
                if feature in symbol_data.columns:
                    for lag in lag_periods:
                        symbol_data[f'{feature}_lag_{lag}'] = symbol_data[feature].shift(lag)
            
            # Create rolling features
            for feature in ['close', 'volume']:
                if feature in symbol_data.columns:
                    symbol_data[f'{feature}_rolling_mean_5'] = symbol_data[feature].rolling(5).mean()
                    symbol_data[f'{feature}_rolling_std_5'] = symbol_data[feature].rolling(5).std()
            
            processed_data.append(symbol_data)
        
        result = pd.concat(processed_data, ignore_index=True)
        logger.info("Lag features created")
        
        return result
    
    def create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create calendar-based features."""
        logger.info("Creating calendar features...")
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Month
        df['month'] = df['date'].dt.month
        
        # Quarter
        df['quarter'] = df['date'].dt.quarter
        
        # Is it a Monday (often has different market behavior)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        
        # Is it end of month
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Is it end of quarter
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        return df
    
    def process_all_features(self) -> pd.DataFrame:
        """Process all features and save the result."""
        logger.info("Starting feature engineering process...")
        
        # Load data
        stock_data, news_data = self.load_data()
        
        # Create technical features
        stock_data = self.create_technical_features(stock_data)
        
        # Create sentiment features
        stock_data = self.create_sentiment_features(stock_data, news_data)
        
        # Create lag features
        stock_data = self.create_lag_features(stock_data)
        
        # Create calendar features
        stock_data = self.create_calendar_features(stock_data)
        
        # Sort by symbol and date
        stock_data = stock_data.sort_values(['Symbol', 'date']).reset_index(drop=True)
        
        # Save processed data
        stock_data.to_csv('data/processed/engineered_features.csv', index=False)
        
        # Save to database
        try:
            db_path = self.config['database']['path']
            with sqlite3.connect(db_path) as conn:
                stock_data.to_sql('engineered_features', conn, if_exists='replace', index=False)
            logger.info("Saved engineered features to database")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
        
        logger.info(f"Feature engineering completed. Final dataset shape: {stock_data.shape}")
        logger.info(f"Features created: {list(stock_data.columns)}")
        
        return stock_data


def main():
    """Main function to run feature engineering."""
    engineer = FeatureEngineer()
    engineered_data = engineer.process_all_features()
    
    # Display summary statistics
    print(f"\nFeature Engineering Summary:")
    print(f"Total records: {len(engineered_data)}")
    print(f"Total features: {len(engineered_data.columns)}")
    print(f"Date range: {engineered_data['date'].min()} to {engineered_data['date'].max()}")
    print(f"Symbols: {engineered_data['Symbol'].unique()}")
    
    # Check for missing values
    missing_pct = (engineered_data.isnull().sum() / len(engineered_data) * 100).sort_values(ascending=False)
    missing_features = missing_pct[missing_pct > 0]
    
    if not missing_features.empty:
        print(f"\nFeatures with missing values:")
        print(missing_features.head(10))
    
    print(f"\nEngineered features saved to: data/processed/engineered_features.csv")


if __name__ == "__main__":
    main()
