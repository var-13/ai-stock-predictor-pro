"""
Stock Market Data Collection Module

This module handles downloading and preprocessing of stock market data
from various sources including Yahoo Finance and news APIs.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import sqlite3
import yaml
import os
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """Main class for collecting stock market and news data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the data collector with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.symbols = self.config['data']['symbols']
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date'] or datetime.now().strftime('%Y-%m-%d')
        
        # Create data directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
    def collect_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Collect historical stock data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Downloading data for {symbol}")
            
            # Download data from Yahoo Finance
            stock = yf.Ticker(symbol)
            data = stock.history(start=self.start_date, end=self.end_date)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Add symbol column
            data['Symbol'] = symbol
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Rename columns to standard format
            data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            # Calculate basic features
            data['price_change'] = data['close'].pct_change()
            data['price_change_next_day'] = data['close'].shift(-1) / data['close'] - 1
            
            # Volume features
            data['volume_sma_20'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma_20']
            
            logger.info(f"Successfully collected {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def collect_all_stocks(self) -> pd.DataFrame:
        """
        Collect data for all symbols in configuration.
        
        Returns:
            Combined DataFrame with all stock data
        """
        all_data = []
        
        for symbol in self.symbols:
            data = self.collect_stock_data(symbol)
            if not data.empty:
                all_data.append(data)
            
            # Add delay to respect API limits
            time.sleep(1)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            # Save to CSV
            combined_data.to_csv('data/raw/stock_data.csv', index=False)
            logger.info(f"Saved combined data with {len(combined_data)} records")
            return combined_data
        else:
            logger.warning("No data collected for any symbols")
            return pd.DataFrame()
    
    def collect_market_indices(self) -> pd.DataFrame:
        """
        Collect market index data (S&P 500, VIX, etc.)
        
        Returns:
            DataFrame with market index data
        """
        indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'DIA': 'Dow Jones',
            '^VIX': 'VIX'
        }
        
        index_data = []
        
        for symbol, name in indices.items():
            try:
                logger.info(f"Downloading {name} data")
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=self.start_date, end=self.end_date)
                
                if not data.empty:
                    data['Index'] = name
                    data.reset_index(inplace=True)
                    data.rename(columns={
                        'Date': 'date',
                        'Close': 'close'
                    }, inplace=True)
                    index_data.append(data[['date', 'close', 'Index']])
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting {name} data: {str(e)}")
        
        if index_data:
            combined_indices = pd.concat(index_data, ignore_index=True)
            combined_indices.to_csv('data/raw/market_indices.csv', index=False)
            return combined_indices
        else:
            return pd.DataFrame()
    
    def collect_news_data(self, symbol: str, days_back: int = 30) -> List[Dict]:
        """
        Collect news articles for a given symbol.
        
        Args:
            symbol: Stock ticker symbol
            days_back: Number of days to look back for news
            
        Returns:
            List of news articles
        """
        # Note: This requires a NewsAPI key from https://newsapi.org/
        api_key = self.config['data'].get('news_api_key')
        
        if not api_key or api_key == 'your_newsapi_key_here':
            logger.warning("No valid NewsAPI key found. Skipping news collection.")
            return []
        
        try:
            from newsapi import NewsApiClient
            newsapi = NewsApiClient(api_key=api_key)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Search for articles
            company_names = {
                'AAPL': 'Apple',
                'GOOGL': 'Google',
                'MSFT': 'Microsoft',
                'AMZN': 'Amazon',
                'TSLA': 'Tesla',
                'NVDA': 'NVIDIA',
                'META': 'Meta',
                'NFLX': 'Netflix'
            }
            
            company_name = company_names.get(symbol, symbol)
            
            articles = newsapi.get_everything(
                q=f"{company_name} stock",
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            
            logger.info(f"Found {len(articles['articles'])} articles for {symbol}")
            return articles['articles']
            
        except Exception as e:
            logger.error(f"Error collecting news for {symbol}: {str(e)}")
            return []
    
    def save_to_database(self, data: pd.DataFrame, table_name: str):
        """
        Save data to SQLite database.
        
        Args:
            data: DataFrame to save
            table_name: Name of the database table
        """
        try:
            db_path = self.config['database']['path']
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            with sqlite3.connect(db_path) as conn:
                data.to_sql(table_name, conn, if_exists='replace', index=False)
                logger.info(f"Saved {len(data)} records to {table_name} table")
                
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")


def main():
    """Main function to run data collection."""
    logger.info("Starting stock market data collection...")
    
    # Initialize collector
    collector = StockDataCollector()
    
    # Collect stock data
    stock_data = collector.collect_all_stocks()
    if not stock_data.empty:
        collector.save_to_database(stock_data, 'stock_prices')
    
    # Collect market indices
    index_data = collector.collect_market_indices()
    if not index_data.empty:
        collector.save_to_database(index_data, 'market_indices')
    
    # Collect news data for each symbol
    all_news = []
    for symbol in collector.symbols:
        news = collector.collect_news_data(symbol)
        for article in news:
            article['symbol'] = symbol
            all_news.append(article)
        time.sleep(1)  # Respect API limits
    
    if all_news:
        news_df = pd.DataFrame(all_news)
        news_df.to_csv('data/raw/news_data.csv', index=False)
        collector.save_to_database(news_df, 'news_articles')
    
    logger.info("Data collection completed!")


if __name__ == "__main__":
    main()
