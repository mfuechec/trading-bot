import yfinance as yf
from datetime import datetime, timedelta
from ..utils.config import SYMBOL, DAYS_TO_DOWNLOAD

def download_stock_data(symbol=SYMBOL, days=DAYS_TO_DOWNLOAD):
    """
    Download minute-level stock data for the specified symbol.
    
    Args:
        symbol (str): Stock symbol to download (default: 'SPY')
        days (int): Number of days of data to download (default: 7)
        
    Returns:
        tuple: (prices, timestamps) where prices is a numpy array of closing prices
               and timestamps is a pandas DatetimeIndex
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"Downloading {days} days of data for {symbol}")
    data = yf.download(symbol, start=start_date, end=end_date, interval='1m')
    
    if data.empty:
        raise ValueError(f"No data downloaded for {symbol}")
        
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Total data points: {len(data)}")
    
    return data['Close'].values, data.index

def verify_data(prices, timestamps):
    """
    Verify downloaded data and print basic statistics.
    
    Args:
        prices (numpy.array): Array of closing prices
        timestamps (pandas.DatetimeIndex): Array of timestamps
    """
    print("\nInitial Data Verification:")
    print("First 10 prices:")
    for price in prices[:10]:
        print(f"  ${float(price):.2f}")
    print(f"Data range: ${float(prices.min()):.2f} to ${float(prices.max()):.2f}")
    print(f"Total data points: {len(prices)}") 