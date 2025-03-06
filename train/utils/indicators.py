import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from train.utils.config import STATE_SIZE

def calculate_rsi(prices, periods=14):
    """Calculate RSI technical indicator"""
    if len(prices) <= periods:
        return np.zeros(len(prices))
    
    # Calculate price changes
    deltas = np.diff(np.insert(prices, 0, prices[0]))
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate average gains and losses
    avg_gains = np.zeros_like(prices)
    avg_losses = np.zeros_like(prices)
    
    # First average
    avg_gains[periods] = np.mean(gains[:periods])
    avg_losses[periods] = np.mean(losses[:periods])
    
    # Calculate subsequent values
    for i in range(periods + 1, len(prices)):
        avg_gains[i] = (avg_gains[i-1] * (periods-1) + gains[i-1]) / periods
        avg_losses[i] = (avg_losses[i-1] * (periods-1) + losses[i-1]) / periods
    
    # Calculate RS and RSI
    rs = np.where(avg_losses != 0, avg_gains/avg_losses, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Fill initial periods
    rsi[:periods] = 50  # Neutral value for initial period
    
    return rsi

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    return upper, lower

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal line"""
    fast_ma = prices.ewm(span=fast, adjust=False).mean()
    slow_ma = prices.ewm(span=slow, adjust=False).mean()
    macd = fast_ma - slow_ma
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_obv(close, volume):
    """Calculate On-Balance Volume"""
    price_change = close.diff()
    obv = (volume * (price_change > 0).astype(int) - 
           volume * (price_change < 0).astype(int)).cumsum()
    return obv 

def identify_price_patterns(df, window=20):
    """Identify common price action patterns"""
    patterns = pd.DataFrame(index=df.index)
    
    # Candlestick body and wicks
    patterns['body'] = df['Close'] - df['Open']
    patterns['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    patterns['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Doji pattern (small body relative to wicks)
    body_size = abs(patterns['body'])
    total_size = df['High'] - df['Low']
    patterns['doji'] = (body_size / total_size) < 0.1
    
    # Pin bars (long wicks)
    patterns['bullish_pin'] = (patterns['lower_wick'] > 2 * body_size) & (patterns['body'] > 0)
    patterns['bearish_pin'] = (patterns['upper_wick'] > 2 * body_size) & (patterns['body'] < 0)
    
    # Inside bars
    patterns['inside_bar'] = (df['High'] <= df['High'].shift(1)) & (df['Low'] >= df['Low'].shift(1))
    
    return patterns

def find_support_resistance(df, window=20, num_points=5):
    """Find support and resistance levels using local extrema"""
    # Find local maxima and minima
    local_max_idx = argrelextrema(df['High'].values, np.greater_equal, order=window)[0]
    local_min_idx = argrelextrema(df['Low'].values, np.less_equal, order=window)[0]
    
    levels = pd.DataFrame(index=df.index)
    levels['is_support'] = False
    levels['is_resistance'] = False
    levels['support_level'] = np.nan
    levels['resistance_level'] = np.nan
    
    # Mark support and resistance points
    levels.iloc[local_min_idx, levels.columns.get_loc('is_support')] = True
    levels.iloc[local_max_idx, levels.columns.get_loc('is_resistance')] = True
    
    # Calculate dynamic support and resistance levels
    for i in range(len(df)):
        if i < window:
            continue
            
        # Look back window periods to find levels
        hist = df.iloc[max(0, i-window):i]
        
        if len(hist) > 0:
            # Support level (average of recent lows)
            support_points = hist['Low'].nsmallest(num_points)
            levels.iloc[i, levels.columns.get_loc('support_level')] = support_points.mean()
            
            # Resistance level (average of recent highs)
            resistance_points = hist['High'].nlargest(num_points)
            levels.iloc[i, levels.columns.get_loc('resistance_level')] = resistance_points.mean()
    
    return levels

def calculate_price_action_features(df, window=20):
    """Calculate price action and support/resistance features"""
    # Get patterns and levels
    patterns = identify_price_patterns(df)
    levels = find_support_resistance(df, window)
    
    # Price action features
    features = pd.DataFrame(index=df.index)
    
    # Normalize body and wick sizes
    features['rel_body'] = patterns['body'] / df['Close']
    features['rel_upper_wick'] = patterns['upper_wick'] / df['Close']
    features['rel_lower_wick'] = patterns['lower_wick'] / df['Close']
    
    # Pattern signals (convert boolean to float)
    features['doji'] = patterns['doji'].astype(float)
    features['bullish_pin'] = patterns['bullish_pin'].astype(float)
    features['bearish_pin'] = patterns['bearish_pin'].astype(float)
    features['inside_bar'] = patterns['inside_bar'].astype(float)
    
    # Support/Resistance features
    features['dist_to_support'] = (df['Close'] - levels['support_level']) / df['Close']
    features['dist_to_resistance'] = (levels['resistance_level'] - df['Close']) / df['Close']
    
    # Breakout signals
    features['support_break'] = (df['Close'] < levels['support_level'].shift(1)).astype(float)
    features['resistance_break'] = (df['Close'] > levels['resistance_level'].shift(1)).astype(float)
    
    return features 

def prepare_features(df):
    """Calculate all technical indicators and prepare feature array"""
    # Verify we have required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame missing required columns. Need {required_columns}")
    
    # Calculate features
    features = pd.DataFrame(index=df.index)
    
    # 1. Returns (percent change in price)
    features['returns'] = df['Close'].pct_change().fillna(0)
    
    # 2. Price relative to 5-period MA
    ma5 = df['Close'].rolling(window=5).mean()
    features['price_to_ma5'] = (df['Close'] / ma5 - 1).fillna(0)
    
    # 3. Price relative to 20-period MA
    ma20 = df['Close'].rolling(window=20).mean()
    features['price_to_ma20'] = (df['Close'] / ma20 - 1).fillna(0)
    
    # 4. RSI
    features['rsi'] = calculate_rsi(df['Close'].values)
    
    # 5. Volume relative to its MA
    volume_ma5 = df['Volume'].rolling(window=5).mean()
    features['relative_volume'] = (df['Volume'] / volume_ma5 - 1).fillna(0)
    
    # Create final feature array with exactly 5 features
    feature_array = np.column_stack([
        features['returns'].values,          # 1. Price returns
        features['price_to_ma5'].values,     # 2. Price vs 5MA
        features['price_to_ma20'].values,    # 3. Price vs 20MA
        features['rsi'].values / 100,        # 4. Normalized RSI
        features['relative_volume'].values    # 5. Relative volume
    ])
    
    # Verify shape
    expected_features = 5  # We have exactly 5 features
    if feature_array.shape[1] != expected_features:
        raise ValueError(f"Feature array has wrong shape. Expected (n_timesteps, {expected_features}), got {feature_array.shape}")
    
    return feature_array 