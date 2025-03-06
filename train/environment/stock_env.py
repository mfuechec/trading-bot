import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
from train.utils.config import (
    MAX_TRADES,
    INITIAL_BALANCE,
    TARGET_RISE,
    MAX_LOSS,
    WIN_REWARD,
    LOSS_PENALTY,
    STATE_SIZE,
    ACTION_SIZE,
    LOOKBACK_WINDOW,
    TRANSACTION_FEE,
    TRADE_DURATION,
    WIN_THRESHOLD,
    LOSS_THRESHOLD
)
from train.environment.step_logic import process_step

class StockEnv(Env):
    def __init__(self, prices, features, timestamps, mode='train'):
        super().__init__()
        
        # Store input data
        self.prices = prices
        self.features = features
        self.timestamps = timestamps
        self.data = features  # Used for length calculation in step_logic.py
        
        # Environment settings
        self.mode = mode
        self.MAX_TRADES = MAX_TRADES
        self.TRADE_DURATION = TRADE_DURATION
        self.WIN_THRESHOLD = WIN_THRESHOLD
        self.LOSS_THRESHOLD = LOSS_THRESHOLD
        
        # Reset state
        self.reset()
        
        # Add trend tracking
        self.lookback = LOOKBACK_WINDOW
        self.trend_direction = 0  # 1 for uptrend, -1 for downtrend, 0 for neutral
        
        # Verify feature dimension
        if self.features.shape[1] != STATE_SIZE - 2:  # -2 for position and balance
            raise ValueError(f"Feature dimension mismatch. Expected {STATE_SIZE-2}, got {self.features.shape[1]}")
        
        # Define action and observation spaces
        self.action_space = Discrete(ACTION_SIZE)
        
        # Observation space includes: features + position + balance
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(STATE_SIZE,),
            dtype=np.float32
        )
        
        self.state_size = STATE_SIZE
    
    def reset(self):
        """Reset the environment state."""
        self.current_step = 0
        
        # Reset all trade tracking
        self.train_active_trades = 0
        self.val_active_trades = 0
        self.test_active_trades = 0
        self.train_positions = []
        self.val_positions = []
        self.test_positions = []
        
        print(f"\nNew Episode: Maintaining {self.train_active_trades} training trades and {self.val_active_trades} validation trades")
        return self.get_state()
    
    def _get_observation(self):
        """Get the current observation (state)"""
        current_features = self.features[self.current_step]  # Shape: (5,)
        position = np.array([self.position])                 # Shape: (1,)
        balance = np.array([self.balance / self.initial_balance])                   # Shape: (1,)
        
        # Debug prints to verify shapes
        # print(f"Features shape: {current_features.shape}")
        # print(f"Position shape: {position.shape}")
        # print(f"Balance shape: {balance.shape}")
        
        # Ensure all arrays are 1-dimensional and concatenate
        obs = np.concatenate([
            current_features.ravel(),  # Ensure 1D
            position.ravel(),         # Ensure 1D
            balance.ravel()           # Ensure 1D
        ])
        
        return obs
    
    def _update_trend(self):
        """Update trend direction using multiple indicators"""
        if self.current_step < self.lookback:
            return
            
        # Get price data for lookback period
        recent_prices = self.prices[max(0, self.current_step-self.lookback):self.current_step+1]
        
        # 1. Moving Average Crossover
        ma20 = np.mean(recent_prices[-20:])
        ma50 = np.mean(recent_prices[-50:])
        ma_cross = 1 if ma20 > ma50 else -1
        
        # 2. Price Slope
        slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        slope_direction = 1 if slope > 0 else -1
        
        # 3. Higher Highs / Lower Lows
        highs = np.max(recent_prices[-5:])
        lows = np.min(recent_prices[-5:])
        prev_highs = np.max(recent_prices[-10:-5])
        prev_lows = np.min(recent_prices[-10:-5])
        hh_ll = 1 if highs > prev_highs and lows > prev_lows else -1
        
        # Combine signals (majority vote)
        signals = [ma_cross, slope_direction, hh_ll]
        self.trend_direction = np.sign(np.sum(signals))
        
        # Debug print
        # print(f"Trend Direction: {self.trend_direction}")
        # print(f"Signals: MA={ma_cross}, Slope={slope_direction}, HH/LL={hh_ll}")
    
    def _is_trend_aligned(self, action):
        """Check if action aligns with current trend"""
        if self.trend_direction > 0:  # Uptrend
            return action == 1  # Only allow LONG
        elif self.trend_direction < 0:  # Downtrend
            return action == 2  # Only allow SHORT
        return True  # Allow all actions in neutral trend
    
    def step(self, action):
        return process_step(self, action)
    
    def render(self):
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position: {self.position}")
        print(f"Current price: ${self.prices[self.current_step]:.2f}")
        print(f"Portfolio value: ${(self.balance + self.position * self.prices[self.current_step]):.2f}")

    def _update_open_trades(self):
        """Update and process open trades."""
        completed = []
        remaining = []
        current_price = self.prices[self.current_step]
        
        for trade in self.open_trades:
            price_change = current_price - trade['entry_price']
            trade['max_gain'] = max(trade['max_gain'], price_change)
            trade['max_loss'] = min(trade['max_loss'], price_change)
            trade['steps_open'] += 1
            
            # Check for win
            if price_change >= TARGET_RISE:
                trade['result'] = WIN_REWARD
                completed.append(trade)
                continue
                
            # Check for loss
            if price_change <= -MAX_LOSS:
                trade['result'] = LOSS_PENALTY
                completed.append(trade)
                continue
                
            # Check for timeout
            if trade['steps_open'] >= TIMEOUT_STEPS:
                trade['result'] = int(price_change * 10)  # 1/10 of price difference
                completed.append(trade)
                continue
                
            remaining.append(trade)
            
        self.open_trades = remaining
        return completed 

    def verify_trade_limits(self):
        """Verify that trade limits are being respected for current mode"""
        active_trades = self.train_active_trades if self.mode == 'train' else self.val_active_trades
        if active_trades > MAX_TRADES:
            print(f"ERROR: {self.mode} trade limit exceeded! Active trades: {active_trades}/{MAX_TRADES}")
            return False
        if active_trades < 0:
            print(f"ERROR: Negative {self.mode} active trades! Count: {active_trades}")
            return False
        return True 

    @property
    def positions(self):
        """Get positions list for current mode."""
        if self.mode == 'train':
            return self.train_positions
        elif self.mode == 'validation':
            return self.val_positions
        return self.test_positions
        
    @property
    def active_trades(self):
        """Get active trades count for current mode."""
        if self.mode == 'train':
            return self.train_active_trades
        elif self.mode == 'validation':
            return self.val_active_trades
        return self.test_active_trades
        
    def has_position(self):
        """Check if there are any open positions in current mode."""
        return len(self.positions) > 0
        
    def open_position(self, position_type, entry_price, entry_time):
        """Opens a new position and increments active trades counter."""
        position = {
            'type': position_type,
            'entry_price': entry_price,
            'entry_time': entry_time
        }
        
        # Add position to correct list based on mode
        if self.mode == 'train':
            self.train_positions.append(position)
            self.train_active_trades += 1
        elif self.mode == 'validation':
            self.val_positions.append(position)
            self.val_active_trades += 1
        else:  # test mode
            self.test_positions.append(position)
            self.test_active_trades += 1
            
        print(f"Opened {position_type} position in {self.mode} mode. Active trades: {self.active_trades}")

    def get_state(self):
        """Returns the current state of the environment."""
        if self.current_step >= len(self.features):
            return np.zeros(self.state_size)
            
        # Get current features
        current_features = self.features[self.current_step]
        
        # Add position information to state
        has_long = 0
        has_short = 0
        for pos in self.positions:
            if pos['type'] == 'LONG':
                has_long = 1
            elif pos['type'] == 'SHORT':
                has_short = 1
                
        # Combine features with position info
        state = np.concatenate([
            current_features,
            [has_long, has_short]
        ])
        
        return state 