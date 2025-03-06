import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import numpy as np
from train.agents.dqn_agent import DQNAgent
from train.environment.stock_env import StockEnv
from train.utils.config import (
    STATE_SIZE, 
    ACTION_SIZE, 
    MAX_TRADES,
    HISTORY_FILE, 
    RESULTS_FILE, 
    CHECKPOINT_DIR,
    BEST_WEIGHTS_FILE
)
from train.utils.indicators import prepare_features

def load_or_create_history():
    """Load or create training history file"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        last_date = datetime.strptime(history['last_trained_day'], '%Y-%m-%d')
        next_date = last_date + timedelta(days=1)
    else:
        next_date = datetime(2025, 2, 10)
        history = {'days_trained': 0, 'last_trained_day': None}
    
    return history, next_date

def train_day():
    # Load training history and get next day
    history, start_date = load_or_create_history()
    end_date = start_date + timedelta(days=1)
    
    print(f"\nTraining on day: {start_date.strftime('%Y-%m-%d')}")
    
    # Download data
    df = yf.download("SPY", 
                     start=start_date, 
                     end=end_date, 
                     interval="1m")
    
    if len(df) < 100:  # Minimum data requirement
        print("Not enough data for this day, skipping...")
        return
    
    print(f"Loaded {len(df)} data points")
    
    # Prepare features
    features = prepare_features(df)
    
    # Split data
    train_size = int(len(df) * 0.6)
    val_size = int(len(df) * 0.2)
    
    train_data = {
        'prices': df[:train_size]['Close'].values,
        'features': features[:train_size],
        'timestamps': df[:train_size].index
    }
    
    val_data = {
        'prices': df[train_size:train_size + val_size]['Close'].values,
        'features': features[train_size:train_size + val_size],
        'timestamps': df[train_size:train_size + val_size].index
    }
    
    test_data = {
        'prices': df[train_size + val_size:]['Close'].values,
        'features': features[train_size + val_size:],
        'timestamps': df[train_size + val_size:].index
    }
    
    # Create environments with modes
    train_env = StockEnv(prices=train_data['prices'], 
                        features=train_data['features'],
                        timestamps=train_data['timestamps'],
                        mode='train')
    
    val_env = StockEnv(prices=val_data['prices'],
                       features=val_data['features'],
                       timestamps=val_data['timestamps'],
                       mode='validation')
    
    test_env = StockEnv(prices=test_data['prices'],
                        features=test_data['features'],
                        timestamps=test_data['timestamps'],
                        mode='test')
    
    # Initialize or load agent
    print("\n=== Creating DQN Agent ===")
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    
    # Try to load previous checkpoint
    print("\n=== Loading Previous Checkpoint ===")
    if not agent.load_checkpoint(BEST_WEIGHTS_FILE):
        print("No previous checkpoint found. Starting fresh training.")
    
    # Train for this day
    print("\nStarting training...")
    best_val_reward = float('-inf')
    total_reward = 0
    episodes = 100  # Number of episodes to train
    
    for episode in range(episodes):
        # Training episode
        print(f"\n=== Training Episode {episode + 1}/{episodes} ===")
        state = train_env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1
            
            if not train_env.verify_trade_limits():
                print("Trade limit verification failed!")
                break
            
            # Train on past experiences
            if len(agent.memory) > agent.batch_size:
                agent.train()
            
            if steps % 100 == 0:
                print(".", end="", flush=True)
        
        total_reward += episode_reward
        
        # Single Validation episode after each training episode
        print(f"\n=== Validation Episode {episode + 1}/{episodes} ===")
        val_reward = 0
        state = val_env.reset() if episode == 0 else next_state  # Only reset on first episode
        done = False
        val_steps = 0
        
        while not done:
            val_steps += 1
            action = agent.act(state, training=False)
            
            # Log action being taken
            action_type = ["HOLD", "LONG", "SHORT"][action]
            print(f"Step {val_steps}: Taking action {action_type}")
            
            next_state, reward, done, _ = val_env.step(action)
            val_reward += float(reward)
            state = next_state
            
            # Print status every step
            print(f"  Active Trades: {val_env.active_trades}/{MAX_TRADES}")
            print(f"  Step Reward: {float(reward):.2f}")
            print(f"  Total Val Reward: {float(val_reward):.2f}")
            
            if not val_env.verify_trade_limits():
                print("Trade limit verification failed during validation!")
                break
        
        # Save if validation performance improved
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            agent.save_checkpoint('best')
            print(f"\nNew best validation reward: {float(val_reward):.2f}")
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        print(f"Training Steps: {steps}, Train Reward: {float(episode_reward):.2f}")
        print(f"Validation Steps: {val_steps}, Val Reward: {float(val_reward):.2f}")

    # Test agent performance
    print("\nTesting agent performance...")
    test_rewards = []
    test_trades_opened = 0
    test_trades_closed = 0
    test_steps = 0
    for _ in range(5):  # Run 5 test episodes
        state = test_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = test_env.step(action)
            episode_reward += reward
            state = next_state
            if reward > 0:
                test_trades_closed += 1
            else:
                test_trades_opened += 1
            test_steps += 1
        test_rewards.append(episode_reward)
    
    # Save results
    results = {
        'training_reward': total_reward / episodes,
        'validation_reward': best_val_reward,
        'test_reward': float(episode_reward),
        'test_trades_opened': test_trades_opened,
        'test_trades_closed': test_trades_closed,
        'test_steps': test_steps,
        'final_active_trades': int(test_env.active_trades),
        'total_pnl': float(test_env.balance - test_env.initial_balance),
        'winning_trades': test_trades_closed - test_trades_opened,
        'losing_trades': test_trades_opened - test_trades_closed,
        'win_rate': (test_trades_closed - test_trades_opened) / test_trades_closed if test_trades_closed > 0 else 0
    }
    
    # Update and save training history
    history['days_trained'] += 1
    history['last_trained_day'] = start_date.strftime('%Y-%m-%d')
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)
    
    # Save daily results
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    all_results[history['last_trained_day']] = results
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f)
    
    print(f"\nDay {history['last_trained_day']} completed:")
    print(f"Average training reward: {results['training_reward']:.2f}")
    print(f"Test reward: {results['test_reward']:.2f}")
    print(f"Test trades opened: {results['test_trades_opened']}")
    print(f"Test trades closed: {results['test_trades_closed']}")
    print(f"Test steps: {results['test_steps']}")
    print(f"Final active trades: {results['final_active_trades']}")
    print(f"Total P/L: {results['total_pnl']:.2f}")
    print(f"Winning trades: {results['winning_trades']}")
    print(f"Losing trades: {results['losing_trades']}")
    print(f"Win rate: {results['win_rate']:.2f}")

if __name__ == "__main__":
    train_day() 