from train.utils.config import WIN_REWARD, LOSS_PENALTY
from train.utils.feature_flags import FeatureFlags

def process_step(env, action):
    """
    Process a single step in the trading environment
    
    Args:
        env: The StockEnv instance
        action: The action to take (0: HOLD, 1: LONG, 2: SHORT)
    
    Returns:
        tuple: (next_state, reward, done, info)
    """
    reward = 0
    positions_to_remove = []
    
    # Check if we're at the end of data
    if env.current_step >= len(env.prices) - 1:
        return env.get_state(), 0, True, {}
    
    # Process existing positions first
    if env.has_position():
        for position in env.positions[:]:
            current_price = float(env.prices[env.current_step])
            entry_price = float(position['entry_price'])
            elapsed_time = env.current_step - position['entry_time']
            
            # Calculate absolute price change
            if position['type'] == 'LONG':
                price_change = current_price - entry_price
            else:  # SHORT
                price_change = entry_price - current_price
                
            if FeatureFlags.DEBUG_POSITIONS:
                print(f"\nChecking position:")
                print(f"- Type: {position['type']}")
                print(f"- Entry price: ${entry_price}")
                print(f"- Current price: ${current_price}")
                print(f"- Price change: ${price_change:.2f}")
            
            # Check win/loss/timeout conditions
            if price_change >= 0.50:  # Win at $0.50 gain
                reward = WIN_REWARD
                positions_to_remove.append(position)
                print(f"WIN: {position['type']} position closed. ${price_change:.2f} P/L")
            elif price_change <= -0.50:  # Loss at $0.50 loss
                reward = LOSS_PENALTY
                positions_to_remove.append(position)
                print(f"LOSS: {position['type']} position closed. ${price_change:.2f} P/L")
            elif elapsed_time >= env.TRADE_DURATION:  # Timeout
                reward = price_change / 10
                positions_to_remove.append(position)
                print(f"TIMEOUT: {position['type']} position closed. ${price_change:.2f} P/L")
            elif FeatureFlags.DEBUG_POSITION_REMAINS_OPEN:
                print("No conditions met - position remains open")
    
    # Process new action (removed elif to allow multiple positions)
    if action != 0 and env.active_trades < env.MAX_TRADES:
        current_price = float(env.prices[env.current_step])
        trend = env.features[env.current_step][0]
        
        # Simplified trend alignment check
        can_trade = True
        if action == 1:  # LONG
            can_trade = trend >= 0
        elif action == 2:  # SHORT
            can_trade = trend <= 0
            
        if can_trade:
            position_type = 'LONG' if action == 1 else 'SHORT'
            env.open_position(position_type, current_price, env.current_step)
            print(f"Opening {position_type} position at price {current_price:.2f}")
            reward = 0
        elif FeatureFlags.ACTION_NOT_ALIGNED_WITH_TREND:
            print(f"Action {action} not aligned with trend {trend}")
            reward = -0.1
    
    # Remove closed positions and decrement active trades
    for position in positions_to_remove:
        if env.mode == 'train':
            if position in env.train_positions:
                env.train_positions.remove(position)
                env.train_active_trades = max(0, env.train_active_trades - 1)
        elif env.mode == 'validation':
            if position in env.val_positions:
                env.val_positions.remove(position)
                env.val_active_trades = max(0, env.val_active_trades - 1)
        else:  # test mode
            if position in env.test_positions:
                env.test_positions.remove(position)
                env.test_active_trades = max(0, env.test_active_trades - 1)
        print(f"Remaining {env.mode} active trades: {env.active_trades}/{env.MAX_TRADES}")
    
    # Update step
    env.current_step += 1
    done = env.current_step >= len(env.prices) - 1
    next_state = env.get_state()
    
    if FeatureFlags.DEBUG_REWARDS:
        print(f"Reward calculation:")
        print(f"- Base reward: {reward}")
        print(f"- Final reward: {reward}")
    
    return next_state, reward, done, {} 