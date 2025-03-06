import numpy as np
from ..utils.config import TARGET_RISE, MAX_LOSS, TIMEOUT_STEPS

def analyze_price_movements(prices):
    """Analyze basic price movement statistics."""
    moves = np.diff(prices)
    avg_move = np.mean(np.abs(moves))
    std_dev = np.std(moves)
    
    print("\nPrice Movement Analysis:")
    print(f"Average move: ${avg_move:.2f}")
    print(f"Standard deviation: ${std_dev:.2f}")
    target_rise = avg_move * 3
    stop_loss = avg_move
    
    print(f"Target rise: ${target_rise:.2f}")
    print(f"Stop loss: ${stop_loss:.2f}")
    
    return avg_move, std_dev

def analyze_patterns(prices, timestamps):
    print("\nPattern Frequency Analysis:")
    print("Checking for $1.00 up moves before $0.50 down moves...")
    
    patterns = []
    for i in range(len(prices) - 100):  # Look ahead up to 100 minutes
        start_price = float(prices[i])
        max_gain = 0
        time_to_target = 0
        
        for j in range(i + 1, min(i + 101, len(prices))):
            current_price = float(prices[j])
            price_diff = current_price - start_price
            
            if price_diff >= 1.0:  # Target rise reached
                patterns.append({
                    'time': timestamps[i],
                    'price': float(prices[i]),
                    'max_gain': price_diff,
                    'minutes': j - i
                })
                break
            elif price_diff <= -0.5:  # Stop loss hit
                break

    print(f"Found {len(patterns)} occurrences in {len(prices)-100} minutes")
    print(f"Pattern frequency: {(len(patterns)/(len(prices)-100))*100:.2f}%")
    
    if patterns:
        avg_time = sum(p['minutes'] for p in patterns) / len(patterns)
        print(f"Average time to reach target: {avg_time:.1f} minutes\n")
        
        print("Example Occurrences:")
        for pattern in patterns[:5]:  # Show first 5 examples
            print(f"Time: {pattern['time']}, Price: ${pattern['price']:.2f}, "
                  f"Max Gain: ${pattern['max_gain']:.2f}, Minutes to Target: {pattern['minutes']}")

def _check_price_pattern(prices, start_idx, target_rise, max_loss):
    """Check if price pattern exists starting at given index."""
    start_price = prices[start_idx]
    max_gain = 0
    max_loss = 0
    
    for i in range(start_idx + 1, min(len(prices), start_idx + 100)):
        price_change = prices[i] - start_price
        max_gain = max(max_gain, price_change)
        max_loss = min(max_loss, price_change)
        
        if max_gain >= target_rise:
            return True
        if max_loss <= -max_loss:
            return False
    
    return False 