import numpy as np
from ..utils.config import TRAIN_SPLIT, VAL_SPLIT, GAP_SIZE

def split_data(prices, timestamps):
    """Split data into training, validation, and test sets."""
    total_points = len(prices)
    train_size = int(total_points * 0.5)
    val_size = int(total_points * 0.25)
    test_size = int(total_points * 0.25)
    gap = 30  # Gap between sets to prevent data leakage
    
    # Split the data
    train_data = prices[:train_size]
    train_times = timestamps[:train_size]
    
    val_data = prices[train_size + gap:train_size + gap + val_size]
    val_times = timestamps[train_size + gap:train_size + gap + val_size]
    
    test_data = prices[train_size + gap + val_size + gap:]
    test_times = timestamps[train_size + gap + val_size + gap:]
    
    print("\nData Split Information:")
    print(f"Total data points: {total_points}")
    print(f"Training data: {len(train_data)} points ({len(train_data)/total_points*100:.1f}%)")
    print(f"Validation data: {len(val_data)} points ({len(val_data)/total_points*100:.1f}%)")
    print(f"Test data: {len(test_data)} points ({len(test_data)/total_points*100:.1f}%)")
    print(f"Gap between sets: {gap} points\n")
    
    print("Price Ranges:")
    print(f"Training: ${float(min(train_data)):.2f} to ${float(max(train_data)):.2f}")
    print(f"Validation: ${float(min(val_data)):.2f} to ${float(max(val_data)):.2f}")
    print(f"Test: ${float(min(test_data)):.2f} to ${float(max(test_data)):.2f}")
    
    return (train_data, train_times), (val_data, val_times), (test_data, test_times) 