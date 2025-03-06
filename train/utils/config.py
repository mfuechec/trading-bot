import os

# At the very start of the file, after imports
print("\n=== Loading Config ===")
print(f"Current working directory: {os.getcwd()}")

# Check if running in Colab
try:
    from google.colab import drive
    IN_COLAB = True
    BASE_PATH = "/content/drive/MyDrive/trading_bot"
except ImportError:
    IN_COLAB = False
    BASE_PATH = os.getcwd()

# File paths (use absolute paths)
DRIVE_PATH = BASE_PATH
HISTORY_FILE = os.path.join(DRIVE_PATH, "training_history.json")
RESULTS_FILE = os.path.join(DRIVE_PATH, "daily_results.json")
CHECKPOINT_DIR = os.path.join(DRIVE_PATH, "checkpoints")
BEST_WEIGHTS_FILE = os.path.join(CHECKPOINT_DIR, "best.weights.h5")

# After setting paths
print("\n=== Path Configuration ===")
print(f"BASE_PATH: {BASE_PATH}")
print(f"DRIVE_PATH: {DRIVE_PATH}")
print(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")
print(f"BEST_WEIGHTS_FILE: {BEST_WEIGHTS_FILE}")

def create_directories():
    """Create all required directories if they don't exist"""
    directories = [DRIVE_PATH, CHECKPOINT_DIR]
    files = [BEST_WEIGHTS_FILE]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Successfully created/verified directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {str(e)}")
    for file in files:
        try:
            with open(file, 'w') as f:
                pass
        except Exception as e:
            print(f"Error creating file {file}: {str(e)}")

# Export the function
__all__ = ['create_directories', 'DRIVE_PATH', 'HISTORY_FILE', 'RESULTS_FILE', 'CHECKPOINT_DIR', 'BEST_WEIGHTS_FILE']

# Initialize directories when config is imported
create_directories()

# Training parameters
EPISODES = 50
BATCH_SIZE = 32
MAX_TRADES = 5

# Trading parameters
TARGET_RISE = 0.5
MAX_LOSS = 0.5
WIN_REWARD = 5
LOSS_PENALTY = -5
TIMEOUT_STEPS = 100

# Environment parameters
STATE_SIZE = 7  # Number of features (5 technical features + position + balance)
ACTION_SIZE = 3  # hold/buy/sell
LOOKBACK_WINDOW = 20

# Data parameters
SYMBOL = 'SPY'
DAYS_TO_DOWNLOAD = 1
TRAIN_SPLIT = 0.5
VAL_SPLIT = 0.25
GAP_SIZE = 30

# Trading parameters
INITIAL_BALANCE = 10000
TRANSACTION_FEE = 0.001  # 0.1% transaction fee

# Environment settings
STATE_SIZE = 7  # Number of features in state
ACTION_SIZE = 3  # HOLD, LONG, SHORT
MAX_TRADES = 5  # Maximum number of concurrent trades
TRADE_DURATION = 60  # Maximum trade duration in minutes
WIN_THRESHOLD = 1.0  # Price movement for win condition
LOSS_THRESHOLD = -0.5  # Price movement for loss condition

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.95  # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Exploration rate decay 