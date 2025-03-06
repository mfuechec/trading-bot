# Stock Market DQN Agent

This project implements a Deep Q-Network (DQN) agent that learns to predict whether a stock price will be $0.50 higher or lower in the next 10 minutes based on historical price data.

## Features

- Uses historical 1-minute stock data for training
- Implements a DQN agent with experience replay and target network
- Provides visualization of training rewards
- Saves trained model for future use

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Project Structure

- `dqn_agent.py`: Contains the DQN agent implementation
- `stock_env.py`: Implements the stock trading environment
- `train.py`: Main training script
- `requirements.txt`: List of required Python packages

## Usage

1. Install the requirements:

```bash
pip install -r requirements.txt
```

2. Run the training script:

```bash
python train.py
```

The script will:

- Download historical stock data for SPY (you can modify the symbol in train.py)
- Train the DQN agent
- Save the training rewards plot as 'training_rewards.png'
- Save the trained model as 'dqn_stock_model.pth'

## Model Details

- State Space: Last 10 minutes of price data
- Action Space: 2 actions (predict higher or lower by $0.50)
- Reward: +1 for correct prediction, -1 for incorrect prediction
- Network Architecture: 3 fully connected layers (input → 128 → 64 → output)

## Notes

- The model uses 1-minute historical data for training
- The prediction horizon is 10 minutes
- The model is trained on SPY by default but can be modified for other symbols
- Training progress is displayed every 10 episodes
