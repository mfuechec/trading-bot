import os
import pickle
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import random
import tensorflow as tf
from train.utils.config import CHECKPOINT_DIR, BEST_WEIGHTS_FILE

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # No need to add 2, it's included in STATE_SIZE
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Create an empty weights file if it doesn't exist
        if not os.path.exists(BEST_WEIGHTS_FILE):
            print(f"Creating initial weights file at: {BEST_WEIGHTS_FILE}")
            model = self._build_model()
            model.save_weights(BEST_WEIGHTS_FILE)
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:  # Limit memory size
            self.memory.pop(0)

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = np.array(state).reshape(1, -1)  # Ensure correct shape
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        minibatch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        # Debug print to check state shape
        # print(f"First state shape: {self.memory[0][0].shape}")

        # Prepare batch data
        for i, idx in enumerate(minibatch):
            state, action, reward, next_state, done = self.memory[idx]
            # Ensure state is the correct size
            if len(state) != self.state_size:
                print(f"Warning: State size mismatch. Expected {self.state_size}, got {len(state)}")
                continue
            states[i] = state
            next_states[i] = next_state
            actions.append(action)
            rewards.append(float(reward))  # Convert reward to scalar
            dones.append(done)

        # Convert to numpy arrays
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)  # Ensure float type
        dones = np.array(dones)

        # Predict Q-values
        target = self.model.predict(states, verbose=0)
        next_target = self.target_model.predict(next_states, verbose=0)

        # Update Q-values
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_target[i])

        # Train the model
        self.model.fit(states, target, epochs=1, verbose=0)

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Periodically update target network
        if np.random.rand() < 0.1:  # 10% chance to update target network
            self.update_target_model()

    def save_checkpoint(self, name):
        """Save the model checkpoint"""
        try:
            print(f"Current working directory: {os.getcwd()}")
            print(f"Attempting to save weights to: {BEST_WEIGHTS_FILE}")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(BEST_WEIGHTS_FILE), exist_ok=True)
            self.model.save_weights(BEST_WEIGHTS_FILE)
            print("Successfully saved weights")
        except Exception as e:
            print(f"Error saving weights: {str(e)}")

    def load_checkpoint(self, name):
        """Load the model checkpoint"""
        try:
            print(f"Attempting to load weights from: {BEST_WEIGHTS_FILE}")
            if os.path.exists(BEST_WEIGHTS_FILE):
                self.model.load_weights(BEST_WEIGHTS_FILE)
                print("Successfully loaded weights")
                return True
            print("No weights file found")
            return False
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            return False 