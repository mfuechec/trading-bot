import unittest
import numpy as np
from train.environment.stock_env import StockEnv
from train.utils.config import MAX_TRADES, WIN_THRESHOLD, LOSS_THRESHOLD, WIN_REWARD, LOSS_PENALTY

class TestStockEnv(unittest.TestCase):
    def setUp(self):
        """Set up test environment with sample data."""
        self.timestamps = np.array(range(20))
        
    def test_trend_alignment(self):
        """Test that positions can only be opened in alignment with the trend."""
        # Create features with alternating trends for trend test
        features = np.array([
            [1, 0, 0, 0, 0],   # Uptrend
            [-1, 0, 0, 0, 0],  # Downtrend
            [1, 0, 0, 0, 0],   # Uptrend
            [-1, 0, 0, 0, 0],  # Downtrend
            [1, 0, 0, 0, 0],   # Uptrend
        ] * 2)
        
        prices = np.array([100.0] * 10)
        env = StockEnv(prices, features, self.timestamps[:10], mode='train')

        # Test each step with different trends
        for step in range(5):
            env.current_step = step
            trend = features[step][0]
            
            # Try to open LONG position
            state, reward, done, _ = env.step(1)  # Action 1 = LONG
            if trend >= 0:
                self.assertEqual(
                    env.active_trades, 
                    1, 
                    f"Step {step}: Should open LONG position in uptrend (trend={trend})"
                )
            else:
                self.assertEqual(
                    env.active_trades, 
                    0, 
                    f"Step {step}: Should not open LONG position in downtrend (trend={trend})"
                )
            
            # Reset for next test
            env.reset()
            env.current_step = step
            
            # Try to open SHORT position
            state, reward, done, _ = env.step(2)  # Action 2 = SHORT
            if trend <= 0:
                self.assertEqual(
                    env.active_trades, 
                    1, 
                    f"Step {step}: Should open SHORT position in downtrend (trend={trend})"
                )
            else:
                self.assertEqual(
                    env.active_trades, 
                    0, 
                    f"Step {step}: Should not open SHORT position in uptrend (trend={trend})"
                )
            
            # Reset for next iteration
            env.reset()
            
            print(f"\nStep {step}:")
            print(f"Trend: {trend}")
            print(f"LONG attempt result: {env.active_trades} trades")
            print(f"SHORT attempt result: {env.active_trades} trades")

    def test_max_trades_limit(self):
        """Test that we cannot exceed MAX_TRADES limit."""
        # Create data with consistent uptrend for MAX_TRADES test
        features = np.array([
            [1, 0, 0, 0, 0],  # All uptrend to allow LONG positions
        ] * 20)
        
        prices = np.array([100.0] * 20)
        env = StockEnv(prices, features, self.timestamps, mode='train')
        
        # Try to open more positions than MAX_TRADES
        for i in range(MAX_TRADES + 2):  # Try to open 2 more than allowed
            state, reward, done, _ = env.step(1)  # Try to open LONG position
            
            # Print current state for debugging
            print(f"Attempt {i+1}: Active trades = {env.active_trades}")
            
            # Verify we never exceed MAX_TRADES
            self.assertLessEqual(
                env.active_trades, 
                MAX_TRADES,
                f"Active trades ({env.active_trades}) exceeded MAX_TRADES ({MAX_TRADES})"
            )
        
        # Verify final state
        self.assertEqual(
            env.active_trades, 
            MAX_TRADES,
            f"Final active trades ({env.active_trades}) should equal MAX_TRADES ({MAX_TRADES})"
        )

    def test_reward_calculation(self):
        """Test reward calculations for wins, losses, and timeouts."""
        print("\n=== Testing Reward Calculations ===")
        
        # Create price data to test different scenarios
        prices = np.array([100.00] * 100)
        prices[1] = 100.51  # Win for LONG (+$0.51)
        prices[2] = 99.49   # Loss for LONG (-$0.51)
        prices[3:63] = 100.10  # Small gain for LONG during timeout period
        
        # Uptrend for all periods to allow LONG positions
        features = np.array([[1, 0, 0, 0, 0]] * 100)
        timestamps = np.array(range(100))
        
        env = StockEnv(prices, features, timestamps, mode='train')
        
        # Test 1: Win scenario for LONG position
        print("\n1. Testing WIN scenario:")
        env.reset()
        print(f"Opening LONG at ${prices[0]:.2f}")
        state, reward, done, _ = env.step(1)  # Open LONG position
        print(f"Open position reward: {reward}")
        self.assertEqual(reward, 0, "Opening position should give 0 reward")
        
        env.current_step = 1
        print(f"Moving to price ${prices[1]:.2f} (gain: +$0.51)")
        state, reward, done, _ = env.step(0)  # HOLD
        print(f"Win reward received: {reward}")
        self.assertEqual(reward, WIN_REWARD, f"Expected WIN_REWARD ({WIN_REWARD})")
        
        # Test 2: Loss scenario for LONG position
        print("\n2. Testing LOSS scenario:")
        env.reset()
        print(f"Opening LONG at ${prices[0]:.2f}")
        state, reward, done, _ = env.step(1)
        print(f"Open position reward: {reward}")
        self.assertEqual(reward, 0, "Opening position should give 0 reward")
        
        env.current_step = 2
        print(f"Moving to price ${prices[2]:.2f} (loss: -$0.51)")
        state, reward, done, _ = env.step(0)
        print(f"Loss reward received: {reward}")
        self.assertEqual(reward, LOSS_PENALTY, f"Expected LOSS_PENALTY ({LOSS_PENALTY})")
        
        # Test 3: Timeout with small gain
        print("\n3. Testing TIMEOUT scenario:")
        env.reset()
        print(f"Opening LONG at ${prices[0]:.2f}")
        state, reward, done, _ = env.step(1)
        print(f"Open position reward: {reward}")
        self.assertEqual(reward, 0, "Opening position should give 0 reward")
        
        env.current_step = 3
        print(f"Starting timeout period at ${prices[3]:.2f} (small gain: +$0.10)")
        steps = 0
        final_reward = 0
        for _ in range(env.TRADE_DURATION):
            state, reward, done, _ = env.step(0)
            steps += 1
            if reward != 0:
                final_reward = reward
                print(f"Timeout occurred after {steps} steps")
                break
        
        expected_timeout_reward = 0.10 / 10
        print(f"Timeout reward received: {final_reward}")
        self.assertAlmostEqual(final_reward, expected_timeout_reward, places=2,
            msg=f"Expected timeout reward of {expected_timeout_reward:.2f}")
        
        print("\nAll reward tests passed successfully!")

if __name__ == '__main__':
    unittest.main() 