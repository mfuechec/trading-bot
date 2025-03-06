"""
Feature flags to control debugging output in the trading system.
"""

class FeatureFlags:
    # Debug Settings
    DEBUG_REWARDS = False                   # Print detailed reward calculations
    DEBUG_POSITIONS = False                 # Print detailed position information
    DEBUG_TRADES = False                    # Print trade entry/exit information
    DEBUG_STEPS = False                     # Print step-by-step environment information
    DEBUG_TRAINING = False                  # Print training progress and metrics
    DEBUG_POSITION_REMAINS_OPEN = False     # Print when a position remains open
    ACTION_NOT_ALIGNED_WITH_TREND = False   # Print when an action is not aligned with the trend

    @classmethod
    def print_status(cls):
        """Print the current status of all debug flags"""
        print("\nDebug Flags Status:")
        print("-----------------")
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                value = getattr(cls, attr)
                print(f"{attr}: {value}") 