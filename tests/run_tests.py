import unittest
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    # Run tests
    unittest.main(module='tests.test_stock_env') 