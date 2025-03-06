def print_results(results):
    """Print formatted trading results."""
    print("\n====== Training Summary ======")
    
    total_trades = (results['regular_wins'] + results['regular_losses'] + 
                   results['timeout_profits'] + results['timeout_losses'])
    
    if total_trades > 0:
        win_rate = ((results['regular_wins'] + results['timeout_profits']) / 
                   total_trades * 100)
        avg_pnl = results['total_pnl'] / total_trades
        
        print(f"Overall Win Rate: {win_rate:.1f}%")
        print(f"Total P&L: ${results['total_pnl']:.2f}")
        print(f"Average P&L per Trade: ${avg_pnl:.2f}")
        print(f"Regular Trades - Wins: {results['regular_wins']}, "
              f"Losses: {results['regular_losses']}")
        print(f"Timeout Trades - Profits: {results['timeout_profits']}, "
              f"Losses: {results['timeout_losses']}")
    
    print(f"Total Trades Attempted: {results['trades_attempted']}")
    print("=" * 30) 