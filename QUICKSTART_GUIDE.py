"""
Quickstart Guide for CEF Ex-Dividend Backtest Engine
"""

from cef_exdiv_backtest_engine import ExDivBacktester, DiagnosticsAnalyzer, BacktestReporter
from cef_connect_scraper import CEFConnectScraper
import glob
import os

def run_quickstart():
    print("="*60)
    print(" CEF EX-DIVIDEND STRATEGY BACKTESTER - QUICKSTART")
    print("="*60)
    
    # Use absolute paths based on the script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "cef_data_cache")
    
    # Setup cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Clear stale cache files to ensure fresh data is always downloaded
    stale_files = glob.glob(os.path.join(cache_dir, "*.pkl"))
    if stale_files:
        print(f"\n[CACHE] Clearing {len(stale_files)} stale cache files to fetch latest data...")
        for f in stale_files:
            try:
                os.remove(f)
            except OSError:
                pass
    
    # 1. Initialize Backtester
    from datetime import datetime
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"\n[1/5] Initializing Backtester (2023-01-01 to {today_str})...")
    backtester = ExDivBacktester(start_date="2023-01-01", end_date=today_str)
    # Force the cache directory to be absolute as well
    backtester.fetcher.cache_dir = cache_dir
    
    # Attach the CEFConnect scraper for premium/discount and Z-score data
    scraper = CEFConnectScraper(cache_dir=cache_dir)
    backtester.set_cefconnect_scraper(scraper)

    # 2. Run Backtest on a subset of highly liquid CEFs
    test_tickers = ['PDI', 'GOF', 'PTY', 'UTF', 'CSQ']
    print(f"\n[2/5] Running backtest on subset: {test_tickers}")
    print(f"       (5-second cooldown between tickers to avoid rate limits)")
    
    trades = backtester.run_backtest(
        tickers=test_tickers,
        entry_types=['t_minus_1_close', 'ex_div_open', 'ex_div_close'],
        holding_periods=[5, 10, 15, 20]
    )
    
    print(f"Generated {len(trades)} potential trades.")
    
    # 3. Analyze Results
    print("\n[3/5] Analyzing Results...")
    analyzer = DiagnosticsAnalyzer(trades)
    stats = analyzer.summary_stats()
    
    # Print reports
    BacktestReporter.print_summary(stats)
    BacktestReporter.print_regime_report(analyzer.regime_analysis())
    BacktestReporter.print_entry_report(analyzer.entry_point_analysis())
    BacktestReporter.print_holding_report(analyzer.holding_period_analysis())
    
    # 4. Generate Interactive Dashboard
    print("\n[4/5] Generating Interactive Dashboard Data...")
    dashboard_path = os.path.join(base_dir, "exdiv_backtest_dashboard.html")
    BacktestReporter.generate_dashboard_data(trades, analyzer, dashboard_path)
    
    # 5. Export to CSV
    print("\n[5/5] Exporting Trades to CSV...")
    csv_path = os.path.join(base_dir, "demo_trades.csv")
    BacktestReporter.export_trades_csv(trades, csv_path)
    
    print("\n[SUCCESS] Quickstart Complete!")
    print(f"   - Check '{csv_path}' for trade details.")
    print(f"   - Open 'exdiv_backtest_dashboard.html' in your browser to view the interactive dashboard.")

if __name__ == "__main__":
    run_quickstart()

