# CEF Ex-Dividend Shorting Strategy Backtester

A comprehensive backtesting engine for testing ex-dividend short strategies on Closed-End Funds (CEFs). Tests multiple entry points, holding periods, VIX regimes, and integrates with CEFConnect for fundamental factor analysis.

---

## Features

### ✅ Complete Backtest Engine
- **Multi-Entry Testing**: T-1 close, ex-dividend open, ex-dividend close
- **Multi-Period Hold**: 5, 10, 15, 20 days (configurable)
- **Cost Modeling**: 
  - Bid-ask spread (0.1% entry + exit)
  - Short borrow fees (tiered by liquidity: 0.1%-2% annual)
  - Scalable cost framework
- **Universe**: All CEFs with 2+ years ex-dividend history

### 📊 Diagnostic Analysis
- **Regime Analysis**: Performance by VIX bucket (<15, 15-20, >20)
- **Entry Point Analysis**: Compare entry timing strategies
- **Holding Period Optimization**: Find ideal exit window
- **Ticker Performance**: Top/bottom performing CEFs
- **Factor Attribution**: (CEFConnect integration ready)
  - Z-score impact on returns
  - Premium/discount correlation
  - Interaction effects

### 📈 Visualizations
- Interactive HTML dashboard with Plotly charts
- Win rate heatmaps (VIX × Holding Period, Entry × Regime, etc.)
- Equity curve and distribution plots
- Summary statistics tables

### 🔄 CEFConnect Integration
- **Scraper**: Fetch live premium/discount, Z-scores, NAV, yield
- **Caching**: Local SQLite/JSON cache of fundamentals
- **Interpolation**: Estimate historical fundamentals for analysis
- **Batch Operations**: Fetch data for multiple CEFs with rate limiting

### 💾 Export & Reporting
- CSV export of all trades (entry, exit, PnL, regime, costs)
- JSON export of summary statistics
- Formatted console reports
- Custom filtering and rule-based analysis

---

## Installation

### Requirements
- Python 3.8+
- pandas, numpy
- yfinance (Yahoo Finance data)
- requests, beautifulsoup4 (CEFConnect scraping)
- plotly (visualization)

### Setup

```bash
# Clone or download repository
cd cef_exdiv_backtest

# Install dependencies
pip install -r requirements.txt

# Run quickstart
python QUICKSTART_GUIDE.py
```

---

## Quick Start

See `QUICKSTART_GUIDE.py` for a runnable example.

### Basic Backtest

```python
from cef_exdiv_backtest_engine import ExDivBacktester, DiagnosticsAnalyzer, BacktestReporter

# Initialize backtester
backtester = ExDivBacktester(start_date="2022-01-01", end_date="2024-12-31")

# Run backtest (fetches all CEFs with ex-divs)
trades = backtester.run_backtest(
    entry_types=['t_minus_1_close', 'ex_div_open', 'ex_div_close'],
    holding_periods=[5, 10, 15, 20]
)

# Analyze
analyzer = DiagnosticsAnalyzer(trades)
stats = analyzer.summary_stats()
print(f"Win Rate: {stats['win_rate']:.1f}%")
```

### View Dashboard

Open `exdiv_backtest_dashboard.html` in your browser.
