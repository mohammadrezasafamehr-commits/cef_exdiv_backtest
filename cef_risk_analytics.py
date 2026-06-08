"""
CEF Ex-Dividend Strategy Risk Analytics Engine
=============================================
Implements risk measures, bootstrap Monte Carlo simulations, and Walk-Forward
validation testing for Closed-End Fund (CEF) ex-dividend strategy trades.

Risk Measures:
    - Max Drawdown
    - Sortino Ratio (annualized & return-by-return)
    - Value at Risk (95% & 99% VaR)
    - Conditional Value at Risk (95% & 99% CVaR)
    - Ulcer Index

Monte Carlo:
    - 10,000 bootstrap simulations of equity growth pathways.
    - Probability of ruin and terminal capital distributions.
    - Path drawdown statistics.

Walk-Forward Test:
    - Chronological sliding windows (e.g., 1-year training, 6-month testing).
    - Out-of-sample parameter execution and comparisons against baseline anchor.
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Set seed for reproducibility
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Risk Measures Calculations
# ──────────────────────────────────────────────────────────────────────────────

class RiskAnalyzer:
    """Calculates comprehensive risk measures from trade return series."""

    @staticmethod
    def calculate_drawdowns(returns_bps: pd.Series, initial_bps: float = 10000.0) -> Tuple[pd.Series, pd.Series, float, float]:
        """
        Calculate cumulative equity curve, peaks, drawdowns in bps, and drawdown percentage.
        Returns: (equity_bps, peaks_bps, drawdowns_bps, max_dd_bps, max_dd_pct)
        """
        if returns_bps.empty:
            return pd.Series(), pd.Series(), 0.0, 0.0
            
        equity = initial_bps + returns_bps.cumsum()
        peaks = equity.cummax()
        drawdowns_bps = equity - peaks
        
        # Percentage drawdown
        drawdowns_pct = (equity - peaks) / peaks * 100.0
        
        max_dd_bps = float(drawdowns_bps.min())
        max_dd_pct = float(drawdowns_pct.min())
        
        return equity, drawdowns_bps, max_dd_bps, max_dd_pct

    @staticmethod
    def sortino_ratio(returns_bps: pd.Series, target_bps: float = 0.0, annualization_factor: float = 252.0) -> float:
        """Calculate Sortino Ratio (excess return over target divided by downside semi-deviation)."""
        if returns_bps.empty:
            return 0.0
            
        excess_returns = returns_bps - target_bps
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float("inf")
            
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        avg_excess = excess_returns.mean()
        
        # Return-by-return Sortino
        r_sortino = avg_excess / downside_deviation
        
        # Annualized Sortino
        return float(r_sortino * np.sqrt(annualization_factor))

    @staticmethod
    def value_at_risk(returns_bps: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate historical Value at Risk (VaR) in basis points."""
        if returns_bps.empty:
            return 0.0
        alpha = 1.0 - confidence_level
        return float(np.percentile(returns_bps, alpha * 100))

    @staticmethod
    def conditional_value_at_risk(returns_bps: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate historical Conditional Value at Risk (CVaR) in basis points."""
        if returns_bps.empty:
            return 0.0
        var_limit = RiskAnalyzer.value_at_risk(returns_bps, confidence_level)
        cvar_returns = returns_bps[returns_bps <= var_limit]
        if len(cvar_returns) == 0:
            return var_limit
        return float(cvar_returns.mean())

    @staticmethod
    def ulcer_index(drawdowns_pct: pd.Series) -> float:
        """Calculate Ulcer Index based on percentage drawdown series."""
        if drawdowns_pct.empty:
            return 0.0
        return float(np.sqrt(np.mean(drawdowns_pct ** 2)))

    @staticmethod
    def calmar_ratio(returns_bps: pd.Series, max_dd_pct: float, annualization_factor: float = 252.0) -> float:
        """Calculate Calmar Ratio (annualized return percentage / max drawdown percentage)."""
        if returns_bps.empty or max_dd_pct == 0:
            return 0.0
        ann_return_pct = (returns_bps.mean() * annualization_factor) / 100.0
        return float(ann_return_pct / abs(max_dd_pct))

    @staticmethod
    def gain_to_pain_ratio(returns_bps: pd.Series) -> float:
        """Calculate Gain-to-Pain Ratio (sum of returns / absolute sum of negative returns)."""
        if returns_bps.empty:
            return 0.0
        losses = returns_bps[returns_bps < 0]
        if len(losses) == 0:
            return float("inf")
        return float(returns_bps.sum() / abs(losses.sum()))

    @staticmethod
    def omega_ratio(returns_bps: pd.Series, target_bps: float = 0.0) -> float:
        """Calculate Omega Ratio relative to target threshold (in bps)."""
        if returns_bps.empty:
            return 0.0
        gains = returns_bps[returns_bps > target_bps] - target_bps
        losses = target_bps - returns_bps[returns_bps < target_bps]
        if len(losses) == 0:
            return float("inf")
        return float(gains.sum() / losses.sum())

    @staticmethod
    def tail_ratio(returns_bps: pd.Series) -> float:
        """Calculate Tail Ratio (95th percentile return / |5th percentile return|)."""
        if returns_bps.empty:
            return 0.0
        p5 = np.percentile(returns_bps, 5)
        p95 = np.percentile(returns_bps, 95)
        if p5 == 0:
            return float("inf")
        return float(p95 / abs(p5))

    @classmethod
    def generate_risk_report(cls, returns_bps: pd.Series) -> Dict:
        """Generate a complete dictionary of risk metrics."""
        if returns_bps.empty:
            return {"status": "No trades to analyze"}
            
        equity, dd_bps, max_dd_bps, max_dd_pct = cls.calculate_drawdowns(returns_bps)
        dd_pct = (equity - equity.cummax()) / equity.cummax() * 100
        
        avg_return = returns_bps.mean()
        std_return = returns_bps.std()
        sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
        
        win_rate = (returns_bps > 0).mean() * 100
        
        winners = returns_bps[returns_bps > 0]
        losers = returns_bps[returns_bps <= 0]
        profit_factor = float(winners.sum() / abs(losers.sum())) if len(losers) > 0 and losers.sum() != 0 else (float("inf") if len(winners) > 0 else 0.0)
        win_loss_ratio = float(winners.mean() / abs(losers.mean())) if len(losers) > 0 and losers.mean() != 0 else (float("inf") if len(winners) > 0 else 0.0)
        
        skewness = returns_bps.skew()
        kurt = returns_bps.kurt()
        
        # Format skewness and kurtosis safely
        skewness_val = float(skewness) if pd.notna(skewness) else 0.0
        kurt_val = float(kurt) if pd.notna(kurt) else 0.0
        
        return {
            "total_trades": len(returns_bps),
            "win_rate": round(win_rate, 2),
            "total_return_bps": round(returns_bps.sum(), 2),
            "avg_return_bps": round(avg_return, 2),
            "std_return_bps": round(std_return, 2),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(cls.sortino_ratio(returns_bps), 4),
            "calmar_ratio": round(cls.calmar_ratio(returns_bps, max_dd_pct), 4) if not np.isinf(cls.calmar_ratio(returns_bps, max_dd_pct)) else float("inf"),
            "gain_to_pain_ratio": round(cls.gain_to_pain_ratio(returns_bps), 4) if not np.isinf(cls.gain_to_pain_ratio(returns_bps)) else float("inf"),
            "omega_ratio": round(cls.omega_ratio(returns_bps), 4) if not np.isinf(cls.omega_ratio(returns_bps)) else float("inf"),
            "tail_ratio": round(cls.tail_ratio(returns_bps), 4) if not np.isinf(cls.tail_ratio(returns_bps)) else float("inf"),
            "skewness": round(skewness_val, 4),
            "kurtosis": round(kurt_val, 4),
            "profit_factor": round(profit_factor, 4) if not np.isinf(profit_factor) else float("inf"),
            "win_loss_ratio": round(win_loss_ratio, 4) if not np.isinf(win_loss_ratio) else float("inf"),
            "max_drawdown_bps": round(max_dd_bps, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "var_95_bps": round(cls.value_at_risk(returns_bps, 0.95), 2),
            "var_99_bps": round(cls.value_at_risk(returns_bps, 0.99), 2),
            "cvar_95_bps": round(cls.conditional_value_at_risk(returns_bps, 0.95), 2),
            "cvar_99_bps": round(cls.conditional_value_at_risk(returns_bps, 0.99), 2),
            "ulcer_index": round(cls.ulcer_index(dd_pct), 2)
        }


# ──────────────────────────────────────────────────────────────────────────────
# Monte Carlo Simulation
# ──────────────────────────────────────────────────────────────────────────────

class MonteCarloSimulator:
    """Simulates trading equity curves via bootstrapping historical trade returns."""

    def __init__(self, returns_bps: pd.Series) -> None:
        self.returns = returns_bps.values
        self.n_trades = len(self.returns)

    def run_simulation(
        self, 
        n_paths: int = 10000, 
        initial_capital: float = 10000.0, 
        path_length: Optional[int] = None,
        ruin_threshold_pct: float = 30.0
    ) -> Dict:
        """
        Run bootstrap Monte Carlo simulation.
        Initial capital defaults to $10,000. Each step compounds path capital:
        capital_t = capital_{t-1} * (1 + return_bps / 10000)
        """
        if self.n_trades == 0:
            return {"error": "No trades to bootstrap"}

        # Use actual trade count if path length is not specified
        length = path_length or self.n_trades
        
        # Pre-allocate array for terminal capital and path maximum drawdowns
        terminal_capital = np.zeros(n_paths)
        max_drawdowns_pct = np.zeros(n_paths)
        ruined_paths_count = 0
        
        ruin_barrier = initial_capital * (1.0 - ruin_threshold_pct / 100.0)

        # Store a sample of 100 paths for plotting/visualization
        plot_paths = []
        n_sample_paths = min(100, n_paths)

        for path_idx in range(n_paths):
            # Bootstrap sample (sample with replacement)
            bootstrapped_returns_bps = np.random.choice(self.returns, size=length, replace=True)
            
            # Compute equity curve path
            # capital_t = capital_0 * cumprod(1 + returns_bps / 10000)
            multipliers = 1.0 + bootstrapped_returns_bps / 10000.0
            equity_curve = initial_capital * np.cumprod(multipliers)
            equity_curve = np.insert(equity_curve, 0, initial_capital)  # prepend starting capital
            
            terminal_capital[path_idx] = equity_curve[-1]
            
            # Calculate peak and drawdown series for this path
            peaks = np.maximum.accumulate(equity_curve)
            drawdowns = (equity_curve - peaks) / peaks * 100.0
            max_drawdowns_pct[path_idx] = np.min(drawdowns)
            
            # Check ruin condition (drawdown breaching threshold)
            if np.any(equity_curve <= ruin_barrier):
                ruined_paths_count += 1
                
            # Keep sample paths for reporting
            if path_idx < n_sample_paths:
                plot_paths.append(equity_curve.tolist())

        # Quantiles calculations
        quantiles = [5, 25, 50, 75, 95]
        terminal_quantiles = {f"q{q}": float(np.percentile(terminal_capital, q)) for q in quantiles}
        drawdown_quantiles = {f"q{q}": float(np.percentile(max_drawdowns_pct, q)) for q in quantiles}

        prob_ruin = (ruined_paths_count / n_paths) * 100.0
        
        return {
            "initial_capital": initial_capital,
            "path_length": length,
            "n_paths": n_paths,
            "probability_of_ruin_pct": round(prob_ruin, 2),
            "ruin_threshold_pct": ruin_threshold_pct,
            "avg_terminal_capital": round(float(np.mean(terminal_capital)), 2),
            "median_terminal_capital": round(float(np.median(terminal_capital)), 2),
            "terminal_capital_quantiles": {k: round(v, 2) for k, v in terminal_quantiles.items()},
            "avg_max_drawdown_pct": round(float(np.mean(max_drawdowns_pct)), 2),
            "max_drawdown_quantiles": {k: round(v, 2) for k, v in drawdown_quantiles.items()},
            "sample_paths": plot_paths
        }


# ──────────────────────────────────────────────────────────────────────────────
# Walk-Forward Validation Engine
# ──────────────────────────────────────────────────────────────────────────────

class WalkForwardValidator:
    """Implements sliding-window Walk-Forward optimization of parameters."""

    def __init__(self, trades_df: pd.DataFrame) -> None:
        self.df = trades_df.copy()
        # Convert date to datetime
        self.df["entry_date"] = pd.to_datetime(self.df["entry_date"])
        # Ensure it is sorted
        self.df = self.df.sort_values("entry_date").reset_index(drop=True)

    def run_walk_forward(
        self,
        train_duration_days: int = 365,
        test_duration_days: int = 180,
        step_duration_days: int = 180,
        metric: str = "sharpe_ratio"
    ) -> Dict:
        """
        Executes Walk-Forward validation.
        Sliding windows: Train on train_duration, select best (entry_type, holding_days) pair.
        Apply best parameters to subsequent test_duration out-of-sample window.
        """
        if self.df.empty:
            return {"error": "No trades to run walk-forward validation"}

        start_date = self.df["entry_date"].min()
        end_date = self.df["entry_date"].max()
        total_days = (end_date - start_date).days
        
        # Dynamically scale windows if total duration is too short for defaults
        if total_days < train_duration_days + test_duration_days:
            train_duration_days = max(30, int(total_days * 0.5))
            test_duration_days = max(15, int(total_days * 0.25))
            step_duration_days = test_duration_days
            print(f"[INFO] Short dataset ({total_days} days). Auto-scaled Walk-Forward windows: Train={train_duration_days} days, Test={test_duration_days} days.")
            
        current_train_start = start_date
        oos_trades_list = []
        window_logs = []
        
        # Grid of optimization parameters
        entry_types = self.df["entry_type"].unique()
        holding_periods = self.df["holding_days"].unique()
        
        window_idx = 1
        
        while True:
            current_train_end = current_train_start + timedelta(days=train_duration_days)
            current_test_end = current_train_end + timedelta(days=test_duration_days)
            
            if current_train_end >= end_date:
                break
                
            # Cap test window at terminal date
            if current_test_end > end_date:
                current_test_end = end_date
                
            # Filter in-sample training trades
            train_mask = (self.df["entry_date"] >= current_train_start) & (self.df["entry_date"] < current_train_end)
            train_trades = self.df[train_mask]
            
            # Filter out-of-sample test trades
            test_mask = (self.df["entry_date"] >= current_train_end) & (self.df["entry_date"] <= current_test_end)
            test_trades = self.df[test_mask]
            
            if train_trades.empty or test_trades.empty:
                # Slide window
                current_train_start += timedelta(days=step_duration_days)
                continue
                
            # Find optimal parameters in-sample
            best_params = None
            best_score = -float("inf")
            
            for etype in entry_types:
                for hdays in holding_periods:
                    param_trades = train_trades[(train_trades["entry_type"] == etype) & (train_trades["holding_days"] == hdays)]
                    if param_trades.empty:
                        continue
                        
                    # Calculate metric score
                    returns = param_trades["net_return_bps"]
                    if metric == "sharpe_ratio":
                        score = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else -100.0
                    else:  # default to total return
                        score = returns.sum()
                        
                    if score > best_score:
                        best_score = score
                        best_params = (etype, hdays)
            
            if best_params is not None:
                opt_etype, opt_hdays = best_params
                # Apply optimal parameters out-of-sample
                oos_selected_mask = (test_trades["entry_type"] == opt_etype) & (test_trades["holding_days"] == opt_hdays)
                oos_window_trades = test_trades[oos_selected_mask]
                
                if not oos_window_trades.empty:
                    oos_trades_list.append(oos_window_trades)
                    
                window_logs.append({
                    "window": window_idx,
                    "train_period": f"{current_train_start.strftime('%Y-%m-%d')} to {current_train_end.strftime('%Y-%m-%d')}",
                    "test_period": f"{current_train_end.strftime('%Y-%m-%d')} to {current_test_end.strftime('%Y-%m-%d')}",
                    "opt_entry_type": opt_etype,
                    "opt_holding_days": int(opt_hdays),
                    "train_score": round(best_score, 4),
                    "n_test_trades": len(oos_window_trades),
                    "test_return_bps": round(oos_window_trades["net_return_bps"].sum(), 2) if not oos_window_trades.empty else 0.0
                })
                
                window_idx += 1
            
            # Slide window forward
            current_train_start += timedelta(days=step_duration_days)

        # Aggregate out-of-sample trades
        if not oos_trades_list:
            return {"error": "Walk-forward validation generated no out-of-sample trades."}
            
        oos_df = pd.concat(oos_trades_list).drop_duplicates().sort_values("entry_date").reset_index(drop=True)
        
        # Calculate OOS performance
        oos_analyzer = RiskAnalyzer.generate_risk_report(oos_df["net_return_bps"])
        
        # Constant Anchor Benchmark (Always use T-1 close + 5 days hold)
        anchor_df = self.df[(self.df["entry_type"] == "t_minus_1_close") & (self.df["holding_days"] == 5)]
        # Filter anchor to match the out-of-sample period (after the first training window)
        first_train_end = start_date + timedelta(days=train_duration_days)
        anchor_filtered = anchor_df[anchor_df["entry_date"] >= first_train_end]
        anchor_analyzer = RiskAnalyzer.generate_risk_report(anchor_filtered["net_return_bps"])
        
        return {
            "window_details": window_logs,
            "out_of_sample_metrics": oos_analyzer,
            "anchor_benchmark_metrics": anchor_analyzer,
            "out_of_sample_trades_count": len(oos_df),
            "anchor_trades_count": len(anchor_filtered)
        }


# ──────────────────────────────────────────────────────────────────────────────
# Interactive Trade EV Calculator
# ──────────────────────────────────────────────────────────────────────────────

def interactive_calculator(df: pd.DataFrame):
    """
    Launches an interactive console utility to calculate expected profit/loss percentage,
    win rates, and sample size for specific trade entries under historical conditions.
    """
    print("=" * 70)
    print("      CEF EX-DIVIDEND STRATEGY - TRADE EV CALCULATOR")
    print("=" * 70)
    
    tickers = sorted(df["ticker"].dropna().unique().tolist())
    
    while True:
        print(f"\nAvailable Tickers in dataset: {', '.join(tickers)}")
        ticker_input = input("Enter Ticker (or 'exit' to quit): ").strip().upper()
        if ticker_input.lower() == 'exit':
            print("Exiting Trade EV Calculator. Thank you!")
            break
            
        if ticker_input not in tickers:
            print(f"[ERROR] Ticker '{ticker_input}' not found in dataset. Please choose from: {', '.join(tickers)}")
            continue
            
        # Select VIX Regime
        print("\nSelect VIX Regime:")
        print("  [1] Calm (<15)")
        print("  [2] Normal (15-20)")
        print("  [3] Stressed (>20)")
        print("  [4] Any Regime")
        regime_choice = input("Enter choice [1-4] (default Any): ").strip()
        
        regime_map = {"1": "Calm (<15)", "2": "Normal (15-20)", "3": "Stressed (>20)"}
        regime_filter = regime_map.get(regime_choice, None)
        
        # Select Entry Type (Shorting time)
        print("\nSelect Entry Point (Shorting time):")
        print("  [1] t_minus_1_close (Short 1 day before ex-div at close)")
        print("  [2] ex_div_open (Short on ex-div day at open)")
        print("  [3] ex_div_close (Short on ex-div day at close)")
        print("  [4] Any Entry")
        entry_choice = input("Enter choice [1-4] (default Any): ").strip()
        
        entry_map = {"1": "t_minus_1_close", "2": "ex_div_open", "3": "ex_div_close"}
        entry_filter = entry_map.get(entry_choice, None)
        
        # Select Holding Days
        print("\nSelect Holding Period:")
        print("  [1] 5 days")
        print("  [2] 10 days")
        print("  [3] 15 days")
        print("  [4] 20 days")
        print("  [5] Any Hold Period")
        hold_choice = input("Enter choice [1-5] (default Any): ").strip()
        
        hold_map = {"1": 5, "2": 10, "3": 15, "4": 20}
        hold_filter = hold_map.get(hold_choice, None)
        
        # Select Date Range
        print("\nEnter custom Date Range (or leave empty for all history):")
        start_date_input = input("  Start Date (YYYY-MM-DD, e.g. 2023-01-01): ").strip()
        end_date_input = input("  End Date (YYYY-MM-DD, e.g. 2023-12-31): ").strip()
        
        # Filter trade data
        filtered = df[df["ticker"].str.upper() == ticker_input].copy()
        scenario_desc = f"Ticker: {ticker_input}"
        
        # Apply Date Range filters
        if start_date_input:
            try:
                pd.to_datetime(start_date_input)
                filtered = filtered[filtered["entry_date"] >= start_date_input]
                scenario_desc += f" | From: {start_date_input}"
            except Exception:
                print(f"[WARNING] Invalid Start Date format: '{start_date_input}'. Ignoring Start Date.")
        
        if end_date_input:
            try:
                pd.to_datetime(end_date_input)
                filtered = filtered[filtered["entry_date"] <= end_date_input]
                scenario_desc += f" | To: {end_date_input}"
            except Exception:
                print(f"[WARNING] Invalid End Date format: '{end_date_input}'. Ignoring End Date.")
        
        if regime_filter:
            filtered = filtered[filtered["regime"] == regime_filter]
            scenario_desc += f" | Regime: {regime_filter}"
        else:
            scenario_desc += " | Regime: Any"
            
        if entry_filter:
            filtered = filtered[filtered["entry_type"] == entry_filter]
            scenario_desc += f" | Entry: {entry_filter}"
        else:
            scenario_desc += " | Entry: Any"
            
        if hold_filter:
            filtered = filtered[filtered["holding_days"] == hold_filter]
            scenario_desc += f" | Hold: {hold_filter}d"
        else:
            scenario_desc += " | Hold: Any"
            
        print("\n" + "-" * 70)
        print(f"SCENARIO: {scenario_desc}")
        print("-" * 70)
        
        n_trades = len(filtered)
        if n_trades == 0:
            print("[WARNING] No past historical trades found matching this exact scenario.")
            print("Please try broader criteria (e.g., 'Any Regime' or 'Any Entry').")
            continue
            
        # Calculate statistics in percentage
        returns_bps = filtered["net_return_bps"]
        avg_ret_pct = returns_bps.mean() / 100.0
        total_ret_pct = returns_bps.sum() / 100.0
        win_rate = (filtered["won"] == True).mean() * 100.0
        
        std_ret_pct = (returns_bps.std() / 100.0) if n_trades > 1 else 0.0
        max_ret_pct = returns_bps.max() / 100.0
        min_ret_pct = returns_bps.min() / 100.0
        
        # Scenario Sharpe
        scenario_sharpe = 0.0
        if std_ret_pct > 0:
            days = hold_filter or int(filtered["holding_days"].mean()) or 5
            scenario_sharpe = (avg_ret_pct / std_ret_pct) * np.sqrt(252 / days)
            
        print(f"  Historic Trades Found: {n_trades} occurrences")
        
        # Historical Premium/Discount and Z-Score calculation
        has_prem = "entry_premium_discount" in filtered.columns and not filtered["entry_premium_discount"].dropna().empty
        has_z = "entry_zscore" in filtered.columns and not filtered["entry_zscore"].dropna().empty
        avg_prem = filtered["entry_premium_discount"].dropna().mean() if has_prem else None
        avg_z = filtered["entry_zscore"].dropna().mean() if has_z else None
        
        # Print expected EV with colors/symbols in safe cp1252 style
        status_prefix = "[EXPECTED GAIN]" if avg_ret_pct >= 0 else "[EXPECTED LOSS]"
        print(f"  {status_prefix} Expected Profit/Loss: {avg_ret_pct:+.3f}% ({avg_ret_pct * 100.0:+.1f} bps)")
        print(f"  Historical Win Rate:   {win_rate:.1f}%")
        print(f"  Total Cum. Return:     {total_ret_pct:+.2f}% ({returns_bps.sum():+,.1f} bps)")
        
        if n_trades > 1:
            print(f"  Trade Volatility (Std): {std_ret_pct:.2f}%")
            print(f"  Scenario Sharpe Ratio:  {scenario_sharpe:.2f}")
            
        # Display Historical Valuation Levels
        prem_str = f"{avg_prem:+.2f}%" if avg_prem is not None else "N/A"
        z_str = f"{avg_z:+.2f}" if avg_z is not None else "N/A"
        print(f"  Historic Avg Valuation: Premium/Discount: {prem_str} | Z-Score: {z_str}")
        
        print(f"  Best Past Trade:       {max_ret_pct:+.2f}%")
        print(f"  Worst Past Trade:      {min_ret_pct:+.2f}%")
        print("-" * 70)
        
        repeat = input("\nDo you want to run another calculation? (y/n, default y): ").strip().lower()
        if repeat == 'n':
            print("Exiting Trade EV Calculator. Thank you!")
            break


# ──────────────────────────────────────────────────────────────────────────────
# Main CLI Execution
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CEF Ex-Dividend Strategy Risk & Validation Engine")
    parser.add_argument("--csv", default="demo_trades.csv", help="Input trades CSV filepath")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial Monte Carlo capital")
    parser.add_argument("--paths", type=int, default=10000, help="Number of Monte Carlo paths")
    parser.add_argument("--ruin", type=float, default=30.0, help="Monte Carlo ruin drawdown threshold (%)")
    parser.add_argument("--train-days", type=int, default=365, help="Walk-forward training window (days)")
    parser.add_argument("--test-days", type=int, default=180, help="Walk-forward testing window (days)")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate PNG visual plots if matplotlib available")
    parser.add_argument("--ticker", default=None, help="Filter trades for a specific ticker (default: all tickers combined)")
    parser.add_argument("--calculator", action="store_true", help="Launch the interactive Trade Expected Value (EV) Calculator")
    args = parser.parse_args()

    print("=" * 70)
    print("      CEF EX-DIVIDEND STRATEGY - RISK & VALIDATION ENGINE")
    print("=" * 70)

    # 1. Load data
    if not os.path.exists(args.csv):
        print(f"[ERROR] Trades CSV file not found at: {args.csv}")
        print("   Please run a backtest first to generate trade results.")
        return

    print(f"Loading trade data from: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"[SUCCESS] Loaded {len(df):,} total trade records.")

    if df.empty or "net_return_bps" not in df.columns:
        print("[ERROR] The trade file is empty or does not contain 'net_return_bps' column.")
        return

    # Convert date to datetime and sort chronologically
    if "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df = df.sort_values("entry_date").reset_index(drop=True)

    # Launch Trade EV Calculator if requested
    if args.calculator:
        interactive_calculator(df)
        return

    # Optional Ticker Filtering
    if args.ticker:
        ticker_upper = args.ticker.upper()
        df = df[df["ticker"].str.upper() == ticker_upper].copy()
        if df.empty:
            print(f"[ERROR] No trades found for ticker '{args.ticker}' in {args.csv}.")
            return
        print(f"[FILTER] Analyzing trades for ticker: {ticker_upper} ({len(df):,} trade records)")
    else:
        tickers_list = sorted(df["ticker"].dropna().unique().tolist())
        print(f"[PORTFOLIO] Analyzing combined portfolio of all tickers: {tickers_list}")

    # Extract unique return series
    # In the main engine, multiple entries/holds are saved in demo_trades.csv
    # For global risk, let's analyze the baseline strategy trades first (always T-1 entry, 5d hold)
    baseline_df = df[(df["entry_type"] == "t_minus_1_close") & (df["holding_days"] == 5)].copy()
    if baseline_df.empty:
        baseline_df = df.copy()
        print("[WARNING] No baseline 't_minus_1_close' + 5d trades found. Analyzing raw trade records.")
    else:
        print(f"Extracted baseline strategy trades (T-1 Close entry, 5d Hold): {len(baseline_df):,} trades.")

    returns = baseline_df["net_return_bps"].dropna()

    # 2. Execute Risk Measures
    print("\n[1/3] Calculating Strategy Risk Measures...")
    risk_report = RiskAnalyzer.generate_risk_report(returns)
    
    print("-" * 50)
    print(f"  Total Strategy Trades: {risk_report['total_trades']:,}")
    print(f"  Win Rate:              {risk_report['win_rate']}%")
    print(f"  Total Return:          {risk_report['total_return_bps']:+,.1f} bps")
    print(f"  Average Return:        {risk_report['avg_return_bps']:+.2f} bps")
    print(f"  Volatility (Std):      {risk_report['std_return_bps']:.2f} bps")
    print(f"  Sharpe Ratio:          {risk_report['sharpe_ratio']:.4f}")
    print(f"  Sortino Ratio:         {risk_report['sortino_ratio']:.4f}")
    print(f"  Calmar Ratio:          {risk_report.get('calmar_ratio', 0.0):.4f}")
    print(f"  Gain-to-Pain Ratio:    {risk_report.get('gain_to_pain_ratio', 0.0):.4f}")
    print(f"  Omega Ratio:           {risk_report.get('omega_ratio', 0.0):.4f}")
    print(f"  Tail Ratio (95/5):     {risk_report.get('tail_ratio', 0.0):.4f}")
    print(f"  Skewness:              {risk_report.get('skewness', 0.0):+.4f}")
    print(f"  Kurtosis:              {risk_report.get('kurtosis', 0.0):+.4f}")
    print(f"  Profit Factor:         {risk_report.get('profit_factor', 0.0):.4f}")
    print(f"  Win/Loss Ratio:        {risk_report.get('win_loss_ratio', 0.0):.4f}")
    print(f"  Maximum Drawdown:      {risk_report['max_drawdown_pct']:.2f}% ({risk_report['max_drawdown_bps']:+,.1f} bps)")
    print(f"  Ulcer Index:           {risk_report['ulcer_index']:.2f}")
    print(f"  Value at Risk (95%):   {risk_report['var_95_bps']:+,.1f} bps")
    print(f"  Value at Risk (99%):   {risk_report['var_99_bps']:+,.1f} bps")
    print(f"  Conditional VaR (95%): {risk_report['cvar_95_bps']:+,.1f} bps")
    print(f"  Conditional VaR (99%): {risk_report['cvar_99_bps']:+,.1f} bps")
    print("-" * 50)

    # 3. Execute Monte Carlo Simulation
    print(f"\n[2/3] Running Monte Carlo Path Simulations ({args.paths:,} paths)...")
    mc = MonteCarloSimulator(returns)
    mc_report = mc.run_simulation(
        n_paths=args.paths,
        initial_capital=args.capital,
        ruin_threshold_pct=args.ruin
    )
    
    print("-" * 50)
    print(f"  Starting Capital:      ${mc_report['initial_capital']:,.2f}")
    print(f"  Simulation Path Length: {mc_report['path_length']:,} trades")
    print(f"  Probability of Ruin:   {mc_report['probability_of_ruin_pct']}% (breaching {mc_report['ruin_threshold_pct']}% Drawdown)")
    print(f"  Average Final Capital: ${mc_report['avg_terminal_capital']:,.2f}")
    print(f"  Median Final Capital:  ${mc_report['median_terminal_capital']:,.2f}")
    print("  Terminal Capital Quantiles:")
    for k, v in mc_report["terminal_capital_quantiles"].items():
        print(f"    - {k[1:]}th Percentile:  ${v:,.2f}")
    print(f"  Average Max Drawdown:  {mc_report['avg_max_drawdown_pct']:.2f}%")
    print("  Max Path Drawdown Quantiles:")
    for k, v in mc_report["max_drawdown_quantiles"].items():
        print(f"    - {k[1:]}th Percentile:  {v:.2f}%")
    print("-" * 50)

    # 4. Execute Walk-Forward Validation
    print("\n[3/4] Running chronological sliding-window Walk-Forward optimization...")
    wfv = WalkForwardValidator(df)
    wf_report = wfv.run_walk_forward(
        train_duration_days=args.train_days,
        test_duration_days=args.test_days,
        step_duration_days=args.test_days
    )
    
    if "error" in wf_report:
        print(f"[WARNING] Walk-Forward Error: {wf_report['error']}")
    else:
        print("\n  WALK-FORWARD SLIDING WINDOW DETAILS")
        print("  " + "-" * 110)
        print(f"  {'Win':<3} {'Training Period':<24} {'Testing Period':<24} {'Opt Entry':<18} {'Opt Hold':<8} {'OOS Return':>12}")
        print("  " + "-" * 110)
        for w in wf_report["window_details"]:
            print(f"  {w['window']:<3} {w['train_period']:<24} {w['test_period']:<24} {w['opt_entry_type']:<18} {int(w['opt_holding_days']):>3}d     {w['test_return_bps']:>+9.1f} bps")
        print("  " + "-" * 110)
        
        oos = wf_report["out_of_sample_metrics"]
        bench = wf_report["anchor_benchmark_metrics"]
        
        print("\n  WALK-FORWARD OUT-OF-SAMPLE VS. ANCHOR BENCHMARK SUMMARY")
        print("  " + "=" * 85)
        print(f"  {'Risk Metric':<28} | {'Walk-Forward OOS (Optimized)':<28} | {'Anchor Baseline (Constant)':<24}")
        print("  " + "-" * 85)
        print(f"  {'Total Out-of-Sample Trades':<28} | {wf_report['out_of_sample_trades_count']:<28} | {wf_report['anchor_trades_count']:<24}")
        print(f"  {'Total Cumulative Return':<28} | {oos.get('total_return_bps', 0.0):>+23.1f} bps | {bench.get('total_return_bps', 0.0):>+19.1f} bps")
        print(f"  {'Win Rate':<28} | {oos.get('win_rate', 0.0):>26.1f}% | {bench.get('win_rate', 0.0):>22.1f}%")
        print(f"  {'Sharpe Ratio':<28} | {oos.get('sharpe_ratio', 0.0):>27.4f} | {bench.get('sharpe_ratio', 0.0):>23.4f}")
        print(f"  {'Sortino Ratio':<28} | {oos.get('sortino_ratio', 0.0):>27.4f} | {bench.get('sortino_ratio', 0.0):>23.4f}")
        print(f"  {'Calmar Ratio':<28} | {oos.get('calmar_ratio', 0.0):>27.4f} | {bench.get('calmar_ratio', 0.0):>23.4f}")
        print(f"  {'Gain-to-Pain Ratio':<28} | {oos.get('gain_to_pain_ratio', 0.0):>27.4f} | {bench.get('gain_to_pain_ratio', 0.0):>23.4f}")
        print(f"  {'Omega Ratio (0 bps)':<28} | {oos.get('omega_ratio', 0.0):>27.4f} | {bench.get('omega_ratio', 0.0):>23.4f}")
        print(f"  {'Tail Ratio (95/5)':<28} | {oos.get('tail_ratio', 0.0):>27.4f} | {bench.get('tail_ratio', 0.0):>23.4f}")
        print(f"  {'Skewness':<28} | {oos.get('skewness', 0.0):>+27.4f} | {bench.get('skewness', 0.0):>+23.4f}")
        print(f"  {'Kurtosis':<28} | {oos.get('kurtosis', 0.0):>+27.4f} | {bench.get('kurtosis', 0.0):>+23.4f}")
        print(f"  {'Profit Factor':<28} | {oos.get('profit_factor', 0.0):>27.4f} | {bench.get('profit_factor', 0.0):>23.4f}")
        print(f"  {'Maximum Drawdown':<28} | {oos.get('max_drawdown_pct', 0.0):>26.2f}% | {bench.get('max_drawdown_pct', 0.0):>22.2f}%")
        print(f"  {'Ulcer Index':<28} | {oos.get('ulcer_index', 0.0):>27.2f} | {bench.get('ulcer_index', 0.0):>23.2f}")
        print("  " + "=" * 85)

    # 5. Multi-Dimensional Trade Performance Breakdown
    print("\n[4/4] Multi-Dimensional Performance Breakdown: VIX Regime x Entry Type x Holding Days")
    print("      Analyzing relationship between VIX level, entry timing, and holding period:")
    required_cols = ["regime", "entry_type", "holding_days", "net_return_bps", "won"]
    if all(c in df.columns for c in required_cols):
        # Group and calculate metrics
        grouped = df.groupby(["regime", "entry_type", "holding_days"]).agg(
            n_trades=("net_return_bps", "count"),
            win_rate=("won", lambda x: float((x == True).mean() * 100)),
            avg_return=("net_return_bps", "mean"),
            total_return=("net_return_bps", "sum"),
            std_return=("net_return_bps", "std")
        ).reset_index()
        
        # Sort values: Calm -> Normal -> Stressed
        regime_order = {"Calm (<15)": 0, "Normal (15-20)": 1, "Stressed (>20)": 2}
        grouped["regime_sort"] = grouped["regime"].map(regime_order).fillna(3)
        grouped = grouped.sort_values(by=["regime_sort", "entry_type", "holding_days"]).reset_index(drop=True)
        
        print("  " + "=" * 116)
        print(f"  {'VIX Regime':<16} | {'Entry Type (Shorting)':<20} | {'Hold':<5} | {'Trades':>6} | {'Win%':>7} | {'Avg Ret':>9} | {'Total Ret':>10} | {'Sharpe':>7}")
        print("  " + "-" * 116)
        
        for _, r in grouped.iterrows():
            avg = r["avg_return"]
            std = r["std_return"]
            sharpe_str = "-"
            if pd.notna(std) and std > 0:
                ann_sharpe = (avg / std) * np.sqrt(252 / r["holding_days"])
                sharpe_str = f"{ann_sharpe:.2f}"
            
            print(
                f"  {r['regime']:<16} | "
                f"{r['entry_type']:<20} | "
                f"{int(r['holding_days']):>3}d | "
                f"{r['n_trades']:>6,} | "
                f"{r['win_rate']:>6.1f}% | "
                f"{r['avg_return']:>+8.1f} | "
                f"{r['total_return']:>+9.1f} | "
                f"{sharpe_str:>7}"
            )
        print("  " + "=" * 116)
        print("  Note: Sharpe ratio above is annualized based on holding period frequency: (Avg/Std) * sqrt(252/HoldDays).\n")
    else:
        print("[WARNING] Could not run multidimensional analysis. Missing required columns in trade data.")

    # 5. Generate beautiful visualization plots if matplotlib is available
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="darkgrid")
            
            # Setup output plots folder inside the workspace
            plots_dir = "./risk_analytics_plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot 1: Monte Carlo Simulation paths
            fig, ax = plt.subplots(figsize=(10, 6))
            sample_paths = np.array(mc_report["sample_paths"])
            steps = np.arange(sample_paths.shape[1])
            
            for path in sample_paths:
                # Color paths green if they win, red if they drop, grey otherwise
                if path[-1] > args.capital * 1.2:
                    ax.plot(steps, path, color="green", alpha=0.1)
                elif path[-1] < args.capital * 0.8:
                    ax.plot(steps, path, color="red", alpha=0.15)
                else:
                    ax.plot(steps, path, color="grey", alpha=0.08)
                    
            # Highlight median/mean path
            ax.plot(steps, np.mean(sample_paths, axis=0), color="cyan", linewidth=2.5, label="Mean Path")
            ax.axhline(args.capital, color="white", linestyle="--", alpha=0.6, label="Starting Capital")
            ax.axhline(args.capital * (1.0 - args.ruin / 100.0), color="magenta", linestyle=":", linewidth=2, label="Ruin Barrier")
            
            ax.set_title(f"Monte Carlo Equity Growth Paths - Bootstrap (100 Sample Paths shown)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Sequential Trades Index", fontsize=11)
            ax.set_ylabel("Portfolio Capital ($)", fontsize=11)
            ax.legend(facecolor="#141c32", edgecolor="grey")
            
            mc_plot_path = os.path.join(plots_dir, "monte_carlo_paths.png")
            plt.savefig(mc_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            # Plot 2: Terminal Capital Distribution
            fig, ax = plt.subplots(figsize=(10, 5))
            # Run a quick high-sample check for plotting terminal distribution hist
            boot_caps = []
            for _ in range(2000):
                boot_caps.append(args.capital * np.prod(1.0 + np.random.choice(returns, size=len(returns), replace=True) / 10000.0))
            
            sns.histplot(boot_caps, bins=40, kde=True, color="#7c3aed", ax=ax, edgecolor="#0a0e17")
            ax.axvline(args.capital, color="red", linestyle="--", linewidth=1.5, label="Starting Capital")
            ax.axvline(np.mean(boot_caps), color="cyan", linestyle="-", linewidth=2, label="Mean Capital")
            ax.set_title("Distribution of Monte Carlo Final Equity Capital", fontsize=13, fontweight="bold")
            ax.set_xlabel("Final Capital ($)", fontsize=11)
            ax.set_ylabel("Frequency Path Density", fontsize=11)
            ax.legend()
            
            dist_plot_path = os.path.join(plots_dir, "terminal_capital_distribution.png")
            plt.savefig(dist_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            print(f"\nGenerated and saved beautiful visual plots inside workspace directory:")
            print(f"   Directory: {plots_dir}/")
            print(f"   Plot 1:    {plots_dir}/monte_carlo_paths.png")
            print(f"   Plot 2:    {plots_dir}/terminal_capital_distribution.png\n")
            
        except ImportError:
            print("\nTip: Install 'matplotlib' and 'seaborn' to generate gorgeous graphical plots of your risk paths.")
            print("   Command: pip install matplotlib seaborn\n")
        except Exception as e:
            print(f"\n[WARNING] Could not generate visual figures due to: {e}\n")

if __name__ == "__main__":
    main()
