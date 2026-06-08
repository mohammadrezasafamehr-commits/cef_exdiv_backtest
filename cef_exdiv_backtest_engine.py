"""
CEF Ex-Dividend Shorting Strategy Backtest Engine
=================================================
Comprehensive backtester for ex-dividend short strategies on Closed-End Funds.
Tests multiple entry points, holding periods, VIX regimes, and optionally
integrates CEFConnect fundamentals for factor analysis.

Classes:
    ExDivTrade       – Dataclass holding a single trade result
    DataFetcher      – Yahoo Finance data retrieval (OHLCV, dividends, VIX)
    CostModel        – Spread and borrow-fee calculations
    ExDivBacktester  – Main backtest loop
    DiagnosticsAnalyzer – Regime, entry, holding-period, and ticker analytics
    BacktestReporter – Console, CSV, JSON, and dashboard-data export
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from cef_tickers import CEF_TICKERS

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Trade result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExDivTrade:
    """Represents one short trade around an ex-dividend event."""
    ticker: str
    ex_div_date: str               # YYYY-MM-DD
    dividend_amount: float
    entry_date: str
    entry_type: str                # 't_minus_1_close', 'ex_div_open', 'ex_div_close'
    entry_price: float
    exit_date: str
    exit_price: float
    holding_days: int
    entry_vix: float
    exit_vix: float
    # Fundamentals (optional – populated when CEFConnect data available)
    entry_zscore: Optional[float] = None
    entry_premium_discount: Optional[float] = None
    # Returns & costs
    gross_return_bps: float = 0.0
    bid_ask_cost_bps: float = 0.0
    borrow_fee_bps: float = 0.0
    net_return_bps: float = 0.0
    won: bool = False
    # Regime flags
    regime: str = "Normal (15-20)"
    regime_calm: bool = False
    regime_stressed: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Data Fetcher – Yahoo Finance wrapper with in-memory cache
# ──────────────────────────────────────────────────────────────────────────────

class DataFetcher:
    """Fetches OHLCV, dividend, and VIX data from Yahoo Finance."""

    def __init__(self, start_date: str, end_date: str, cache_dir: str = "./cef_data_cache") -> None:
        self.start = pd.Timestamp(start_date)
        self.end = pd.Timestamp(end_date)
        # Extend data window to cover look-back / look-forward needs
        self._fetch_start = (self.start - timedelta(days=260)).strftime("%Y-%m-%d")
        self._fetch_end = (self.end + timedelta(days=60)).strftime("%Y-%m-%d")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._div_cache: Dict[str, pd.DataFrame] = {}
        self._nav_cache: Dict[str, pd.DataFrame] = {}
        self._vix: Optional[pd.DataFrame] = None
        # Rate limiting settings
        self.request_delay = 1.0    # seconds between individual API calls
        self.group_delay = 5.0      # seconds between ticker groups
        self.max_retries = 3        # max retry attempts on failure

    # ── price data ──────────────────────────────────────────────────────────
    def get_prices(self, ticker: str) -> pd.DataFrame:
        """Return OHLCV DataFrame for *ticker* (cached)."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]
            
        cache_file = os.path.join(self.cache_dir, f"{ticker}_prices.pkl")
        if os.path.exists(cache_file):
            try:
                df = pd.read_pickle(cache_file)
                if not df.empty:
                    cached_min = df.index.min()
                    cached_max = df.index.max()
                    req_start = pd.Timestamp(self._fetch_start)
                    req_end = pd.Timestamp(self._fetch_end)
                    if cached_min <= req_start and cached_max >= req_end:
                        self._price_cache[ticker] = df
                        return df
            except Exception:
                pass

        for attempt in range(1, self.max_retries + 1):
            try:
                t = yf.Ticker(ticker)
                df = t.history(start=self._fetch_start, end=self._fetch_end, auto_adjust=False)
                if df.empty:
                    log.warning("No price data for %s", ticker)
                    self._price_cache[ticker] = pd.DataFrame()
                    return self._price_cache[ticker]
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                self._price_cache[ticker] = df
                df.to_pickle(cache_file)
                time.sleep(self.request_delay)
                return self._price_cache[ticker]
            except Exception as exc:
                backoff = self.group_delay * (2 ** (attempt - 1))  # 5s, 10s, 20s
                if attempt < self.max_retries:
                    log.warning("Error fetching prices for %s (attempt %d/%d): %s — retrying in %.0fs",
                                ticker, attempt, self.max_retries, exc, backoff)
                    time.sleep(backoff)
                else:
                    log.warning("Error fetching prices for %s after %d attempts: %s", ticker, self.max_retries, exc)
                    self._price_cache[ticker] = pd.DataFrame()
        return self._price_cache.get(ticker, pd.DataFrame())

    # ── dividends ───────────────────────────────────────────────────────────
    def get_dividends(self, ticker: str) -> pd.DataFrame:
        """Return dividends (date-indexed, 'Dividends' column) for *ticker*."""
        if ticker in self._div_cache:
            return self._div_cache[ticker]
            
        cache_file = os.path.join(self.cache_dir, f"{ticker}_divs.pkl")
        if os.path.exists(cache_file):
            try:
                df = pd.read_pickle(cache_file)
                if not df.empty:
                    cached_min = df.index.min()
                    cached_max = df.index.max()
                    req_start = pd.Timestamp(self.start)
                    req_end = pd.Timestamp(self.end)
                    if cached_min <= req_start and cached_max >= req_end:
                        self._div_cache[ticker] = df
                        return df
            except Exception:
                pass

        for attempt in range(1, self.max_retries + 1):
            try:
                t = yf.Ticker(ticker)
                divs = t.dividends
                if divs.empty:
                    self._div_cache[ticker] = pd.DataFrame()
                    return self._div_cache[ticker]
                divs.index = pd.to_datetime(divs.index).tz_localize(None)
                df = divs.to_frame(name="Dividends")
                # Filter to study window
                df = df[(df.index >= self.start) & (df.index <= self.end)]
                self._div_cache[ticker] = df
                df.to_pickle(cache_file)
                time.sleep(self.request_delay)
                return self._div_cache[ticker]
            except Exception as exc:
                backoff = self.group_delay * (2 ** (attempt - 1))
                if attempt < self.max_retries:
                    log.warning("Error fetching dividends for %s (attempt %d/%d): %s — retrying in %.0fs",
                                ticker, attempt, self.max_retries, exc, backoff)
                    time.sleep(backoff)
                else:
                    log.warning("Error fetching dividends for %s after %d attempts: %s", ticker, self.max_retries, exc)
                    self._div_cache[ticker] = pd.DataFrame()
        return self._div_cache.get(ticker, pd.DataFrame())

    def get_nav_and_stats(self, ticker: str, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch NAV history and calculate premium/discount and rolling Z-score."""
        if ticker in self._nav_cache:
            return self._nav_cache[ticker]
            
        cache_file = os.path.join(self.cache_dir, f"{ticker}_nav.pkl")
        if os.path.exists(cache_file):
            try:
                df = pd.read_pickle(cache_file)
                if not df.empty:
                    cached_min = df.index.min()
                    cached_max = df.index.max()
                    req_start = pd.Timestamp(self._fetch_start)
                    req_end = pd.Timestamp(self._fetch_end)
                    if cached_min <= req_start and cached_max >= req_end:
                        self._nav_cache[ticker] = df
                        return df
            except Exception:
                pass

        for attempt in range(1, self.max_retries + 1):
            try:
                nav_ticker = f"X{ticker}X"
                t = yf.Ticker(nav_ticker)
                nav_df = t.history(start=self._fetch_start, end=self._fetch_end, auto_adjust=False)
                if nav_df.empty:
                    self._nav_cache[ticker] = pd.DataFrame()
                    return self._nav_cache[ticker]
                
                if isinstance(nav_df.columns, pd.MultiIndex):
                    nav_df.columns = nav_df.columns.get_level_values(0)
                nav_df.index = pd.to_datetime(nav_df.index).tz_localize(None)
                nav_df = nav_df[["Close"]].rename(columns={"Close": "NAV"})
                
                # Join with price to calculate premium/discount
                if not prices_df.empty:
                    stats = nav_df.join(prices_df[["Close"]].rename(columns={"Close": "Price"}), how="inner")
                    stats["PremiumDiscount"] = (stats["Price"] - stats["NAV"]) / stats["NAV"] * 100
                    stats["RollingMean"] = stats["PremiumDiscount"].rolling(window=252, min_periods=30).mean()
                    stats["RollingStd"] = stats["PremiumDiscount"].rolling(window=252, min_periods=30).std()
                    stats["ZScore"] = (stats["PremiumDiscount"] - stats["RollingMean"]) / stats["RollingStd"]
                    self._nav_cache[ticker] = stats
                    stats.to_pickle(cache_file)
                    time.sleep(self.request_delay)
                    return stats
                else:
                    self._nav_cache[ticker] = pd.DataFrame()
                    return self._nav_cache[ticker]
                    
            except Exception as exc:
                backoff = self.group_delay * (2 ** (attempt - 1))
                if attempt < self.max_retries:
                    log.warning("Error fetching NAV for %s (attempt %d/%d): %s — retrying in %.0fs",
                                ticker, attempt, self.max_retries, exc, backoff)
                    time.sleep(backoff)
                else:
                    log.warning("Error fetching NAV for %s after %d attempts: %s", ticker, self.max_retries, exc)
                    self._nav_cache[ticker] = pd.DataFrame()
        return self._nav_cache.get(ticker, pd.DataFrame())

    # ── VIX ─────────────────────────────────────────────────────────────────
    def get_vix(self) -> pd.DataFrame:
        """Return VIX daily close."""
        if self._vix is None:
            for attempt in range(1, self.max_retries + 1):
                try:
                    vix = yf.Ticker("^VIX")
                    df = vix.history(start=self._fetch_start, end=self._fetch_end, auto_adjust=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    self._vix = df[["Close"]].rename(columns={"Close": "VIX"})
                    time.sleep(self.request_delay)
                    break
                except Exception as exc:
                    backoff = self.group_delay * (2 ** (attempt - 1))
                    if attempt < self.max_retries:
                        log.warning("Error fetching VIX (attempt %d/%d): %s — retrying in %.0fs",
                                    attempt, self.max_retries, exc, backoff)
                        time.sleep(backoff)
                    else:
                        log.warning("Error fetching VIX after %d attempts: %s", self.max_retries, exc)
                        self._vix = pd.DataFrame(columns=["VIX"])
        return self._vix

    def get_vix_on_date(self, date: pd.Timestamp) -> float:
        """Return the VIX close on *date*, falling back to nearest prior day."""
        vix = self.get_vix()
        if vix.empty:
            return 18.0  # neutral fallback
        idx = vix.index.get_indexer([date], method="ffill")
        if idx[0] == -1:
            return 18.0
        return float(vix.iloc[idx[0]]["VIX"])

    @staticmethod
    def classify_regime(vix: float) -> Tuple[str, bool, bool]:
        """Return (label, is_calm, is_stressed)."""
        if vix < 15:
            return "Calm (<15)", True, False
        elif vix <= 20:
            return "Normal (15-20)", False, False
        else:
            return "Stressed (>20)", False, True

    # ── universe discovery ──────────────────────────────────────────────────
    def get_cef_universe(self, min_div_events: int = 2) -> List[str]:
        """
        Return tickers from *CEF_TICKERS* that have at least
        *min_div_events* ex-dividend dates in the study window.
        """
        valid: List[str] = []
        total = len(CEF_TICKERS)
        for i, tkr in enumerate(CEF_TICKERS, 1):
            if i % 50 == 0 or i == total:
                log.info("Screening universe: %d / %d  (found %d so far)", i, total, len(valid))
            divs = self.get_dividends(tkr)
            if len(divs) >= min_div_events:
                valid.append(tkr)
        log.info("Universe: %d CEFs with >= %d ex-div events", len(valid), min_div_events)
        return valid


# ──────────────────────────────────────────────────────────────────────────────
# Cost Model
# ──────────────────────────────────────────────────────────────────────────────

class CostModel:
    """Calculate trading costs for short-sale strategies."""

    # Annual borrow-fee tiers (as a fraction, e.g. 0.001 = 0.1 %)
    BORROW_TIERS = {
        "low":    0.001,   # 0.1 % annual  – very liquid
        "medium": 0.005,   # 0.5 % annual
        "high":   0.020,   # 2.0 % annual  – illiquid / hard-to-borrow
    }

    # Spread cost per leg in basis points (entry + exit)
    SPREAD_PER_LEG_BPS = 10  # 0.10 % per leg → 0.20 % round-trip

    # Tickers considered highly liquid (low borrow cost)
    _HIGH_LIQUIDITY = {
        "PDI", "PTY", "GOF", "EOS", "UTF", "CSQ", "USA", "OXLC", "CLM",
        "CRF", "DNP", "HTD", "HYT", "BBN", "BSTZ", "BST", "ADX", "QQQX",
        "BXMX", "EXG", "ETG", "ETY", "ETJ", "ETV", "ETW", "UTG", "RQI",
        "RNP", "RFI", "NFJ", "GDV", "PFF", "DSL", "PHK", "NVG", "NEA",
        "NAD", "NZF", "NUV", "KYN", "STK", "BDJ", "GGN", "FAX", "AOD",
        "GAB", "AVK", "BGY", "IGR", "IGD", "HQL", "HQH",
    }

    @classmethod
    def total_cost_bps(cls, ticker: str, holding_days: int, entry_price: float) -> Tuple[float, float]:
        """Return (bid_ask_and_commission_bps, borrow_bps)."""
        # Borrow cost: flat 4.2% annual
        annual_borrow = 0.042
        borrow_bps = (annual_borrow / 365.0) * holding_days * 10_000.0
        
        # Spread cost: $0.03 per share round-trip
        # Commission: $1.30 per 100 shares round-trip = $0.013 per share
        total_share_cost = 0.03 + 0.013  # $0.043 per share round-trip
        transaction_bps = (total_share_cost / entry_price) * 10_000.0
        
        return transaction_bps, borrow_bps


# ──────────────────────────────────────────────────────────────────────────────
# Backtest Engine
# ──────────────────────────────────────────────────────────────────────────────

class ExDivBacktester:
    """
    Enumerate ex-dividend events across the CEF universe and generate
    short trades for every combination of entry type × holding period.
    """

    ENTRY_TYPES = ["t_minus_1_close", "ex_div_open", "ex_div_close"]
    HOLDING_PERIODS = [5, 10, 15, 20]

    def __init__(self, start_date: str = "2022-01-01",
                 end_date: str = "2024-12-31") -> None:
        self.fetcher = DataFetcher(start_date, end_date)
        self._cefconnect_scraper = None  # lazy import

    def set_cefconnect_scraper(self, scraper) -> None:
        """Optionally attach a CEFConnectScraper for fundamental data."""
        self._cefconnect_scraper = scraper

    # ── main entry point ────────────────────────────────────────────────────
    def run_backtest(
        self,
        tickers: Optional[List[str]] = None,
        entry_types: Optional[List[str]] = None,
        holding_periods: Optional[List[int]] = None,
        min_div_events: int = 2,
    ) -> List[ExDivTrade]:
        """
        Run the backtest.
        """
        entry_types = entry_types or self.ENTRY_TYPES
        holding_periods = holding_periods or self.HOLDING_PERIODS

        if tickers is None:
            tickers = self.fetcher.get_cef_universe(min_div_events)

        trades: List[ExDivTrade] = []
        total = len(tickers)

        for idx, tkr in enumerate(tickers, 1):
            log.info("Backtesting %s  (%d / %d)", tkr, idx, total)
            divs = self.fetcher.get_dividends(tkr)
            prices = self.fetcher.get_prices(tkr)
            if divs.empty or prices.empty:
                # Still apply inter-ticker cooldown to avoid burst requests
                if idx < total:
                    log.info("Rate limit cooldown: waiting %.0fs before next ticker...", self.fetcher.group_delay)
                    time.sleep(self.fetcher.group_delay)
                continue

            for ex_date, row in divs.iterrows():
                div_amount = float(row["Dividends"])
                for etype in entry_types:
                    for hperiod in holding_periods:
                        trade = self._create_trade(
                            tkr, ex_date, div_amount, prices, etype, hperiod
                        )
                        if trade is not None:
                            trades.append(trade)

            # Inter-ticker cooldown to prevent rate-limiting by Yahoo Finance
            if idx < total:
                log.info("Rate limit cooldown: waiting %.0fs before next ticker...", self.fetcher.group_delay)
                time.sleep(self.fetcher.group_delay)

        log.info("Backtest complete: %d trades generated", len(trades))
        return trades

    # ── single trade construction ───────────────────────────────────────────
    def _create_trade(
        self,
        ticker: str,
        ex_date: pd.Timestamp,
        div_amount: float,
        prices: pd.DataFrame,
        entry_type: str,
        holding_days: int,
    ) -> Optional[ExDivTrade]:
        """Build one ExDivTrade or return None if data is insufficient."""
        try:
            # --- resolve entry date & price ---
            entry_date, entry_price = self._resolve_entry(
                prices, ex_date, entry_type
            )
            if entry_date is None or entry_price is None or entry_price <= 0:
                return None

            # --- resolve exit date & price ---
            exit_date, exit_price = self._resolve_exit(
                prices, entry_date, holding_days
            )
            if exit_date is None or exit_price is None or exit_price <= 0:
                return None

            # --- gross return (short: profit when price drops) ---
            gross_bps = (entry_price - exit_price) / entry_price * 10_000

            # --- costs ---
            ba_cost, borrow_cost = CostModel.total_cost_bps(ticker, holding_days, entry_price)
            net_bps = gross_bps - ba_cost - borrow_cost

            # --- VIX / regime ---
            entry_vix = self.fetcher.get_vix_on_date(entry_date)
            exit_vix = self.fetcher.get_vix_on_date(exit_date)
            regime, is_calm, is_stressed = DataFetcher.classify_regime(entry_vix)

            # --- Historical Z-Score & Premium/Discount ---
            zscore, prem_disc = None, None
            nav_stats = self.fetcher.get_nav_and_stats(ticker, prices)
            if not nav_stats.empty and entry_date in nav_stats.index:
                row_stats = nav_stats.loc[entry_date]
                if pd.notna(row_stats.get("ZScore")):
                    zscore = float(row_stats["ZScore"])
                if pd.notna(row_stats.get("PremiumDiscount")):
                    prem_disc = float(row_stats["PremiumDiscount"])

            return ExDivTrade(
                ticker=ticker,
                ex_div_date=ex_date.strftime("%Y-%m-%d"),
                dividend_amount=div_amount,
                entry_date=entry_date.strftime("%Y-%m-%d"),
                entry_type=entry_type,
                entry_price=round(entry_price, 4),
                exit_date=exit_date.strftime("%Y-%m-%d"),
                exit_price=round(exit_price, 4),
                holding_days=holding_days,
                entry_vix=round(entry_vix, 2),
                exit_vix=round(exit_vix, 2),
                entry_zscore=zscore,
                entry_premium_discount=prem_disc,
                gross_return_bps=round(gross_bps, 2),
                bid_ask_cost_bps=round(ba_cost, 2),
                borrow_fee_bps=round(borrow_cost, 2),
                net_return_bps=round(net_bps, 2),
                won=(net_bps > 0),
                regime=regime,
                regime_calm=is_calm,
                regime_stressed=is_stressed,
            )
        except Exception as exc:
            log.debug("Trade creation failed %s %s %s: %s",
                      ticker, ex_date, entry_type, exc)
            return None

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_entry(
        prices: pd.DataFrame, ex_date: pd.Timestamp, entry_type: str
    ) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
        """Resolve entry date and price depending on *entry_type*."""
        idx = prices.index

        if entry_type == "t_minus_1_close":
            # Last trading day before ex-date
            prior = idx[idx < ex_date]
            if prior.empty:
                return None, None
            dt = prior[-1]
            return dt, float(prices.loc[dt, "Close"])

        elif entry_type == "ex_div_open":
            # Ex-dividend day open
            if ex_date in idx:
                return ex_date, float(prices.loc[ex_date, "Open"])
            # Find nearest trading day on or after
            after = idx[idx >= ex_date]
            if after.empty:
                return None, None
            dt = after[0]
            return dt, float(prices.loc[dt, "Open"])

        elif entry_type == "ex_div_close":
            if ex_date in idx:
                return ex_date, float(prices.loc[ex_date, "Close"])
            after = idx[idx >= ex_date]
            if after.empty:
                return None, None
            dt = after[0]
            return dt, float(prices.loc[dt, "Close"])

        return None, None

    @staticmethod
    def _resolve_exit(
        prices: pd.DataFrame, entry_date: pd.Timestamp, holding_days: int
    ) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
        """Resolve exit date/price: the close *holding_days* trading days after entry."""
        idx = prices.index
        after = idx[idx > entry_date]
        if len(after) < holding_days:
            return None, None
        exit_dt = after[holding_days - 1]
        return exit_dt, float(prices.loc[exit_dt, "Close"])


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics Analyzer
# ──────────────────────────────────────────────────────────────────────────────

class DiagnosticsAnalyzer:
    """Analyse a list of ExDivTrade objects."""

    def __init__(self, trades: List[ExDivTrade]) -> None:
        self.trades = trades
        self._df: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.DataFrame([asdict(t) for t in self.trades])
        return self._df

    # ── summary stats ───────────────────────────────────────────────────────
    def summary_stats(self) -> Dict:
        """Return aggregate statistics for the trade set."""
        if not self.trades:
            return {"total_trades": 0}

        net = self.df["net_return_bps"]
        winners = net[net > 0]
        losers = net[net <= 0]

        win_rate = len(winners) / len(net) * 100 if len(net) > 0 else 0
        avg_winner = float(winners.mean()) if len(winners) > 0 else 0
        avg_loser = float(losers.mean()) if len(losers) > 0 else 0
        win_loss_ratio = abs(avg_winner / avg_loser) if avg_loser != 0 else (float("inf") if len(winners) > 0 else 0.0)
        profit_factor = (
            float(winners.sum() / abs(losers.sum()))
            if losers.sum() != 0
            else (float("inf") if len(winners) > 0 else 0.0)
        )
        sharpe = float(net.mean() / net.std()) if net.std() > 0 else 0

        # Advanced Risk Calculations
        initial_bps = 10000.0
        equity = initial_bps + net.cumsum()
        peaks = equity.cummax()
        drawdowns_pct = (equity - peaks) / peaks * 100.0
        max_dd_bps = float((equity - peaks).min()) if not equity.empty else 0.0
        max_dd_pct = float(drawdowns_pct.min()) if not drawdowns_pct.empty else 0.0
        ulcer_idx = float(np.sqrt(np.mean(drawdowns_pct ** 2))) if not drawdowns_pct.empty else 0.0

        # Sortino Ratio
        excess_returns = net - 0.0
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            sortino = float("inf") if excess_returns.mean() > 0 else 0.0
        else:
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
            r_sortino = excess_returns.mean() / downside_deviation
            sortino = float(r_sortino * np.sqrt(252.0))

        # Calmar Ratio
        ann_return_pct = (net.mean() * 252.0) / 100.0
        calmar = float(ann_return_pct / abs(max_dd_pct)) if max_dd_pct != 0 else (float("inf") if ann_return_pct > 0 else 0.0)

        # Gain-to-Pain Ratio
        neg_returns = net[net < 0]
        gain_to_pain = float(net.sum() / abs(neg_returns.sum())) if not neg_returns.empty and neg_returns.sum() != 0 else (float("inf") if net.sum() > 0 else 0.0)

        # Omega Ratio
        omega = float(winners.sum() / abs(neg_returns.sum())) if not neg_returns.empty and neg_returns.sum() != 0 else (float("inf") if winners.sum() > 0 else 0.0)

        # Tail Ratio
        p5 = np.percentile(net, 5) if not net.empty else 0.0
        p95 = np.percentile(net, 95) if not net.empty else 0.0
        tail = float(p95 / abs(p5)) if p5 != 0 else (float("inf") if p95 > 0 else 0.0)

        # Skewness and Kurtosis
        skewness_val = float(net.skew()) if pd.notna(net.skew()) else 0.0
        kurt_val = float(net.kurt()) if pd.notna(net.kurt()) else 0.0

        # VaR and CVaR 95%
        var_95 = float(np.percentile(net, 5)) if not net.empty else 0.0
        cvar_returns = net[net <= var_95]
        cvar_95 = float(cvar_returns.mean()) if not cvar_returns.empty else var_95

        return {
            "total_trades": len(net),
            "win_rate": round(win_rate, 2),
            "avg_winner_bps": round(avg_winner, 2),
            "avg_loser_bps": round(avg_loser, 2),
            "win_loss_ratio": round(win_loss_ratio, 2) if not np.isinf(win_loss_ratio) else float("inf"),
            "profit_factor": round(profit_factor, 2) if not np.isinf(profit_factor) else float("inf"),
            "total_return_bps": round(float(net.sum()), 2),
            "avg_return_bps": round(float(net.mean()), 2),
            "std_return_bps": round(float(net.std()), 2),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4) if not np.isinf(sortino) else float("inf"),
            "calmar_ratio": round(calmar, 4) if not np.isinf(calmar) else float("inf"),
            "gain_to_pain_ratio": round(gain_to_pain, 4) if not np.isinf(gain_to_pain) else float("inf"),
            "omega_ratio": round(omega, 4) if not np.isinf(omega) else float("inf"),
            "tail_ratio": round(tail, 4) if not np.isinf(tail) else float("inf"),
            "skewness": round(skewness_val, 4),
            "kurtosis": round(kurt_val, 4),
            "max_drawdown_bps": round(max_dd_bps, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "var_95_bps": round(var_95, 2),
            "cvar_95_bps": round(cvar_95, 2),
            "ulcer_index": round(ulcer_idx, 2),
            "max_return_bps": round(float(net.max()), 2),
            "min_return_bps": round(float(net.min()), 2),
        }

    # ── regime analysis ─────────────────────────────────────────────────────
    def regime_analysis(self) -> pd.DataFrame:
        """Group trades by VIX regime."""
        if self.df.empty:
            return pd.DataFrame()
        g = self.df.groupby("regime").agg(
            n_trades=("net_return_bps", "count"),
            win_rate=("won", lambda x: round(x.mean() * 100, 2)),
            avg_return_bps=("net_return_bps", lambda x: round(x.mean(), 2)),
            total_return_bps=("net_return_bps", lambda x: round(x.sum(), 2)),
            avg_vix=("entry_vix", lambda x: round(x.mean(), 2)),
        ).reset_index()
        return g

    # ── entry-point analysis ────────────────────────────────────────────────
    def entry_point_analysis(self) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame()
        g = self.df.groupby("entry_type").agg(
            n_trades=("net_return_bps", "count"),
            win_rate=("won", lambda x: round(x.mean() * 100, 2)),
            avg_return_bps=("net_return_bps", lambda x: round(x.mean(), 2)),
            total_return_bps=("net_return_bps", lambda x: round(x.sum(), 2)),
        ).reset_index()
        return g

    # ── holding-period analysis ─────────────────────────────────────────────
    def holding_period_analysis(self) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame()
        g = self.df.groupby("holding_days").agg(
            n_trades=("net_return_bps", "count"),
            win_rate=("won", lambda x: round(x.mean() * 100, 2)),
            avg_return_bps=("net_return_bps", lambda x: round(x.mean(), 2)),
            total_return_bps=("net_return_bps", lambda x: round(x.sum(), 2)),
        ).reset_index()
        return g

    # ── ticker analysis ─────────────────────────────────────────────────────
    def ticker_analysis(self, top_n: int = 20) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame()
            
        def safe_mean(x):
            valid = [v for v in x if v is not None and not pd.isna(v)]
            return round(sum(valid) / len(valid), 2) if valid else None
            
        g = self.df.groupby("ticker").agg(
            n_trades=("net_return_bps", "count"),
            win_rate=("won", lambda x: round(x.mean() * 100, 2)),
            avg_return_bps=("net_return_bps", lambda x: round(x.mean(), 2)),
            total_return_bps=("net_return_bps", lambda x: round(x.sum(), 2)),
            avg_zscore=("entry_zscore", safe_mean),
            avg_prem_disc=("entry_premium_discount", safe_mean),
        ).reset_index()
        g = g.sort_values("total_return_bps", ascending=False)
        return g.head(top_n)

    # ── cross-tab heatmap data (regime × holding) ───────────────────────────
    def regime_holding_heatmap(self) -> Dict:
        """Return data for a VIX-regime × holding-period win-rate heatmap."""
        if self.df.empty:
            return {"regimes": [], "holding_periods": [], "win_rates": []}
        pivot = self.df.pivot_table(
            values="won", index="regime", columns="holding_days",
            aggfunc="mean"
        ) * 100
        return {
            "regimes": pivot.index.tolist(),
            "holding_periods": [int(c) for c in pivot.columns.tolist()],
            "win_rates": pivot.values.round(2).tolist(),
        }

    # ── equity curve ────────────────────────────────────────────────────────
    def equity_curve(self) -> List[Dict]:
        """Cumulative PnL sorted by entry date."""
        if self.df.empty:
            return []
        sorted_df = self.df.sort_values("entry_date")
        cum = sorted_df["net_return_bps"].cumsum()
        result = []
        for dt, val in zip(sorted_df["entry_date"], cum):
            result.append({"date": dt, "cumulative_pnl_bps": round(float(val), 2)})
        return result

    # ── return distribution ─────────────────────────────────────────────────
    def return_distribution(self) -> List[float]:
        """List of net returns for histogram."""
        if self.df.empty:
            return []
        return self.df["net_return_bps"].round(2).tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Reporter – Print & Export
# ──────────────────────────────────────────────────────────────────────────────

class BacktestReporter:
    """Print formatted reports and export data."""

    @staticmethod
    def print_summary(stats: Dict) -> None:
        """Print summary statistics to console."""
        print("\n" + "=" * 60)
        print("  CEF EX-DIVIDEND BACKTEST — SUMMARY")
        print("=" * 60)
        if stats.get("total_trades", 0) == 0:
            print("  No trades generated.")
            return
        print(f"  Total Trades:      {stats['total_trades']:,}")
        print(f"  Win Rate:          {stats['win_rate']:.1f}%")
        print(f"  Avg Winner:        {stats['avg_winner_bps']:+.1f} bps")
        print(f"  Avg Loser:         {stats['avg_loser_bps']:+.1f} bps")
        print(f"  Win/Loss Ratio:    {stats['win_loss_ratio']:.2f}")
        print(f"  Profit Factor:     {stats['profit_factor']:.2f}")
        print(f"  Total Return:      {stats['total_return_bps']:+,.1f} bps")
        print(f"  Avg Return:        {stats['avg_return_bps']:+.2f} bps")
        print(f"  Std Return:        {stats['std_return_bps']:.2f} bps")
        print(f"  Sharpe Ratio:      {stats['sharpe_ratio']:.4f}")
        print(f"  Best Trade:        {stats['max_return_bps']:+.1f} bps")
        print(f"  Worst Trade:       {stats['min_return_bps']:+.1f} bps")
        print("=" * 60 + "\n")

    @staticmethod
    def print_regime_report(regime_df: pd.DataFrame) -> None:
        """Print regime analysis table."""
        print("\n  REGIME ANALYSIS")
        print("  " + "-" * 56)
        if regime_df.empty:
            print("  No data.")
            return
        print(f"  {'Regime':<18} {'Trades':>7} {'Win%':>7} {'Avg Ret':>9} {'Total':>10} {'AvgVIX':>7}")
        print("  " + "-" * 56)
        for _, r in regime_df.iterrows():
            print(
                f"  {r['regime']:<18} {r['n_trades']:>7,} {r['win_rate']:>6.1f}% "
                f"{r['avg_return_bps']:>+8.1f} {r['total_return_bps']:>+9.1f} {r['avg_vix']:>7.1f}"
            )
        print()

    @staticmethod
    def print_entry_report(entry_df: pd.DataFrame) -> None:
        """Print entry-point analysis table."""
        print("\n  ENTRY POINT ANALYSIS")
        print("  " + "-" * 50)
        if entry_df.empty:
            print("  No data.")
            return
        print(f"  {'Entry Type':<18} {'Trades':>7} {'Win%':>7} {'Avg Ret':>9} {'Total':>10}")
        print("  " + "-" * 50)
        for _, r in entry_df.iterrows():
            print(
                f"  {r['entry_type']:<18} {r['n_trades']:>7,} {r['win_rate']:>6.1f}% "
                f"{r['avg_return_bps']:>+8.1f} {r['total_return_bps']:>+9.1f}"
            )
        print()

    @staticmethod
    def print_holding_report(holding_df: pd.DataFrame) -> None:
        """Print holding-period analysis table."""
        print("\n  HOLDING PERIOD ANALYSIS")
        print("  " + "-" * 50)
        if holding_df.empty:
            print("  No data.")
            return
        print(f"  {'Hold Days':<12} {'Trades':>7} {'Win%':>7} {'Avg Ret':>9} {'Total':>10}")
        print("  " + "-" * 50)
        for _, r in holding_df.iterrows():
            print(
                f"  {int(r['holding_days']):>5}d       {r['n_trades']:>7,} {r['win_rate']:>6.1f}% "
                f"{r['avg_return_bps']:>+8.1f} {r['total_return_bps']:>+9.1f}"
            )
        print()

    @staticmethod
    def print_ticker_report(ticker_df: pd.DataFrame) -> None:
        """Print top tickers table."""
        print("\n  TOP TICKERS BY TOTAL RETURN")
        print("  " + "-" * 55)
        if ticker_df.empty:
            print("  No data.")
            return
        print(f"  {'Ticker':<8} {'Trades':>7} {'Win%':>7} {'Avg Ret':>9} {'Total Ret':>10}")
        print("  " + "-" * 55)
        for _, r in ticker_df.iterrows():
            print(
                f"  {r['ticker']:<8} {r['n_trades']:>7,} {r['win_rate']:>6.1f}% "
                f"{r['avg_return_bps']:>+8.1f} {r['total_return_bps']:>+9.1f}"
            )
        print()

    # ── CSV export ──────────────────────────────────────────────────────────
    @staticmethod
    def export_trades_csv(trades: List[ExDivTrade], filepath: str) -> None:
        """Export full trade list to CSV."""
        df = pd.DataFrame([asdict(t) for t in trades])
        df.to_csv(filepath, index=False)
        log.info("Exported %d trades to %s", len(trades), filepath)

    # ── JSON export ─────────────────────────────────────────────────────────
    @staticmethod
    def export_summary_json(stats: Dict, filepath: str) -> None:
        """Export summary stats to JSON."""
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        log.info("Exported summary to %s", filepath)

    # ── dashboard data ──────────────────────────────────────────────────────
    @staticmethod
    def generate_dashboard_data(
        trades: List[ExDivTrade],
        analyzer: DiagnosticsAnalyzer,
        html_filepath: str = "exdiv_backtest_dashboard.html",
    ) -> None:
        """Generate JSON and embed it into the dashboard HTML file."""
        stats = analyzer.summary_stats()
        regime_df = analyzer.regime_analysis()
        entry_df = analyzer.entry_point_analysis()
        holding_df = analyzer.holding_period_analysis()
        top_tickers_df = analyzer.ticker_analysis(top_n=30)

        # Compute ticker-specific analysis
        ticker_data = {}
        if trades:
            unique_tickers = sorted(list(set(t.ticker for t in trades)))
            for tkr in unique_tickers:
                tkr_trades = [t for t in trades if t.ticker == tkr]
                tkr_analyzer = DiagnosticsAnalyzer(tkr_trades)
                tkr_stats = tkr_analyzer.summary_stats()
                tkr_regime = tkr_analyzer.regime_analysis()
                tkr_entry = tkr_analyzer.entry_point_analysis()
                tkr_holding = tkr_analyzer.holding_period_analysis()
                
                ticker_data[tkr] = {
                    "summary": tkr_stats,
                    "regime_analysis": tkr_regime.to_dict(orient="records") if not tkr_regime.empty else [],
                    "entry_analysis": tkr_entry.to_dict(orient="records") if not tkr_entry.empty else [],
                    "holding_analysis": tkr_holding.to_dict(orient="records") if not tkr_holding.empty else [],
                    "heatmap_data": tkr_analyzer.regime_holding_heatmap(),
                    "equity_curve": tkr_analyzer.equity_curve(),
                    "return_distribution": tkr_analyzer.return_distribution()
                }

        data = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": stats,
            "regime_analysis": regime_df.to_dict(orient="records") if not regime_df.empty else [],
            "entry_analysis": entry_df.to_dict(orient="records") if not entry_df.empty else [],
            "holding_analysis": holding_df.to_dict(orient="records") if not holding_df.empty else [],
            "heatmap_data": analyzer.regime_holding_heatmap(),
            "top_tickers": top_tickers_df.to_dict(orient="records") if not top_tickers_df.empty else [],
            "equity_curve": analyzer.equity_curve(),
            "return_distribution": analyzer.return_distribution(),
            "ticker_data": ticker_data,
            "raw_trades": [
                {"t": t.ticker, "r": t.regime, "e": t.entry_type, "h": t.holding_days, "n": t.net_return_bps, "w": t.won, "p": t.entry_premium_discount, "z": t.entry_zscore, "d": t.entry_date}
                for t in trades
            ] if trades else []
        }

        # Embed into HTML
        if not os.path.exists(html_filepath):
            log.error("Dashboard HTML file not found at %s. Cannot embed data.", html_filepath)
            return
            
        with open(html_filepath, "r", encoding="utf-8") as f:
            html = f.read()
            
        # Clean out any old embedded data if we ran this before
        import re
        html = re.sub(r'<script id="embedded-data">.*?</script>', '', html, flags=re.DOTALL)
        
        # Clean NaNs and Infinities recursively to make sure it is 100% compliant JSON
        def clean_nans(obj):
            import math
            if isinstance(obj, dict):
                return {k: clean_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nans(v) for v in obj]
            elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj
            
        cleaned_data = clean_nans(data)
        json_str = json.dumps(cleaned_data)
        script_tag = f'<script id="embedded-data">\nwindow.DASHBOARD_DATA = {json_str};\n</script>\n</body>'
        html = html.replace('</body>', script_tag)
        
        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(html)
            
        log.info("Dashboard data embedded into %s", html_filepath)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point (optional)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CEF Ex-Div Backtest Engine")
    parser.add_argument("--start", default="2022-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--tickers", nargs="*", default=None, help="Specific tickers")
    parser.add_argument("--csv", default=None, help="Export trades CSV path")
    parser.add_argument("--json", default=None, help="Export summary JSON path")
    parser.add_argument("--dashboard", default="dashboard_data.json",
                        help="Dashboard data JSON path")
    args = parser.parse_args()

    bt = ExDivBacktester(args.start, args.end)
    all_trades = bt.run_backtest(tickers=args.tickers)

    analyzer = DiagnosticsAnalyzer(all_trades)
    stats = analyzer.summary_stats()

    BacktestReporter.print_summary(stats)
    BacktestReporter.print_regime_report(analyzer.regime_analysis())
    BacktestReporter.print_entry_report(analyzer.entry_point_analysis())
    BacktestReporter.print_holding_report(analyzer.holding_period_analysis())
    BacktestReporter.print_ticker_report(analyzer.ticker_analysis())

    if args.csv:
        BacktestReporter.export_trades_csv(all_trades, args.csv)
    if args.json:
        BacktestReporter.export_summary_json(stats, args.json)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_filepath = os.path.join(base_dir, "exdiv_backtest_dashboard.html")
    BacktestReporter.generate_dashboard_data(all_trades, analyzer, html_filepath)
    print("\n✅ Dashboard data embedded into exdiv_backtest_dashboard.html")
    print("   Double-click exdiv_backtest_dashboard.html to view.\n")
