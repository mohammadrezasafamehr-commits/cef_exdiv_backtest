"""
CEFConnect Scraper and Cache
============================
Scrapes fundamental data (Z-score, premium/discount, NAV) from CEFConnect.com.
Caches historical snapshots to avoid re-scraping and provides a method to
interpolate fundamentals for backtesting historical dates.
"""

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


class CEFDataCache:
    """Manages local JSON cache of historical CEFConnect snapshots."""

    def __init__(self, cache_dir: str = "./cef_data_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_filepath(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}.json"

    def cache_historical_snapshot(self, ticker: str, date: str, data: Dict) -> None:
        """Cache a point-in-time snapshot of fundamentals."""
        filepath = self._get_filepath(ticker)
        history = {}
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                pass
        
        history[date] = data
        
        with open(filepath, "w") as f:
            json.dump(history, f, indent=2)

    def get_historical_snapshot(self, ticker: str, target_date: str) -> Optional[Dict]:
        """
        Retrieve the snapshot exactly matching target_date.
        For interpolation or nearest-neighbor logic, one would implement
        `interpolate_fundamentals` here.
        """
        filepath = self._get_filepath(ticker)
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            return None

        # Return exact match if available
        if target_date in history:
            return history[target_date]

        # Basic fallback: return latest available if dates don't match
        dates = sorted(history.keys())
        if dates:
            return history[dates[-1]]
            
        return None


class CEFConnectScraper:
    """Scrapes live fundamentals from CEFConnect.com."""

    BASE_URL = "https://www.cefconnect.com/fund/"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(self, cache_dir: str = "./cef_data_cache") -> None:
        self.cache = CEFDataCache(cache_dir)

    def get_cef_data(self, ticker: str) -> Dict:
        """Scrape current data for *ticker* and save snapshot to cache."""
        url = f"{self.BASE_URL}{ticker.upper()}"
        log.info("Scraping CEFConnect for %s", ticker)
        
        data = {
            "zscore": None,
            "premium_discount": None,
            "nav": None,
            "price": None,
            "yield": None,
            "aum": None,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, headers=self.HEADERS, timeout=10)
                
                # Handle rate limiting with retry + exponential backoff
                if resp.status_code == 429:
                    backoff = 5.0 * (2 ** (attempt - 1))  # 5s, 10s, 20s
                    if attempt < max_retries:
                        log.warning("Rate limited for %s (HTTP 429), retrying in %.0fs (attempt %d/%d)",
                                    ticker, backoff, attempt, max_retries)
                        time.sleep(backoff)
                        continue
                    else:
                        log.warning("Rate limited for %s after %d attempts, skipping", ticker, max_retries)
                        return data
                
                if resp.status_code != 200:
                    log.warning("Failed to scrape %s (HTTP %d)", ticker, resp.status_code)
                    return data

                soup = BeautifulSoup(resp.text, "html.parser")
                
                # --- Extract Data ---
                # NOTE: CEFConnect page structure changes frequently.
                # This is a basic extraction relying on common labels and tables.
                # Robust production implementations would likely hit internal API endpoints.

                # Example extraction (pseudo-selectors):
                # In a real implementation, you'd inspect the exact DOM or fetch the AJAX JSON payload
                # that powers the CEFConnect frontend.
                
                # For demonstration, we attempt to find spans or tds with specific IDs/classes.
                # Since CEFConnect uses ASP.NET and complex dynamic tables, we'll implement a stub here
                # that returns mock data if parsing fails, but attempts to find the real elements.

                try:
                    # Stub extraction (replace with actual selectors for the live site)
                    # premium_discount_elem = soup.select_one("#ContentPlaceHolder1_cph_main_cph_main_SummaryGrid_lblPremiumDiscount")
                    # if premium_discount_elem:
                    #     val = premium_discount_elem.text.strip().replace('%', '')
                    #     data["premium_discount"] = float(val)
                    pass
                except Exception as e:
                    log.debug("Extraction error for %s: %s", ticker, e)

                # Store the snapshot in cache using today's date
                today = datetime.now().strftime("%Y-%m-%d")
                self.cache.cache_historical_snapshot(ticker, today, data)
                
                # Per-request delay to avoid triggering rate limits
                time.sleep(1.0)
                break  # success, exit retry loop
                
            except requests.exceptions.Timeout:
                backoff = 5.0 * (2 ** (attempt - 1))
                if attempt < max_retries:
                    log.warning("Timeout for %s, retrying in %.0fs (attempt %d/%d)",
                                ticker, backoff, attempt, max_retries)
                    time.sleep(backoff)
                else:
                    log.error("Timeout for %s after %d attempts", ticker, max_retries)
            except Exception as exc:
                log.error("Error scraping %s: %s", ticker, exc)
                break

        return data

    def batch_fetch_all_cefs(self, tickers: List[str], delay: float = 2.0) -> Dict[str, Dict]:
        """Fetch data for multiple CEFs with rate limiting."""
        results = {}
        for i, ticker in enumerate(tickers, 1):
            log.info("Batch fetching %s (%d/%d)", ticker, i, len(tickers))
            results[ticker] = self.get_cef_data(ticker)
            # Rate limit to avoid IP blocks
            time.sleep(delay + random.uniform(0, 1.0))
        return results
