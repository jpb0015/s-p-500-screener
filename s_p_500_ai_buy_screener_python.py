#!/usr/bin/env python3
"""
S&P 500 AI Buy Screener
-----------------------

What it does
- Pulls S&P 500 tickers (Wikipedia) with a safe static fallback
- Downloads 2 years of daily price data via yfinance
- Builds features (SMA50/200, RSI14, momentum, volatility)
- Trains a pooled RandomForest classifier to predict next-N-day positive return
- Produces a ranked BUY list with probabilities and guardrails (e.g., SMA50>SMA200)
- Optional alerts: Email (SMTP) and Discord (webhook)

Quick start
1) pip install yfinance pandas scikit-learn ta numpy requests beautifulsoup4 lxml joblib
2) python sp500_ai_screener.py --proba 0.58 --lookahead 5 --min-days 260

Scheduling (macOS/Linux)
- crontab -e
- Add (runs at 8:00 AM CT on weekdays):
  0 8 * * 1-5 /usr/bin/python3 /path/to/sp500_ai_screener.py --proba 0.6 >> /path/to/log.txt 2>&1

Environment variables for alerts (optional)
- EMAIL_SMTP, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_TO
- DISCORD_WEBHOOK

Outputs
- data/buy_signals.csv (ranked predictions for today)
- data/model.joblib (trained model)
- data/metrics.json (recent backtest precision/coverage)

"""

from __future__ import annotations
import os
import sys
import json
import time
import math
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# Third-party
try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("yfinance is required: pip install yfinance")

try:
    from ta.momentum import RSIIndicator
    from ta.trend import SMAIndicator
except Exception as e:
    raise SystemExit("ta is required: pip install ta")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump, load

# ----------------------------------
# Config
# ----------------------------------
LOOKBACK_YEARS_DEFAULT = 2
RANDOM_STATE = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

STATIC_SP500 = [
    # Minimal static fallback subset (extend if you like)
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","LLY","AVGO","TSLA",
    "JPM","V","XOM","UNH","PG","JNJ","MA","COST","HD","ORCL"
]

# ----------------------------------
# Utilities
# ----------------------------------

def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def safe_read_html_sp500() -> List[str]:
    """Try to fetch S&P 500 tickers from Wikipedia, else return STATIC_SP500."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        # Heuristic: first table typically holds constituents
        tickers = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        tickers = [t.strip().upper() for t in tickers if t and t != "nan"]
        # Remove weird footnote superscripts
        tickers = [t.replace("\n", "").split(" ")[0] for t in tickers]
        if len(tickers) < 50:
            raise ValueError("Too few tickers scraped; using fallback")
        return tickers
    except Exception as e:
        log(f"Wikipedia scrape failed ({e}); falling back to static {len(STATIC_SP500)} tickers.")
        return STATIC_SP500


def fetch_history(ticker: str, start: datetime) -> pd.DataFrame:
    """Download OHLCV and return a cleaned DataFrame."""
    try:
        df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.title)
        df = df[~df.index.duplicated(keep='last')]
        return df
    except Exception:
        return pd.DataFrame()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    # SMAs
    df["SMA50"] = SMAIndicator(close, 50).sma_indicator()
    df["SMA200"] = SMAIndicator(close, 200).sma_indicator()
    # RSI
    df["RSI14"] = RSIIndicator(close, 14).rsi()
    # Momentum & volatility
    df["RET_1D"] = close.pct_change()
    df["RET_5D"] = close.pct_change(5)
    df["RET_20D"] = close.pct_change(20)
    df["VOL_20D"] = df["RET_1D"].rolling(20).std()
    # Price relative to SMAs
    df["PRC_OVER_SMA50"] = close / df["SMA50"] - 1.0
    df["PRC_OVER_SMA200"] = close / df["SMA200"] - 1.0
    return df


def make_labels(df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
    future_ret = df["Close"].pct_change(lookahead).shift(-lookahead)
    df["TARGET"] = (future_ret > 0).astype(int)
    return df


FEATURE_COLS = [
    "SMA50","SMA200","RSI14","RET_1D","RET_5D","RET_20D","VOL_20D","PRC_OVER_SMA50","PRC_OVER_SMA200"
]


def build_pooled_dataset(price_map: Dict[str, pd.DataFrame], lookahead: int, min_days: int) -> pd.DataFrame:
    frames = []
    for tk, df in price_map.items():
        if df is None or df.empty or len(df) < min_days:
            continue
        df = df.copy()
        df = add_features(df)
        df = make_labels(df, lookahead)
        df["ticker"] = tk
        df.dropna(inplace=True)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, axis=0)
    # Keep only what we need
    cols = FEATURE_COLS + ["TARGET", "ticker"]
    all_df = all_df.loc[:, cols]
    return all_df


@dataclass
class TrainResult:
    model: RandomForestClassifier
    cutoff_date: str
    precision: float
    coverage: float


def train_model_pooled(ds: pd.DataFrame, min_train_days: int = 260) -> TrainResult:
    # Time-based split: use oldest 85% for train, next 10% for validation, last 5% for holdout metrics
    ds = ds.copy()
    # Use index order as proxy for time within each ticker; pooled is approximate but okay
    n = len(ds)
    if n < min_train_days:
        raise ValueError("Not enough pooled rows to train.")

    cutoff_train = int(n * 0.85)
    cutoff_valid = int(n * 0.95)

    X = ds[FEATURE_COLS]
    y = ds["TARGET"].astype(int)

    X_train, y_train = X.iloc[:cutoff_train], y.iloc[:cutoff_train]
    X_valid, y_valid = X.iloc[cutoff_train:cutoff_valid], y.iloc[cutoff_train:cutoff_valid]
    X_test,  y_test  = X.iloc[cutoff_valid:], y.iloc[cutoff_valid:]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    # Evaluate on holdout
    if len(X_test) > 0:
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.55).astype(int)
        precision = precision_score(y_test, preds, zero_division=0)
        coverage = float((preds == 1).mean())
    else:
        precision = 0.0
        coverage = 0.0

    dump(model, os.path.join(DATA_DIR, "model.joblib"))
    return TrainResult(model=model, cutoff_date=datetime.now().strftime('%Y-%m-%d'), precision=precision, coverage=coverage)


def predict_today(price_map: Dict[str, pd.DataFrame], model: RandomForestClassifier, proba_threshold: float, lookahead: int, min_days: int) -> pd.DataFrame:
    rows = []
    for tk, df in price_map.items():
        if df is None or df.empty or len(df) < min_days:
            continue
        df = df.copy()
        df = add_features(df)
        df.dropna(inplace=True)
        if df.empty:
            continue
        latest = df.iloc[-1]
        X_row = latest[FEATURE_COLS].values.reshape(1, -1)
        p = float(model.predict_proba(X_row)[0, 1])
        guard_bull = (latest["SMA50"] > latest["SMA200"]) and (latest["RSI14"] < 70)
        if p >= proba_threshold and guard_bull:
            rows.append({
                "ticker": tk,
                "proba_up": round(p, 4),
                "close": float(latest["Close"]),
                "rsi14": float(latest["RSI14"]),
                "sma50_over_200": float(latest["SMA50"]/latest["SMA200"] - 1.0),
                "lookahead_days": lookahead
            })
    out = pd.DataFrame(rows).sort_values("proba_up", ascending=False)
    return out


def backtest_recent(price_map: Dict[str, pd.DataFrame], model: RandomForestClassifier, proba_threshold: float, lookahead: int, min_days: int, days: int = 120) -> Tuple[float, float]:
    """Compute simple precision & coverage over the last N days using end-of-day signals.
    This is a light sanity check, not a full walk-forward backtest.
    """
    preds, trues = [], []
    for tk, df in price_map.items():
        if df is None or df.empty or len(df) < (min_days + lookahead + days):
            continue
        df = df.copy()
        df = add_features(df)
        df = make_labels(df, lookahead)
        df.dropna(inplace=True)
        # Evaluate only on last `days` rows
        eval_df = df.iloc[-days:]
        for _, row in eval_df.iterrows():
            X = row[FEATURE_COLS].values.reshape(1, -1)
            p = float(model.predict_proba(X)[0, 1])
            guard_bull = (row["SMA50"] > row["SMA200"]) and (row["RSI14"] < 70)
            pred = int(p >= proba_threshold and guard_bull)
            preds.append(pred)
            trues.append(int(row["TARGET"]))
    if not preds:
        return 0.0, 0.0
    precision = precision_score(trues, preds, zero_division=0)
    coverage = float(np.mean(preds))
    return float(precision), float(coverage)


# ----------------------------------
# Alerts (optional)
# ----------------------------------

def send_email(subject: str, body: str):
    import smtplib
    from email.mime.text import MIMEText
    host = os.getenv("EMAIL_SMTP"); port = int(os.getenv("EMAIL_PORT", "587"))
    user = os.getenv("EMAIL_USER"); pwd = os.getenv("EMAIL_PASS"); to = os.getenv("EMAIL_TO")
    if not all([host, port, user, pwd, to]):
        log("Email env vars not fully set; skipping email alert.")
        return
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = to
    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, pwd)
        server.sendmail(user, [to], msg.as_string())
    log("Email alert sent.")


def send_discord(body: str):
    import requests
    url = os.getenv("DISCORD_WEBHOOK")
    if not url:
        log("No DISCORD_WEBHOOK set; skipping Discord alert.")
        return
    try:
        resp = requests.post(url, json={"content": body}, timeout=10)
        if resp.status_code >= 200 and resp.status_code < 300:
            log("Discord alert sent.")
        else:
            log(f"Discord webhook error: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        log(f"Discord webhook exception: {e}")


# ----------------------------------
# Main
# ----------------------------------

def main():
    parser = argparse.ArgumentParser(description="S&P 500 AI Buy Screener")
    parser.add_argument("--proba", type=float, default=0.60, help="Probability threshold for buy signal")
    parser.add_argument("--lookahead", type=int, default=5, help="Days ahead to predict positive return")
    parser.add_argument("--min-days", type=int, default=260, help="Minimum history per ticker to include")
    parser.add_argument("--years", type=int, default=LOOKBACK_YEARS_DEFAULT, help="Years of history to download")
    parser.add_argument("--send-email", action="store_true", help="Send email alert if any signals")
    parser.add_argument("--send-discord", action="store_true", help="Send Discord alert if any signals")
    parser.add_argument("--save-metrics", action="store_true", help="Save backtest metrics to data/metrics.json")
    args = parser.parse_args()

    start = datetime.now() - timedelta(days=365*args.years)

    log("Loading S&P 500 tickers...")
    tickers = safe_read_html_sp500()
    log(f"Using {len(tickers)} tickers.")

    price_map: Dict[str, pd.DataFrame] = {}
    for i, tk in enumerate(tickers, start=1):
        df = fetch_history(tk, start)
        if df is None or df.empty:
            continue
        price_map[tk] = df
        if i % 50 == 0:
            log(f"Downloaded {i} / {len(tickers)} tickers...")
        time.sleep(0.05)  # polite pacing to avoid rate limits

    if not price_map:
        raise SystemExit("No data downloaded. Check your internet or yfinance limits.")

    log("Building dataset...")
    ds = build_pooled_dataset(price_map, lookahead=args.lookahead, min_days=args.min_days)
    if ds.empty:
        raise SystemExit("Dataset empty after feature engineering. Try reducing --min-days or years.")
    log(f"Dataset rows: {len(ds):,}")

    log("Training model...")
    res = train_model_pooled(ds)
    log(f"Holdout precision@0.55: {res.precision:.3f}, coverage: {res.coverage:.3f}")

    log("Predicting today...")
    buy_df = predict_today(price_map, res.model, args.proba, lookahead=args.lookahead, min_days=args.min_days)

    out_csv = os.path.join(DATA_DIR, "buy_signals.csv")
    buy_df.to_csv(out_csv, index=False)
    log(f"Saved {len(buy_df)} signals to {out_csv}")

    if args.save_metrics:
        prec, cov = backtest_recent(price_map, res.model, args.proba, lookahead=args.lookahead, min_days=args.min_days)
        metrics = {
            "as_of": datetime.now().strftime('%Y-%m-%d'),
            "precision_estimate": round(prec, 3),
            "coverage_estimate": round(cov, 3),
            "threshold": args.proba,
            "lookahead": args.lookahead
        }
        with open(os.path.join(DATA_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        log(f"Saved metrics to {os.path.join(DATA_DIR, 'metrics.json')}")

    # Alerts
    if len(buy_df) > 0:
        body_lines = [
            f"AI BUY signals ({len(buy_df)}) — {datetime.now().strftime('%Y-%m-%d')}",
            "ticker | proba | close | rsi14 | sma50-200%"
        ]
        for _, r in buy_df.iterrows():
            body_lines.append(f"{r['ticker']:5} | {r['proba_up']:.3f} | {r['close']:.2f} | {r['rsi14']:.1f} | {100*r['sma50_over_200']:.2f}%")
        body = "\n".join(body_lines)
        if args.send_email:
            send_email(subject="AI BUY signals", body=body)
        if args.send_discord:
            send_discord(body)

    log("Done.")


if __name__ == "__main__":
    main()
