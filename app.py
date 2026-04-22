# app.py
import math
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Bandarmology Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --panel: #0d1b2e;
            --panel-2: #10243b;
            --border: rgba(255,255,255,0.08);
            --text: #e8f0ff;
            --muted: #9fb0c7;
            --green: #19d38a;
            --green-soft: rgba(25, 211, 138, 0.15);
            --blue: #4da3ff;
            --blue-soft: rgba(77, 163, 255, 0.16);
            --red: #ff5d73;
            --red-soft: rgba(255, 93, 115, 0.14);
            --yellow: #f6c35b;
            --yellow-soft: rgba(246, 195, 91, 0.14);
            --shadow: 0 10px 30px rgba(0,0,0,0.30);
            --radius: 18px;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(77,163,255,0.08), transparent 24%),
                radial-gradient(circle at top right, rgba(25,211,138,0.06), transparent 26%),
                linear-gradient(180deg, #06111d 0%, #081425 100%);
            color: var(--text);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #07111f 0%, #0a1626 100%);
            border-right: 1px solid var(--border);
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        .app-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #f4f8ff;
            margin-bottom: 0.2rem;
            letter-spacing: 0.3px;
        }

        .app-subtitle {
            color: var(--muted);
            font-size: 0.95rem;
            margin-bottom: 1.2rem;
        }

        .glass-box {
            background: linear-gradient(180deg, rgba(16,36,59,0.94), rgba(10,24,40,0.96));
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        .summary-card {
            border-radius: 22px;
            padding: 1rem 1rem 0.85rem 1rem;
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            min-height: 122px;
        }

        .summary-blue {
            background: linear-gradient(180deg, rgba(77,163,255,0.18), rgba(16,29,47,0.95));
        }

        .summary-green {
            background: linear-gradient(180deg, rgba(25,211,138,0.18), rgba(16,29,47,0.95));
        }

        .summary-red {
            background: linear-gradient(180deg, rgba(255,93,115,0.18), rgba(16,29,47,0.95));
        }

        .summary-label {
            color: #d9e6ff;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .summary-value {
            font-size: 2.0rem;
            font-weight: 800;
            color: white;
            line-height: 1;
            margin-bottom: 0.45rem;
        }

        .summary-note {
            color: var(--muted);
            font-size: 0.8rem;
        }

        .toolbar-chip {
            display: inline-block;
            margin-right: 0.45rem;
            margin-bottom: 0.35rem;
            padding: 0.36rem 0.8rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.04);
            color: #d8e6ff;
            font-size: 0.8rem;
        }

        .stock-card {
            background: linear-gradient(180deg, rgba(13,27,46,0.98), rgba(9,22,37,0.98));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .row-between {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }

        .row-wrap {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            flex-wrap: wrap;
        }

        .ticker {
            color: white;
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: 0.5px;
        }

        .company {
            color: var(--muted);
            font-size: 0.83rem;
            margin-top: 0.18rem;
        }

        .badge {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 800;
            padding: 0.32rem 0.62rem;
            border-radius: 999px;
            letter-spacing: 0.3px;
        }

        .badge-blue {
            background: var(--blue-soft);
            color: #88c1ff;
            border: 1px solid rgba(77,163,255,0.24);
        }

        .badge-green {
            background: var(--green-soft);
            color: #73efba;
            border: 1px solid rgba(25,211,138,0.24);
        }

        .badge-red {
            background: var(--red-soft);
            color: #ff9baa;
            border: 1px solid rgba(255,93,115,0.24);
        }

        .badge-yellow {
            background: var(--yellow-soft);
            color: #ffd684;
            border: 1px solid rgba(246,195,91,0.24);
        }

        .score-box {
            text-align: right;
        }

        .score-label {
            color: var(--muted);
            font-size: 0.72rem;
            margin-bottom: 0.18rem;
        }

        .score-value {
            font-size: 1.4rem;
            font-weight: 800;
            color: white;
            line-height: 1.05;
        }

        .price-box {
            text-align: right;
            margin-top: 0.25rem;
        }

        .price {
            color: #f3f8ff;
            font-size: 1.1rem;
            font-weight: 750;
        }

        .pct-pos {
            color: #68ebb1;
            font-weight: 700;
            font-size: 0.86rem;
        }

        .pct-neg {
            color: #ff8d9d;
            font-weight: 700;
            font-size: 0.86rem;
        }

        .tag {
            display: inline-block;
            padding: 0.28rem 0.58rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 700;
            color: #dceaff;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            margin-right: 0.36rem;
            margin-top: 0.18rem;
        }

        .progress-wrap {
            margin-top: 0.8rem;
            margin-bottom: 0.55rem;
        }

        .progress-track {
            width: 100%;
            height: 10px;
            background: rgba(255,255,255,0.06);
            border-radius: 999px;
            overflow: hidden;
        }

        .progress-fill {
            height: 10px;
            border-radius: 999px;
            background: linear-gradient(90deg, #1ec98e 0%, #4da3ff 100%);
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.5rem;
            margin-top: 0.8rem;
        }

        .mini-box {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 0.65rem 0.7rem;
        }

        .mini-label {
            color: var(--muted);
            font-size: 0.7rem;
            margin-bottom: 0.22rem;
        }

        .mini-value {
            color: #eff5ff;
            font-size: 0.9rem;
            font-weight: 700;
        }

        .section-title {
            color: #edf4ff;
            font-size: 1.05rem;
            font-weight: 800;
            margin: 0.2rem 0 0.8rem 0;
        }

        .footer-note {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 0.8rem;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(13,27,46,0.97), rgba(10,24,40,0.96));
            border: 1px solid rgba(255,255,255,0.07);
            padding: 0.8rem;
            border-radius: 18px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            height: 46px;
            border-radius: 14px;
            padding-left: 18px;
            padding-right: 18px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            color: #d6e4ff;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(180deg, rgba(25,211,138,0.22), rgba(77,163,255,0.12));
            border: 1px solid rgba(25,211,138,0.28);
            color: white;
        }

        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
            background: rgba(255,255,255,0.03) !important;
            color: #eaf2ff !important;
            border-radius: 12px !important;
        }

        .stDataFrame, div[data-testid="stTable"] {
            border-radius: 18px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# STATIC DATA
# =========================================================
IDX_STOCKS = [
    ("BBCA", "Bank Central Asia Tbk"),
    ("BBRI", "Bank Rakyat Indonesia Tbk"),
    ("BMRI", "Bank Mandiri Tbk"),
    ("BBNI", "Bank Negara Indonesia Tbk"),
    ("TLKM", "Telkom Indonesia Tbk"),
    ("ASII", "Astra International Tbk"),
    ("UNTR", "United Tractors Tbk"),
    ("ADRO", "Alamtri Resources Indonesia Tbk"),
    ("ANTM", "Aneka Tambang Tbk"),
    ("MDKA", "Merdeka Copper Gold Tbk"),
    ("INCO", "Vale Indonesia Tbk"),
    ("AMMN", "Amman Mineral Internasional Tbk"),
    ("GOTO", "GoTo Gojek Tokopedia Tbk"),
    ("BRIS", "Bank Syariah Indonesia Tbk"),
    ("CPIN", "Charoen Pokphand Indonesia Tbk"),
    ("ICBP", "Indofood CBP Sukses Makmur Tbk"),
    ("INDF", "Indofood Sukses Makmur Tbk"),
    ("JPFA", "Japfa Comfeed Indonesia Tbk"),
    ("KLBF", "Kalbe Farma Tbk"),
    ("MIKA", "Mitra Keluarga Karyasehat Tbk"),
    ("SIDO", "Industri Jamu dan Farmasi Sido Muncul Tbk"),
    ("HRUM", "Harum Energy Tbk"),
    ("ITMG", "Indo Tambangraya Megah Tbk"),
    ("PTBA", "Bukit Asam Tbk"),
    ("MEDC", "Medco Energi Internasional Tbk"),
    ("PGAS", "Perusahaan Gas Negara Tbk"),
    ("AKRA", "AKR Corporindo Tbk"),
    ("ERAA", "Erajaya Swasembada Tbk"),
    ("EXCL", "XL Axiata Tbk"),
    ("ISAT", "Indosat Tbk"),
    ("SMGR", "Semen Indonesia Tbk"),
    ("WIKA", "Wijaya Karya Tbk"),
    ("ADHI", "Adhi Karya Tbk"),
    ("PTPP", "PP Persero Tbk"),
    ("SSMS", "Sawit Sumbermas Sarana Tbk"),
    ("AALI", "Astra Agro Lestari Tbk"),
    ("LSIP", "PP London Sumatra Indonesia Tbk"),
    ("BBTN", "Bank Tabungan Negara Tbk"),
    ("MAPI", "Mitra Adiperkasa Tbk"),
    ("ACES", "Aspirasi Hidup Indonesia Tbk"),
    ("AUTO", "Astra Otoparts Tbk"),
    ("AVIA", "Avia Avian Tbk"),
    ("BIRD", "Blue Bird Tbk"),
    ("ESSA", "ESSA Industries Indonesia Tbk"),
    ("HEAL", "Medikaloka Hermina Tbk"),
    ("TOWR", "Sarana Menara Nusantara Tbk"),
]


BROKERS = ["YP", "CC", "RX", "PD", "NI", "AK", "YU", "ZP", "BK", "DH"]


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def load_data(period_label: str = "1 Hari", use_dummy: bool = True) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Create dummy OHLCV data and pseudo flow/orderbook data.
    This fallback keeps the app functional even without live APIs.
    """
    if not use_dummy:
        # Placeholder for future real integration
        # Example:
        # - IDX data provider
        # - yfinance fallback
        # - broker summary API
        # - foreign flow API
        pass

    seed_map = {"1 Hari": 11, "5 Hari": 22, "1 Bulan": 33}
    rng = np.random.default_rng(seed_map.get(period_label, 11))

    num_days = 240
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=num_days)

    rows = []
    ohlcv_map: Dict[str, pd.DataFrame] = {}

    for i, (ticker, company) in enumerate(IDX_STOCKS):
        base_price = rng.integers(50, 9500)
        drift = rng.uniform(-0.0008, 0.0028)
        vol = rng.uniform(0.012, 0.04)

        rets = rng.normal(drift, vol, num_days)
        close = base_price * np.exp(np.cumsum(rets))
        close = np.clip(close, 50, None)

        open_ = close * (1 + rng.normal(0, 0.01, num_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.01, 0.007, num_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.01, 0.007, num_days)))
        volume = rng.integers(2_000_000, 150_000_000, num_days).astype(float)

        df = pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        ).reset_index(drop=True)

        # Simulated external-like fields
        df["net_foreign"] = rng.normal(0, 8_000_000_000, num_days)
        df["broker_net"] = rng.normal(0, 9_000_000_000, num_days)
        df["bid_volume"] = rng.integers(500_000, 6_000_000, num_days)
        df["offer_volume"] = rng.integers(500_000, 6_000_000, num_days)
        df["big_bid_wall"] = rng.integers(0, 2, num_days)
        df["big_offer_wall"] = rng.integers(0, 2, num_days)

        ohlcv_map[ticker] = df

        rows.append(
            {
                "ticker": ticker,
                "company": company,
                "last_price": float(df["close"].iloc[-1]),
                "prev_price": float(df["close"].iloc[-2]),
                "last_volume": float(df["volume"].iloc[-1]),
                "avg_volume_20": float(df["volume"].tail(20).mean()),
                "foreign_5d": float(df["net_foreign"].tail(5).sum()),
                "foreign_10d": float(df["net_foreign"].tail(10).sum()),
                "foreign_20d": float(df["net_foreign"].tail(20).sum()),
                "broker_5d": float(df["broker_net"].tail(5).sum()),
                "broker_10d": float(df["broker_net"].tail(10).sum()),
                "broker_20d": float(df["broker_net"].tail(20).sum()),
                "bid_volume": float(df["bid_volume"].iloc[-1]),
                "offer_volume": float(df["offer_volume"].iloc[-1]),
                "big_bid_wall": int(df["big_bid_wall"].iloc[-1]),
                "big_offer_wall": int(df["big_offer_wall"].iloc[-1]),
            }
        )

    master_df = pd.DataFrame(rows)
    master_df["pct_change"] = ((master_df["last_price"] / master_df["prev_price"]) - 1.0) * 100.0
    master_df["volume_ratio"] = master_df["last_volume"] / master_df["avg_volume_20"].replace(0, np.nan)

    return master_df, ohlcv_map


# =========================================================
# INDICATOR HELPERS
# =========================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd = ema12 - ema26
    signal = ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def compute_bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ma, upper, lower


def detect_hammer(open_: float, high: float, low: float, close: float) -> bool:
    body = abs(close - open_)
    candle_range = max(high - low, 1e-9)
    lower_shadow = min(open_, close) - low
    upper_shadow = high - max(open_, close)
    return (lower_shadow > body * 2.0) and (upper_shadow < body * 1.2) and (body / candle_range < 0.4)


def detect_bullish_engulfing(prev_open: float, prev_close: float, open_: float, close: float) -> bool:
    prev_bearish = prev_close < prev_open
    curr_bullish = close > open_
    body_engulf = (close >= prev_open) and (open_ <= prev_close)
    return prev_bearish and curr_bullish and body_engulf


# =========================================================
# ANALYZERS
# =========================================================
def technical_analyzer(stock_row: pd.Series, hist: pd.DataFrame) -> Dict[str, float]:
    close = hist["close"]
    volume = hist["volume"]

    rsi = float(compute_rsi(close).iloc[-1])
    macd, signal, hist_macd = compute_macd(close)
    macd_val = float(macd.iloc[-1])
    signal_val = float(signal.iloc[-1])
    hist_val = float(hist_macd.iloc[-1])

    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])

    _, bb_upper, bb_lower = compute_bollinger(close)
    last_close = float(close.iloc[-1])
    last_bb_upper = float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else last_close
    last_bb_lower = float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else last_close

    vol_ratio = float(volume.iloc[-1] / max(volume.tail(20).mean(), 1))

    score = 0.0

    # RSI
    if 50 <= rsi <= 65:
        score += 9
    elif 45 <= rsi < 50 or 65 < rsi <= 72:
        score += 6
    elif 35 <= rsi < 45:
        score += 3
    else:
        score += 1

    # MACD
    if macd_val > signal_val and hist_val > 0:
        score += 9
    elif macd_val > signal_val:
        score += 7
    elif hist_val > 0:
        score += 5
    else:
        score += 2

    # MA alignment
    if ma20 > ma50 > ma200:
        score += 9
    elif ma20 > ma50:
        score += 7
    elif last_close > ma20:
        score += 4
    else:
        score += 1

    # Bollinger + trend positioning
    if last_close > last_bb_upper:
        score += 4
    elif last_close > ma20:
        score += 6
    elif last_close >= last_bb_lower:
        score += 4
    else:
        score += 2

    # Volume spike
    if vol_ratio >= 2.0:
        score += 8
    elif vol_ratio >= 1.5:
        score += 6
    elif vol_ratio >= 1.2:
        score += 4
    else:
        score += 2

    return {
        "technical_score": min(score, 35.0),
        "rsi": rsi,
        "macd": macd_val,
        "macd_signal": signal_val,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "bb_upper": last_bb_upper,
        "bb_lower": last_bb_lower,
        "volume_ratio": vol_ratio,
    }


def flow_analyzer(stock_row: pd.Series, hist: pd.DataFrame) -> Dict[str, float]:
    foreign_5d = float(stock_row["foreign_5d"])
    foreign_10d = float(stock_row["foreign_10d"])
    foreign_20d = float(stock_row["foreign_20d"])
    broker_5d = float(stock_row["broker_5d"])
    broker_10d = float(stock_row["broker_10d"])
    broker_20d = float(stock_row["broker_20d"])

    score = 0.0

    # Foreign trend
    foreign_positive_count = sum(v > 0 for v in [foreign_5d, foreign_10d, foreign_20d])
    if foreign_positive_count == 3:
        score += 12
    elif foreign_positive_count == 2:
        score += 8
    elif foreign_positive_count == 1:
        score += 4
    else:
        score += 1

    # Broker accumulation
    broker_positive_count = sum(v > 0 for v in [broker_5d, broker_10d, broker_20d])
    if broker_positive_count == 3:
        score += 12
    elif broker_positive_count == 2:
        score += 8
    elif broker_positive_count == 1:
        score += 4
    else:
        score += 1

    # Real accumulation logic
    price_up = stock_row["pct_change"] > 0
    vol_ratio = stock_row["volume_ratio"]
    if broker_5d > 0 and foreign_5d > 0 and price_up and vol_ratio > 1.2:
        score += 11
        accumulation_state = "Strong Accumulation"
    elif broker_5d > 0 and price_up:
        score += 8
        accumulation_state = "Broker Supported"
    elif foreign_5d > 0:
        score += 6
        accumulation_state = "Foreign Supported"
    else:
        score += 2
        accumulation_state = "Weak / Mixed"

    return {
        "flow_score": min(score, 35.0),
        "foreign_5d": foreign_5d,
        "foreign_10d": foreign_10d,
        "foreign_20d": foreign_20d,
        "broker_5d": broker_5d,
        "broker_10d": broker_10d,
        "broker_20d": broker_20d,
        "accumulation_state": accumulation_state,
    }


def orderbook_analyzer(stock_row: pd.Series, hist: pd.DataFrame) -> Dict[str, float]:
    bid_volume = float(stock_row["bid_volume"])
    offer_volume = float(stock_row["offer_volume"])
    big_bid_wall = int(stock_row["big_bid_wall"])
    big_offer_wall = int(stock_row["big_offer_wall"])

    imbalance = bid_volume / max(offer_volume, 1)
    pressure = bid_volume - offer_volume

    score = 0.0

    if imbalance >= 1.5:
        score += 7
    elif imbalance >= 1.15:
        score += 5
    elif imbalance >= 1.0:
        score += 3
    else:
        score += 1

    if big_bid_wall and not big_offer_wall:
        score += 5
    elif big_bid_wall and big_offer_wall:
        score += 3
    elif big_offer_wall:
        score += 1
    else:
        score += 2

    if pressure > 0:
        score += 3
    else:
        score += 1

    return {
        "orderbook_score": min(score, 15.0),
        "bid_offer_imbalance": imbalance,
        "big_bid_wall": big_bid_wall,
        "big_offer_wall": big_offer_wall,
        "order_pressure": pressure,
    }


def pattern_analyzer(stock_row: pd.Series, hist: pd.DataFrame) -> Dict[str, float]:
    score = 0.0

    last = hist.iloc[-1]
    prev = hist.iloc[-2]

    hammer = detect_hammer(last["open"], last["high"], last["low"], last["close"])
    engulfing = detect_bullish_engulfing(prev["open"], prev["close"], last["open"], last["close"])

    resistance_20 = float(hist["high"].tail(20).max())
    support_20 = float(hist["low"].tail(20).min())
    breakout = float(last["close"]) >= (resistance_20 * 0.995)
    bounce_support = float(last["close"]) > support_20 * 1.03

    vol_ratio = float(hist["volume"].iloc[-1] / max(hist["volume"].tail(20).mean(), 1))

    if hammer:
        score += 4
    if engulfing:
        score += 4
    if breakout:
        score += 4
    elif bounce_support:
        score += 2

    if vol_ratio >= 1.5:
        score += 3
    elif vol_ratio >= 1.2:
        score += 2
    else:
        score += 1

    return {
        "pattern_score": min(score, 15.0),
        "hammer": hammer,
        "bullish_engulfing": engulfing,
        "breakout": breakout,
        "support_20": support_20,
        "resistance_20": resistance_20,
    }


# =========================================================
# SCORE + CLASSIFICATION
# =========================================================
def calculate_total_score(tech: Dict[str, float], flow: Dict[str, float], orderbook: Dict[str, float], pattern: Dict[str, float]) -> float:
    total = tech["technical_score"] + flow["flow_score"] + orderbook["orderbook_score"] + pattern["pattern_score"]
    return float(max(0, min(100, round(total, 2))))


def classify_signal(score: float) -> str:
    if score >= 80:
        return "Akumulasi Kuat"
    if score >= 65:
        return "Partisipasi"
    if score >= 50:
        return "Watchlist"
    return "Distribusi"


def signal_badge_class(signal_name: str) -> str:
    if signal_name == "Akumulasi Kuat":
        return "badge-blue"
    if signal_name == "Partisipasi":
        return "badge-green"
    if signal_name == "Watchlist":
        return "badge-yellow"
    return "badge-red"


def score_to_strength_width(score: float) -> float:
    return max(4.0, min(score, 100.0))


def human_money(x: float) -> str:
    sign = "+" if x > 0 else ""
    abs_x = abs(x)
    if abs_x >= 1_000_000_000_000:
        return f"{sign}{x/1_000_000_000_000:.2f}T"
    if abs_x >= 1_000_000_000:
        return f"{sign}{x/1_000_000_000:.2f}B"
    if abs_x >= 1_000_000:
        return f"{sign}{x/1_000_000:.2f}M"
    return f"{sign}{x:,.0f}"


def build_tags(row: pd.Series) -> List[str]:
    tags = []
    if row["volume_ratio"] >= 1.5:
        tags.append("Unusual Vol")
    if row["last_price"] > row["ma20"] and row["ma20"] > row["ma50"]:
        tags.append("Crossing")
    if row["last_price"] <= 500:
        tags.append("Penny")
    if row["avg_volume_20"] >= 8_000_000:
        tags.append("Liquid")
    if row["hammer"]:
        tags.append("Hammer")
    if row["bullish_engulfing"]:
        tags.append("Engulfing")
    if row["breakout"]:
        tags.append("Breakout")
    return tags[:5]


def assign_top_brokers(seed_text: str) -> Tuple[str, str]:
    base = sum(ord(c) for c in seed_text)
    buy = BROKERS[base % len(BROKERS)]
    sell = BROKERS[(base + 3) % len(BROKERS)]
    return buy, sell


# =========================================================
# RENDERING
# =========================================================
def render_header() -> None:
    st.markdown('<div class="app-title">Bandarmology Screener</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="app-subtitle">Stock screener dark-theme premium style • Last update: {datetime.now().strftime("%d %b %Y %H:%M:%S")}</div>',
        unsafe_allow_html=True,
    )


def render_summary_cards(df: pd.DataFrame) -> None:
    akumulasi = int((df["signal"] == "Akumulasi Kuat").sum())
    partisipasi = int((df["signal"] == "Partisipasi").sum())
    distribusi = int((df["signal"] == "Distribusi").sum())

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="summary-card summary-blue">
                <div class="summary-label">Akumulasi</div>
                <div class="summary-value">{akumulasi}</div>
                <div class="summary-note">Skor 80–100 • Bandar accumulation bias</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="summary-card summary-green">
                <div class="summary-label">Partisipasi</div>
                <div class="summary-value">{partisipasi}</div>
                <div class="summary-note">Skor 65–79 • Flow mulai aktif</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="summary-card summary-red">
                <div class="summary-label">Distribusi</div>
                <div class="summary-value">{distribusi}</div>
                <div class="summary-note">Skor &lt; 50 • Hindari saat ini</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_toolbar_info(period_label: str, filtered_count: int) -> None:
    st.markdown(
        f"""
        <div class="glass-box">
            <span class="toolbar-chip">Periode: {period_label}</span>
            <span class="toolbar-chip">Jumlah Emiten: {filtered_count}</span>
            <span class="toolbar-chip">Mode: Dummy Fallback Active</span>
            <span class="toolbar-chip">Scoring: Tech 35 • Flow 35 • Orderbook 15 • Pattern 15</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stock_cards(df: pd.DataFrame, limit: int = 12) -> None:
    st.markdown("### Top Ranked Signals")

    for _, row in df.head(limit).iterrows():
        top_buy, top_sell = assign_top_brokers(row["ticker"])

        with st.container():
            c1, c2 = st.columns([3, 1])

            with c1:
                st.markdown(f"## {row['ticker']}  •  {row['signal']}")
                st.caption(row["company"])

                tags = row["tags"] if isinstance(row["tags"], list) else []
                if tags:
                    st.write(" | ".join(tags))

            with c2:
                st.metric("Score", f"{row['total_score']:.0f}", f"{row['pct_change']:+.2f}%")
                st.write(f"Price: {row['last_price']:,.0f}")

            st.progress(min(max(int(row["total_score"]), 0), 100))

            a, b, c = st.columns(3)
            a.metric("Net Akumulasi", human_money(row["broker_5d"]))
            b.metric("Top Buy / Sell", f"{top_buy} / {top_sell}")
            c.metric("Vol Ratio", f"{row['volume_ratio']:.2f}x")

            st.divider()


def render_chart(hist: pd.DataFrame, ticker: str) -> None:
    chart_df = hist.tail(80).copy()

    ma20 = chart_df["close"].rolling(20).mean()
    ma50 = chart_df["close"].rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=chart_df["date"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="OHLC",
        )
    )
    fig.add_trace(go.Scatter(x=chart_df["date"], y=ma20, mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=chart_df["date"], y=ma50, mode="lines", name="MA50"))

    fig.update_layout(
        title=f"{ticker} Price Structure",
        height=440,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_table(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Screening Table</div>', unsafe_allow_html=True)

    show_cols = [
        "rank",
        "ticker",
        "company",
        "signal",
        "total_score",
        "last_price",
        "pct_change",
        "technical_score",
        "flow_score",
        "orderbook_score",
        "pattern_score",
        "rsi",
        "volume_ratio",
        "bid_offer_imbalance",
        "foreign_5d",
        "broker_5d",
    ]
    table_df = df[show_cols].copy()

    numeric_cols_2 = [
        "total_score",
        "pct_change",
        "technical_score",
        "flow_score",
        "orderbook_score",
        "pattern_score",
        "rsi",
        "volume_ratio",
        "bid_offer_imbalance",
    ]
    for col in numeric_cols_2:
        table_df[col] = pd.to_numeric(table_df[col], errors="coerce").round(2)

    for money_col in ["foreign_5d", "broker_5d"]:
        table_df[money_col] = table_df[money_col].apply(human_money)

    st.dataframe(table_df, use_container_width=True, hide_index=True)


# =========================================================
# MAIN APP LOGIC
# =========================================================
def prepare_scored_dataframe(base_df: pd.DataFrame, ohlcv_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    enriched_rows = []

    for _, row in base_df.iterrows():
        hist = ohlcv_map[row["ticker"]]

        tech = technical_analyzer(row, hist)
        flow = flow_analyzer(row, hist)
        orderbook = orderbook_analyzer(row, hist)
        pattern = pattern_analyzer(row, hist)

        total_score = calculate_total_score(tech, flow, orderbook, pattern)
        signal = classify_signal(total_score)

        merged = row.to_dict()
        merged.update(tech)
        merged.update(flow)
        merged.update(orderbook)
        merged.update(pattern)
        merged["total_score"] = total_score
        merged["signal"] = signal

        enriched_rows.append(merged)

    df = pd.DataFrame(enriched_rows)
    df["tags"] = df.apply(build_tags, axis=1)
    df = df.sort_values(["total_score", "pct_change", "volume_ratio"], ascending=[False, False, False]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def main() -> None:
    inject_css()
    render_header()

    st.sidebar.markdown("## Filters")

    auto_refresh = st.sidebar.checkbox("Auto refresh every 60 seconds", value=False)
    use_dummy = st.sidebar.toggle("Use dummy fallback data", value=True)

    # Tabs-like period selector
    tab1, tab2, tab3 = st.tabs(["1 Hari", "5 Hari", "1 Bulan"])
    selected_period = "1 Hari"
    with tab1:
        st.caption("Mode cepat untuk snapshot harian.")
        selected_period = "1 Hari"
    with tab2:
        st.caption("Lihat kecenderungan dalam 5 hari.")
    with tab3:
        st.caption("Lihat bias 1 bulan.")

    period_choice = st.sidebar.selectbox("Pilih periode", ["1 Hari", "5 Hari", "1 Bulan"], index=0)

    # Data load
    base_df, ohlcv_map = load_data(period_label=period_choice, use_dummy=use_dummy)
    df = prepare_scored_dataframe(base_df, ohlcv_map)

    # Sidebar filters
    min_price = st.sidebar.number_input("Minimum harga", min_value=0.0, value=0.0, step=50.0)
    max_price = st.sidebar.number_input("Maksimum harga", min_value=0.0, value=float(max(1000.0, df["last_price"].max())), step=50.0)
    min_volume = st.sidebar.number_input("Minimum volume", min_value=0.0, value=0.0, step=1000000.0)
    min_score = st.sidebar.slider("Minimum score", min_value=0, max_value=100, value=50)
    signal_filter = st.sidebar.multiselect(
        "Kategori sinyal",
        options=["Akumulasi Kuat", "Partisipasi", "Watchlist", "Distribusi"],
        default=["Akumulasi Kuat", "Partisipasi", "Watchlist"],
    )
    watchlist_only = st.sidebar.checkbox("Watchlist only", value=False)

    default_watchlist = ["BBCA", "BBRI", "BMRI", "TLKM", "ASII", "ANTM", "ADRO", "GOTO", "BRIS", "MDKA"]
    search_query = st.text_input("Cari emiten / nama perusahaan", placeholder="Contoh: BBCA, bank, antm, goto")

    if watchlist_only:
        df = df[df["ticker"].isin(default_watchlist)].copy()

    filtered = df[
        (df["last_price"] >= min_price)
        & (df["last_price"] <= max_price)
        & (df["last_volume"] >= min_volume)
        & (df["total_score"] >= min_score)
        & (df["signal"].isin(signal_filter))
    ].copy()

    if search_query.strip():
        q = search_query.strip().lower()
        filtered = filtered[
            filtered["ticker"].str.lower().str.contains(q)
            | filtered["company"].str.lower().str.contains(q)
        ].copy()

    render_summary_cards(filtered if not filtered.empty else df)
    render_toolbar_info(period_choice, len(filtered))

    # Quick metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        avg_score = filtered["total_score"].mean() if not filtered.empty else 0
        st.metric("Avg Score", f"{avg_score:.1f}")
    with c2:
        top_pct = filtered["pct_change"].max() if not filtered.empty else 0
        st.metric("Top % Change", f"{top_pct:+.2f}%")
    with c3:
        avg_vol_ratio = filtered["volume_ratio"].mean() if not filtered.empty else 0
        st.metric("Avg Vol Ratio", f"{avg_vol_ratio:.2f}x")
    with c4:
        strongest_flow = filtered["flow_score"].max() if not filtered.empty else 0
        st.metric("Top Flow Score", f"{strongest_flow:.0f}")

    left, right = st.columns([1.2, 1.0])

    with left:
        if filtered.empty:
            st.warning("Tidak ada saham yang cocok dengan filter saat ini.")
        else:
            render_stock_cards(filtered, limit=12)

    with right:
        st.markdown('<div class="section-title">Selected Stock Chart</div>', unsafe_allow_html=True)
        selected_ticker = st.selectbox("Pilih saham untuk chart", options=filtered["ticker"].tolist() if not filtered.empty else df["ticker"].tolist())
        render_chart(ohlcv_map[selected_ticker], selected_ticker)

        st.markdown(
            """
            <div class="glass-box">
                <div class="section-title">Analyzer Engine Notes</div>
                <div class="footer-note">
                    technical_analyzer: RSI, MACD, MA cross 20/50/200, Bollinger Band, volume spike.<br><br>
                    flow_analyzer: net foreign buy 5D/10D/20D, broker accumulation, akumulasi detection.<br><br>
                    orderbook_analyzer: bid/offer imbalance, big wall, order pressure.<br><br>
                    pattern_analyzer: hammer, bullish engulfing, support/resistance breakout, volume confirmation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_table(filtered if not filtered.empty else df)

    st.markdown(
        """
        <div class="glass-box">
            <div class="section-title">Future API Integration Placeholders</div>
            <div class="footer-note">
                You can replace dummy data inside <b>load_data()</b> with:
                <br>• IDX OHLCV feed / broker summary source
                <br>• foreign flow API
                <br>• orderbook / depth API
                <br>• yfinance or paid market data provider
                <br><br>
                Keep the analyzer functions unchanged, and only swap the incoming dataset schema as needed.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if auto_refresh:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
