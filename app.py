import os
import time
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="IDX High Prob Screener",
    page_icon="📈",
    layout="wide",
)

# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #08111f 0%, #0c1728 100%);
        color: #eaf2ff;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 98rem;
    }
    .main-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f4f7ff;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        color: #96a7c2;
        margin-bottom: 1rem;
    }
    .metric-card {
        border-radius: 18px;
        padding: 16px 18px;
        color: white;
        box-shadow: 0 10px 24px rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-green {
        background: linear-gradient(135deg, #0f9d58 0%, #0a7d43 100%);
    }
    .metric-red {
        background: linear-gradient(135deg, #c62828 0%, #8e1e1e 100%);
    }
    .metric-yellow {
        background: linear-gradient(135deg, #d4a017 0%, #9a7410 100%);
    }
    .metric-blue {
        background: linear-gradient(135deg, #1646ff 0%, #1137c4 100%);
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.95;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.25;
        margin-top: 0.15rem;
    }
    .small-note {
        color: #8fa6ca;
        font-size: 0.82rem;
    }
    .telegram-ok {
        padding: 10px 14px;
        border-radius: 12px;
        background: rgba(15, 157, 88, 0.18);
        border: 1px solid rgba(15, 157, 88, 0.4);
        color: #b7f7d2;
        margin-top: 6px;
        margin-bottom: 8px;
    }
    .telegram-warn {
        padding: 10px 14px;
        border-radius: 12px;
        background: rgba(212, 160, 23, 0.16);
        border: 1px solid rgba(212, 160, 23, 0.35);
        color: #ffe8a3;
        margin-top: 6px;
        margin-bottom: 8px;
    }
    .telegram-err {
        padding: 10px 14px;
        border-radius: 12px;
        background: rgba(198, 40, 40, 0.16);
        border: 1px solid rgba(198, 40, 40, 0.38);
        color: #ffc0c0;
        margin-top: 6px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# TELEGRAM HELPERS
# =========================
def get_secret_or_env(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


DEFAULT_BOT_TOKEN = get_secret_or_env("TELEGRAM_BOT_TOKEN", "")
DEFAULT_CHAT_ID = get_secret_or_env("TELEGRAM_CHAT_ID", "")


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> tuple[bool, str]:
    if not bot_token or not chat_id:
        return False, "Bot token / chat id kosong"
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(
            url,
            data={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return True, "Notifikasi Telegram terkirim"
        return False, f"HTTP {resp.status_code}: {resp.text[:250]}"
    except Exception as e:
        return False, str(e)


# =========================
# UTILITIES
# =========================
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def fmt_price(x):
    if pd.isna(x):
        return "-"
    if x >= 1000:
        return f"{x:,.0f}"
    if x >= 100:
        return f"{x:,.1f}"
    return f"{x:,.2f}"


def fmt_pct(x):
    if pd.isna(x):
        return "-"
    return f"{x:.2f}%"


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calc_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def detect_crossover(a_prev, a_now, b_prev, b_now) -> str:
    if pd.isna(a_prev) or pd.isna(a_now) or pd.isna(b_prev) or pd.isna(b_now):
        return "NONE"
    if a_prev <= b_prev and a_now > b_now:
        return "BULL"
    if a_prev >= b_prev and a_now < b_now:
        return "BEAR"
    return "NONE"


def detect_candles(df: pd.DataFrame) -> tuple[bool, bool]:
    if len(df) < 2:
        return False, False
    prev = df.iloc[-2]
    curr = df.iloc[-1]

    # Hammer
    body = abs(curr["Close"] - curr["Open"])
    rng = max(curr["High"] - curr["Low"], 1e-9)
    lower_shadow = min(curr["Open"], curr["Close"]) - curr["Low"]
    upper_shadow = curr["High"] - max(curr["Open"], curr["Close"])
    hammer = (lower_shadow > 2 * body) and (upper_shadow < body) and (body / rng < 0.35)

    # Bullish engulfing
    prev_red = prev["Close"] < prev["Open"]
    curr_green = curr["Close"] > curr["Open"]
    engulf = (
        prev_red
        and curr_green
        and curr["Open"] <= prev["Close"]
        and curr["Close"] >= prev["Open"]
    )
    return hammer, engulf


def phase_label(price, ma20, ma50, rsi):
    if pd.isna(price) or pd.isna(ma20) or pd.isna(ma50) or pd.isna(rsi):
        return "WAIT"
    if price > ma20 > ma50 and rsi >= 55:
        return "BULLISH"
    if price < ma20 < ma50 and rsi < 45:
        return "BEARISH"
    return "SIDEWAYS"


def price_action_day(close_now, close_prev, low_now, high_now):
    if any(pd.isna(x) for x in [close_now, close_prev, low_now, high_now]):
        return "-"
    chg = ((close_now - close_prev) / close_prev) * 100 if close_prev else 0
    candle_range = max(high_now - low_now, 1e-9)
    lower_tail = (min(close_now, close_prev) - low_now) / candle_range
    if chg > 2:
        return "NAIK KUAT"
    if 0.5 < chg <= 2:
        return "NAIK"
    if -0.5 <= chg <= 0.5:
        return "STABIL"
    if lower_tail > 0.35:
        return "TURUN, EKOR BAWAH"
    return "TURUN"


def classify_setup(price, ma20, ma50, ma200, bb_upper, bb_lower, macd_now, signal_now, rsi_now):
    if pd.isna(price):
        return "WAIT"
    if price > ma20 > ma50 and ma20 > ma200 and macd_now > signal_now and rsi_now >= 58:
        return "BREAKOUT"
    if price > ma20 and price > ma50 and 48 <= rsi_now <= 62:
        return "REBOUND"
    if ma20 > ma50 and price > ma20 and rsi_now > 55:
        return "GC ONLY"
    if price < ma20 and macd_now < signal_now:
        return "DEAD CROSS"
    if price >= bb_upper:
        return "OVERHEAT"
    if price <= bb_lower:
        return "OVERSOLD"
    return "WAIT"


def classify_action(score, price, ma20, macd_now, signal_now):
    if pd.isna(price) or pd.isna(ma20):
        return "WAIT"
    if score >= 7 and price > ma20 and macd_now >= signal_now:
        return "ENTRY ONLY"
    if score >= 5:
        return "ON TRACK"
    if price < ma20 and macd_now < signal_now:
        return "EXIT"
    return "WAIT"


def signal_text(score, phase, action, rsi, macd_now, signal_now):
    if action == "ENTRY ONLY" and phase == "BULLISH":
        return "🔥 ENTRY SIGNAL"
    if action == "ON TRACK" and phase == "BULLISH":
        return "✅ ON TRACK - OK"
    if action == "EXIT" or (phase == "BEARISH" and macd_now < signal_now and rsi < 45):
        return "❌ EXIT / DEAD CROSS"
    if score >= 5:
        return "⚠️ HOLD - WASPADA"
    return "⏳ WAIT"


def build_tp_sl(price, atr):
    if pd.isna(price):
        return np.nan, np.nan, np.nan, np.nan
    atr = 0 if pd.isna(atr) else atr
    tp1 = price * 1.03
    tp2 = price * 1.06
    tp3 = price * 1.10
    trail_sl = max(price * 0.97, price - atr)
    return tp1, tp2, tp3, trail_sl


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# =========================
# DEFAULT TICKERS
# =========================
DEFAULT_TICKERS = [
    "AALI.JK","ACES.JK","ADRO.JK","AKRA.JK","AMMN.JK","AMRT.JK","ANTM.JK","ASII.JK","ASSA.JK","AVIA.JK",
    "BBCA.JK","BBNI.JK","BBRI.JK","BBTN.JK","BMRI.JK","BRIS.JK","BRMS.JK","CPIN.JK","CTRA.JK","DEWA.JK",
    "DOID.JK","ELSA.JK","EMTK.JK","ENRG.JK","ERAA.JK","ESSA.JK","EXCL.JK","GOTO.JK","HEAL.JK","HRUM.JK",
    "ICBP.JK","INCO.JK","INDF.JK","INDY.JK","INKP.JK","INTP.JK","ISAT.JK","ITMG.JK","JPFA.JK","JSMR.JK",
    "KLBF.JK","LPPF.JK","LSIP.JK","MAPI.JK","MBMA.JK","MDIA.JK","MDKA.JK","MEDC.JK","MIKA.JK","MNCN.JK",
    "MTEL.JK","PGAS.JK","PTBA.JK","PTPP.JK","PWON.JK","SIDO.JK","SMGR.JK","SMRA.JK","TLKM.JK","TOWR.JK",
    "UNTR.JK","UNVR.JK","WIKA.JK","WMUU.JK","ZATA.JK","BIPI.JK","BNBR.JK","BULL.JK","BUMI.JK","COAL.JK",
    "INET.JK","KETR.JK","KOTA.JK","NICL.JK","PYFA.JK","SRTG.JK","TINS.JK","TKIM.JK","BBYB.JK",
]
DEFAULT_TICKERS = sorted(list(set(DEFAULT_TICKERS)))


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def get_all_idx_tickers() -> list[str]:
    urls = [
        "https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham/",
        "https://www.idx.co.id/en/market-data/stocks-data/stock-list",
    ]
    collected = set()

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
    }

    for url in urls:
        try:
            tables = pd.read_html(url, storage_options=headers)
            for table in tables:
                cols = [str(c).strip().upper() for c in table.columns]
                table.columns = cols
                candidate_cols = [
                    c for c in cols
                    if c in ["KODE", "KODE SAHAM", "CODE", "STOCK CODE", "SYMBOL"]
                ]
                if not candidate_cols:
                    continue
                code_col = candidate_cols[0]
                codes = (
                    table[code_col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .str.replace(r"[^A-Z0-9]", "", regex=True)
                )
                codes = codes[codes.str.match(r"^[A-Z]{2,5}$", na=False)]
                collected.update({f"{c}.JK" for c in codes.tolist()})
        except Exception:
            continue

    if collected:
        return sorted(collected)
    return DEFAULT_TICKERS


# =========================
# DATA ANALYSIS
# =========================
@st.cache_data(ttl=120, show_spinner=False)
def analyze_ticker(ticker: str, period: str = "6mo", interval: str = "1d"):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty or len(df) < 60:
            return None

        # Flatten if MultiIndex columns appear
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        if len(df) < 60:
            return None

        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df["MA200"] = df["Close"].rolling(200).mean()
        std20 = df["Close"].rolling(20).std()
        df["BB_UP"] = df["MA20"] + 2 * std20
        df["BB_LO"] = df["MA20"] - 2 * std20
        df["RSI"] = calc_rsi(df["Close"], 14)
        df["MACD"], df["SIGNAL"], df["HIST"] = calc_macd(df["Close"])
        df["ATR"] = calc_atr(df, 14)
        df["VOL20"] = df["Volume"].rolling(20).mean()
        df["RVOL"] = df["Volume"] / df["VOL20"]

        last = df.iloc[-1]
        prev = df.iloc[-2]

        price = safe_float(last["Close"])
        close_prev = safe_float(prev["Close"])
        open_now = safe_float(last["Open"])
        high_now = safe_float(last["High"])
        low_now = safe_float(last["Low"])
        vol = safe_float(last["Volume"])
        ma20 = safe_float(last["MA20"])
        ma50 = safe_float(last["MA50"])
        ma200 = safe_float(last["MA200"])
        bb_up = safe_float(last["BB_UP"])
        bb_lo = safe_float(last["BB_LO"])
        rsi_now = safe_float(last["RSI"])
        macd_now = safe_float(last["MACD"])
        signal_now = safe_float(last["SIGNAL"])
        hist_now = safe_float(last["HIST"])
        atr_now = safe_float(last["ATR"])
        rvol = safe_float(last["RVOL"])

        hammer, engulf = detect_candles(df)
        cross20_50 = detect_crossover(prev["MA20"], last["MA20"], prev["MA50"], last["MA50"])
        macd_cross = detect_crossover(prev["MACD"], last["MACD"], prev["SIGNAL"], last["SIGNAL"])

        # Rebound / breakout logic
        highest_20_prev = df["High"].rolling(20).max().shift(1).iloc[-1]
        rebound = (price > ma20) and (prev["Close"] <= prev["MA20"] or rsi_now > 50)
        breakout = price > highest_20_prev and rvol > 1.2 if not pd.isna(highest_20_prev) else False
        accumulation = (rvol > 1.2 and price >= open_now and price > ma20)
        foreign_proxy = 1 if (rvol > 1.5 and price > prev["Close"]) else 0

        score = 0
        score += 2 if 50 <= rsi_now <= 70 else 0
        score += 2 if macd_cross == "BULL" or (macd_now > signal_now and hist_now > 0) else 0
        score += 2 if price > ma20 and price > ma50 else 0
        score += 1 if rvol > 1.5 else 0
        score += 2 if breakout or rebound or engulf or hammer or cross20_50 == "BULL" else 0
        score += 1 if foreign_proxy else 0

        phase = phase_label(price, ma20, ma50, rsi_now)
        setup = classify_setup(price, ma20, ma50, ma200, bb_up, bb_lo, macd_now, signal_now, rsi_now)
        action = classify_action(score, price, ma20, macd_now, signal_now)
        signal = signal_text(score, phase, action, rsi_now, macd_now, signal_now)

        day_text = price_action_day(price, close_prev, low_now, high_now)
        gain = ((price - close_prev) / close_prev * 100) if close_prev else np.nan
        entry_zone = ma20 if not pd.isna(ma20) else price
        tp1, tp2, tp3, trail_sl = build_tp_sl(price, atr_now)
        profit_pct = ((tp1 - price) / price * 100) if price else np.nan
        s1 = price * 0.96 if not pd.isna(price) else np.nan
        r1 = price * 1.04 if not pd.isna(price) else np.nan

        regime = "Bullish" if phase == "BULLISH" else "Bearish" if phase == "BEARISH" else "Neutral"

        return {
            "Ticker": ticker.replace(".JK", ""),
            "Regime": regime,
            "FASE": phase,
            "SETUP": setup,
            "AKSI": action,
            "GAIN": gain,
            "DAY": day_text,
            "ENTRY": entry_zone,
            "TP1": tp1,
            "TP2": tp2,
            "TP3": tp3,
            "TRAIL_SL": trail_sl,
            "PROFIT%": profit_pct,
            "RSI": rsi_now,
            "SINYAL": signal,
            "S1": s1,
            "R1": r1,
            "VOL": vol,
            "RVOL": rvol,
            "Score": score,
            "Breakout": breakout,
            "Rebound": rebound,
            "Accum": accumulation,
            "MACD_Bull": macd_now > signal_now,
            "Price": price,
            "MA20": ma20,
            "MA50": ma50,
        }
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def run_scan(tickers: list[str], max_workers: int = 8):
    rows = []
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()]))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_ticker, t): t for t in tickers}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                rows.append(result)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["Score", "RVOL", "GAIN"], ascending=[False, False, False]).reset_index(drop=True)
    return df


# =========================
# NOTIFICATION STATE
# =========================
if "sent_entry_signals" not in st.session_state:
    st.session_state.sent_entry_signals = {}


def build_signal_key(row: pd.Series) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"{today}|{row['Ticker']}|{row['AKSI']}|{row['SINYAL']}"


def maybe_send_entry_notifications(df: pd.DataFrame, bot_token: str, chat_id: str, enabled: bool):
    sent_count = 0
    errors = []
    if not enabled or df.empty:
        return sent_count, errors

    entry_df = df[
        (df["AKSI"] == "ENTRY ONLY")
        & (df["SINYAL"].str.contains("ENTRY SIGNAL", na=False))
        & (df["Score"] >= 7)
    ].copy()

    for _, row in entry_df.iterrows():
        key = build_signal_key(row)
        if st.session_state.sent_entry_signals.get(key):
            continue

        msg = (
            f"<b>📈 ENTRY SIGNAL IDX</b>\n"
            f"<b>{row['Ticker']}</b> | Score: <b>{int(row['Score'])}</b>\n"
            f"Fase: <b>{row['FASE']}</b>\n"
            f"Setup: <b>{row['SETUP']}</b>\n"
            f"Aksi: <b>{row['AKSI']}</b>\n"
            f"Harga: <b>{fmt_price(row['Price'])}</b>\n"
            f"Entry: <b>{fmt_price(row['ENTRY'])}</b>\n"
            f"TP1/TP2/TP3: <b>{fmt_price(row['TP1'])}</b> / <b>{fmt_price(row['TP2'])}</b> / <b>{fmt_price(row['TP3'])}</b>\n"
            f"Trail SL: <b>{fmt_price(row['TRAIL_SL'])}</b>\n"
            f"RSI: <b>{row['RSI']:.1f}</b> | RVOL: <b>{row['RVOL']:.2f}x</b>\n"
            f"Day: <b>{row['DAY']}</b>\n"
            f"Sinyal: <b>{row['SINYAL']}</b>"
        )
        ok, info = send_telegram_message(bot_token, chat_id, msg)
        if ok:
            st.session_state.sent_entry_signals[key] = True
            sent_count += 1
        else:
            errors.append(f"{row['Ticker']}: {info}")
    return sent_count, errors


# =========================
# TABLE STYLING
# =========================
def row_bg(row):
    if row["Regime"] == "Bullish":
        return ["background-color: rgba(15,157,88,0.15);"] * len(row)
    if row["Regime"] == "Bearish":
        return ["background-color: rgba(198,40,40,0.14);"] * len(row)
    return ["background-color: rgba(212,160,23,0.11);"] * len(row)


def color_gain(v):
    if pd.isna(v):
        return "color: #d0d7e6"
    return "color: #62f5a6; font-weight:700" if v >= 0 else "color: #ff7b7b; font-weight:700"


def color_score(v):
    if pd.isna(v):
        return "color:#d0d7e6"
    if v >= 7:
        return "background-color: rgba(15,157,88,0.85); color:white; font-weight:800"
    if v >= 5:
        return "background-color: rgba(212,160,23,0.75); color:#111; font-weight:800"
    return "background-color: rgba(198,40,40,0.75); color:white; font-weight:800"


def color_signal(v):
    text = str(v)
    if "ENTRY SIGNAL" in text or "ON TRACK" in text:
        return "background-color: rgba(15,157,88,0.82); color:white; font-weight:800"
    if "WASPADA" in text:
        return "background-color: rgba(212,160,23,0.78); color:#111; font-weight:800"
    if "EXIT" in text or "DEAD" in text:
        return "background-color: rgba(198,40,40,0.78); color:white; font-weight:800"
    return "color:#dce7ff"


def color_action(v):
    text = str(v)
    if text == "ENTRY ONLY":
        return "background-color: rgba(15,157,88,0.86); color:white; font-weight:800"
    if text == "ON TRACK":
        return "background-color: rgba(32,92,255,0.82); color:white; font-weight:800"
    if text == "EXIT":
        return "background-color: rgba(198,40,40,0.8); color:white; font-weight:800"
    return "background-color: rgba(100,100,100,0.25); color:#eaf2ff; font-weight:700"


def color_phase(v):
    text = str(v)
    if text == "BULLISH":
        return "background-color: rgba(15,157,88,0.82); color:white; font-weight:800"
    if text == "BEARISH":
        return "background-color: rgba(198,40,40,0.82); color:white; font-weight:800"
    return "background-color: rgba(212,160,23,0.76); color:#111; font-weight:800"


def style_dataframe(df: pd.DataFrame):
    return (
        df.style
        .apply(row_bg, axis=1)
        .map(color_gain, subset=["GAIN", "PROFIT%"])
        .map(color_score, subset=["Score"])
        .map(color_signal, subset=["SINYAL"])
        .set_properties(subset=["RANK"], **{
            "font-weight": "800",
            "background-color": "rgba(22,70,255,0.32)",
            "color": "#ffffff",
        })
        .set_properties(subset=["Ticker"], **{
            "font-weight": "800",
            "color": "#9dd0ff",
            "background-color": "rgba(8, 30, 60, 0.40)",
        })
        .set_properties(subset=["FASE_UI", "SETUP_UI", "AKSI_UI", "DAY_UI", "ENTRY_ZONE", "TP_PLAN", "LEVELS", "VOL_UI", "REGIME_UI"], **{
            "font-weight": "700",
        })
        .format({
            "GAIN": fmt_pct,
            "TRAIL_SL": fmt_price,
            "PROFIT%": fmt_pct,
            "RSI": lambda x: "-" if pd.isna(x) else f"{x:.1f}",
        })
    )


# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Pengaturan")
auto_refresh = st.sidebar.checkbox("Auto refresh 60 detik", value=True)
if auto_refresh:
    st_autorefresh(interval=60_000, key="refresh_scanner")

scan_all_idx = st.sidebar.checkbox("Scan semua ticker IDX", value=True)
use_default = st.sidebar.checkbox("Pakai watchlist default", value=not scan_all_idx)
custom_input = st.sidebar.text_area(
    "Custom ticker (.JK dipisah koma)",
    value="BBCA.JK, BBRI.JK, BMRI.JK, TLKM.JK, ASII.JK, ADRO.JK, WMUU.JK, ENRG.JK, BIPI.JK, BNBR.JK",
    height=110,
)

max_workers = st.sidebar.slider("Thread scanner", 4, 24, 12)
max_price_filter_enabled = st.sidebar.checkbox("Batasi harga maksimal", value=False)
max_price = st.sidebar.number_input("Harga maksimal", min_value=1, value=1000, step=10)
show_top_n = st.sidebar.slider("Tampilkan Top N", 10, 150, 50)
search_text = st.sidebar.text_input("Cari emiten")

st.sidebar.markdown("---")
st.sidebar.subheader("📨 Telegram")
telegram_enabled = st.sidebar.checkbox("Aktifkan notifikasi entry", value=False)
bot_token = st.sidebar.text_input("Bot Token", value=DEFAULT_BOT_TOKEN, type="password")
chat_id = st.sidebar.text_input("Chat ID", value=DEFAULT_CHAT_ID)
send_test = st.sidebar.button("Kirim test Telegram")

if send_test:
    ok, info = send_telegram_message(
        bot_token,
        chat_id,
        "<b>🤖 Test Notifikasi Berhasil</b>\nScreener Streamlit sudah terkoneksi ke Telegram."
    )
    css_class = "telegram-ok" if ok else "telegram-err"
    st.sidebar.markdown(f'<div class="{css_class}">{info}</div>', unsafe_allow_html=True)

st.sidebar.markdown(
    "<div class='small-note'>Simpan TELEGRAM_BOT_TOKEN dan TELEGRAM_CHAT_ID di Streamlit secrets supaya otomatis terbaca.</div>",
    unsafe_allow_html=True,
)

# =========================
# HEADER
# =========================
st.markdown("<div class='main-title'>📈 HIGH PROB SCREENER IDX — SWING & DAY TRADE</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Bullish / Bearish regime + entry signal only Telegram alert + auto refresh 60 detik</div>",
    unsafe_allow_html=True,
)

# =========================
# BUILD TICKER LIST
# =========================
custom_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
tickers = []

if scan_all_idx:
    with st.spinner("Mengambil daftar semua ticker IDX resmi..."):
        tickers.extend(get_all_idx_tickers())
elif use_default:
    tickers.extend(DEFAULT_TICKERS)

tickers.extend(custom_tickers)
tickers = [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
tickers = sorted(list(dict.fromkeys(tickers)))

if not tickers:
    st.warning("Ticker kosong. Isi watchlist dulu.")
    st.stop()

# =========================
# SCAN
# =========================
with st.spinner(f"Scanning {len(tickers)} ticker..."):
    df = run_scan(tickers, max_workers=max_workers)

if df.empty:
    st.error("Tidak ada data yang berhasil di-scan. Coba ganti watchlist atau refresh lagi.")
    st.stop()

# Filter
if max_price_filter_enabled:
    df = df[df["Price"] <= max_price].copy()

if search_text.strip():
    q = search_text.strip().upper()
    df = df[df["Ticker"].str.contains(q, na=False)].copy()

if df.empty:
    st.warning("Tidak ada saham yang cocok dengan filter saat ini.")
    st.stop()

# =========================
# MARKET SUMMARY
# =========================
bullish_count = int((df["Regime"] == "Bullish").sum())
bearish_count = int((df["Regime"] == "Bearish").sum())
neutral_count = int((df["Regime"] == "Neutral").sum())
total_count = max(len(df), 1)
bullish_pct = bullish_count / total_count * 100
bearish_pct = bearish_count / total_count * 100

# Market status rule
rsi_bull_ratio = (df["RSI"] > 50).mean() * 100
rsi_bear_ratio = (df["RSI"] < 50).mean() * 100
above_ma_ratio = ((df["Price"] > df["MA20"]) & (df["Price"] > df["MA50"])).mean() * 100
below_ma_ratio = (df["Price"] < df["MA20"]).mean() * 100

if rsi_bull_ratio > 60 and above_ma_ratio > 50:
    market_status = "BULLISH 🟢"
    status_class = "metric-green"
elif rsi_bear_ratio > 60 and below_ma_ratio > 60:
    market_status = "BEARISH 🔴"
    status_class = "metric-red"
else:
    market_status = "SIDEWAYS 🟡"
    status_class = "metric-yellow"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"<div class='metric-card {status_class}'><div class='metric-label'>Market Status</div><div class='metric-value'>{market_status}</div></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div class='metric-card metric-green'><div class='metric-label'>Bullish</div><div class='metric-value'>{bullish_count} ({bullish_pct:.1f}%)</div></div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"<div class='metric-card metric-red'><div class='metric-label'>Bearish</div><div class='metric-value'>{bearish_count} ({bearish_pct:.1f}%)</div></div>",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"<div class='metric-card metric-yellow'><div class='metric-label'>Neutral</div><div class='metric-value'>{neutral_count}</div></div>",
        unsafe_allow_html=True,
    )

st.caption(
    f"RSI>50: {rsi_bull_ratio:.1f}% | RSI<50: {rsi_bear_ratio:.1f}% | Price above MA20&MA50: {above_ma_ratio:.1f}% | Price below MA20: {below_ma_ratio:.1f}%"
)

# =========================
# NOTIFICATION
# =========================
sent_count, notif_errors = maybe_send_entry_notifications(df, bot_token, chat_id, telegram_enabled)
if telegram_enabled and sent_count > 0:
    st.markdown(
        f"<div class='telegram-ok'>✅ {sent_count} entry signal baru berhasil dikirim ke Telegram.</div>",
        unsafe_allow_html=True,
    )
elif telegram_enabled and (not bot_token or not chat_id):
    st.markdown(
        "<div class='telegram-warn'>⚠️ Telegram aktif tapi bot token / chat id belum diisi.</div>",
        unsafe_allow_html=True,
    )

if notif_errors:
    st.markdown(
        "<div class='telegram-err'>" + "<br>".join([f"❌ {e}" for e in notif_errors[:5]]) + "</div>",
        unsafe_allow_html=True,
    )

# =========================
# FINAL TABLE
# =========================
df_display = df.copy()
df_display.insert(0, "No", np.arange(1, len(df_display) + 1))
df_display = df_display.head(show_top_n).copy()

def rank_badge(n: int) -> str:
    if n == 1:
        return "🥇 #1"
    if n == 2:
        return "🥈 #2"
    if n == 3:
        return "🥉 #3"
    return f"#{n}"


def regime_badge(x: str) -> str:
    return "🟢 BULLISH" if x == "Bullish" else "🔴 BEARISH" if x == "Bearish" else "🟡 NEUTRAL"


def phase_badge(x: str) -> str:
    return "🟢 PART" if x == "BULLISH" else "🔴 LEMAH" if x == "BEARISH" else "🟡 AKUM"


def setup_badge(x: str) -> str:
    mapping = {
        "BREAKOUT": "🚀 BREAKOUT",
        "REBOUND": "🟢 REBOUND",
        "GC ONLY": "✨ GC ONLY",
        "DEAD CROSS": "💀 DEAD CROSS",
        "OVERHEAT": "🔥 OVERHEAT",
        "OVERSOLD": "🛟 OVERSOLD",
        "WAIT": "⏳ WAIT",
    }
    return mapping.get(x, x)


def action_badge(x: str) -> str:
    mapping = {
        "ENTRY ONLY": "🟢 ENTRY NOW",
        "ON TRACK": "📈 ON TRACK",
        "EXIT": "🚪 EXIT",
        "WAIT": "⏳ WAIT",
    }
    return mapping.get(x, x)


def day_badge(x: str) -> str:
    mapping = {
        "NAIK KUAT": "🚀 NAIK KUAT",
        "NAIK": "📈 NAIK",
        "STABIL": "➖ STABIL",
        "TURUN, EKOR BAWAH": "🪝 TURUN, EKOR BAWAH",
        "TURUN": "🔻 TURUN",
    }
    return mapping.get(x, x)


df_display["RANK"] = df_display["No"].apply(rank_badge)
df_display["FASE_UI"] = df_display["FASE"].apply(phase_badge)
df_display["SETUP_UI"] = df_display["SETUP"].apply(setup_badge)
df_display["AKSI_UI"] = df_display["AKSI"].apply(action_badge)
df_display["DAY_UI"] = df_display["DAY"].apply(day_badge)
df_display["REGIME_UI"] = df_display["Regime"].apply(regime_badge)
df_display["ENTRY_ZONE"] = df_display.apply(lambda r: f"{fmt_price(r['ENTRY'])} / {fmt_price(r['Price'])}", axis=1)
df_display["TP_PLAN"] = df_display.apply(lambda r: f"{fmt_price(r['TP1'])} / {fmt_price(r['TP2'])} / {fmt_price(r['TP3'])}", axis=1)
df_display["LEVELS"] = df_display.apply(lambda r: f"S1 {fmt_price(r['S1'])} | R1 {fmt_price(r['R1'])}", axis=1)
df_display["VOL_UI"] = df_display.apply(lambda r: f"{r['VOL']/1_000_000:.2f}M | RVOL {r['RVOL']:.2f}x" if pd.notna(r['VOL']) and pd.notna(r['RVOL']) else "-", axis=1)

show_cols = [
    "RANK", "Ticker", "FASE_UI", "SETUP_UI", "AKSI_UI", "GAIN", "DAY_UI", "ENTRY_ZONE",
    "TP_PLAN", "TRAIL_SL", "PROFIT%", "RSI", "SINYAL", "LEVELS", "VOL_UI", "Score", "REGIME_UI"
]

st.dataframe(
    style_dataframe(df_display[show_cols]),
    use_container_width=True,
    height=760,
)

# =========================
# EXTRA INFO
# =========================
entry_candidates = df[(df["AKSI"] == "ENTRY ONLY") & (df["Score"] >= 7)].head(10)
if not entry_candidates.empty:
    st.subheader("🔥 Kandidat Entry Terbaik")
    st.dataframe(
        entry_candidates[["Ticker", "Score", "FASE", "SETUP", "ENTRY", "TP1", "TRAIL_SL", "RSI", "RVOL", "SINYAL"]]
        .style
        .map(color_score, subset=["Score"])
        .map(color_signal, subset=["SINYAL"])
        .format({
            "ENTRY": fmt_price,
            "TP1": fmt_price,
            "TRAIL_SL": fmt_price,
            "RSI": lambda x: f"{x:.1f}",
            "RVOL": lambda x: f"{x:.2f}x",
        }),
        use_container_width=True,
        height=320,
    )

st.markdown("---")
st.markdown(
    "<div class='small-note'>Catatan: data menggunakan yfinance sehingga ada kemungkinan delay dan tidak sama persis dengan data broker / TradingView real-time. Untuk scan banyak ticker, pakai watchlist pilihan agar app tetap cepat.</div>",
    unsafe_allow_html=True,
)
