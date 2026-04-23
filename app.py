import requests
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(page_title="Scalping Tight Real-Time", layout="wide")

# =========================================================
# CONFIG
# =========================================================
MAX_SELECTED = 5
DEFAULT_SYMBOLS = "BBCA,BBRI,GOTO"
TOP_N = 5

# =========================================================
# GLOBAL STYLE
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #081018;
    color: white;
}
.block-container {
    max-width: 99%;
    padding-top: 0.8rem;
    padding-bottom: 1rem;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #071019 0%, #0a1320 100%);
}
[data-testid="stSidebar"] {
    background-color: #09111d;
}
h1, h2, h3, h4, h5, h6, p, span, div, label {
    color: #e8f0ff !important;
}
.small-note {
    font-size: 12px;
    color: #9db1cc !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def normalize_jk_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if not s:
        return ""
    if ":" in s:
        return s
    if not s.endswith(".JK"):
        s = f"{s}.JK"
    return s


def latest(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def fmt_price(v):
    if pd.isna(v):
        return "-"
    if v >= 100:
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def fmt_pct(v):
    if pd.isna(v):
        return "-"
    return f"{v:.1f}%"


def rsi_cell_text(v):
    if pd.isna(v):
        return "-"
    return f"{v:.1f}"


def human_value(v):
    if pd.isna(v):
        return "-"
    if v >= 1_000_000_000_000:
        return f"{v / 1_000_000_000_000:.1f}T"
    if v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    return f"{v:,.0f}"


# =========================================================
# TELEGRAM
# =========================================================
def send_telegram_message(bot_token: str, chat_id: str, message: str):
    if not bot_token or not chat_id:
        return False, "Bot token / chat_id kosong"

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            return True, "Terkirim"
        return False, f"HTTP {r.status_code}: {r.text}"
    except Exception as e:
        return False, str(e)


# =========================================================
# DATA SOURCE
# =========================================================
@st.cache_data(ttl=600)
def get_daily_data(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return pd.DataFrame()

        needed = ["Open", "High", "Low", "Close", "Volume"]
        for col in needed:
            if col not in df.columns:
                return pd.DataFrame()

        return df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120)
def get_intraday_5m(symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period="5d",
            interval="5m",
            auto_adjust=False,
            progress=False,
            threads=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return pd.DataFrame()
        return df.dropna().copy()
    except Exception:
        return pd.DataFrame()


def get_live_price_twelvedata(symbol: str, api_key: str):
    if not api_key:
        return np.nan

    base = symbol.replace(".JK", "")
    candidates = [
        f"{base}",
        f"IDX:{base}",
        f"JK:{base}",
        f"{base}.JK"
    ]

    for sym in candidates:
        try:
            url = "https://api.twelvedata.com/price"
            r = requests.get(url, params={"symbol": sym, "apikey": api_key}, timeout=10)
            data = r.json()
            if "price" in data:
                return float(data["price"])
        except Exception:
            pass

    return np.nan


def get_live_price_yf(symbol: str):
    try:
        df = yf.download(
            symbol,
            period="1d",
            interval="1m",
            auto_adjust=False,
            progress=False,
            threads=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if not df.empty and "Close" in df.columns:
            close_series = df["Close"].dropna()
            if not close_series.empty:
                return float(close_series.iloc[-1])
    except Exception:
        pass
    return np.nan


def get_live_price(symbol: str, provider: str, api_key: str):
    if provider == "Twelve Data":
        px = get_live_price_twelvedata(symbol, api_key)
        if not pd.isna(px):
            return px
    return get_live_price_yf(symbol)


# =========================================================
# INDICATORS
# =========================================================
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["MA5"] = x["Close"].rolling(5).mean()
    x["MA10"] = x["Close"].rolling(10).mean()
    x["MA20"] = x["Close"].rolling(20).mean()
    x["MA50"] = x["Close"].rolling(50).mean()
    x["EMA9"] = x["Close"].ewm(span=9, adjust=False).mean()
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()

    delta = x["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    x["RSI"] = 100 - (100 / (1 + rs))

    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACD_SIGNAL"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["MACD_HIST"] = x["MACD"] - x["MACD_SIGNAL"]

    x["BB_MID"] = x["Close"].rolling(20).mean()
    std20 = x["Close"].rolling(20).std()
    x["BB_UPPER"] = x["BB_MID"] + 2 * std20
    x["BB_LOWER"] = x["BB_MID"] - 2 * std20

    x["VOL_MA5"] = x["Volume"].rolling(5).mean()
    x["VOL_MA20"] = x["Volume"].rolling(20).mean()

    x["SUPPORT20"] = x["Low"].rolling(20).min()
    x["RESIST20"] = x["High"].rolling(20).max()

    high_low = x["High"] - x["Low"]
    high_close = np.abs(x["High"] - x["Close"].shift())
    low_close = np.abs(x["Low"] - x["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    x["ATR14"] = tr.rolling(14).mean()

    body = (x["Close"] - x["Open"]).abs()
    upper_wick = x["High"] - x[["Open", "Close"]].max(axis=1)
    lower_wick = x[["Open", "Close"]].min(axis=1) - x["Low"]
    candle_range = (x["High"] - x["Low"]).replace(0, np.nan)
    x["BODY"] = body
    x["UPPER_WICK"] = upper_wick.clip(lower=0)
    x["LOWER_WICK"] = lower_wick.clip(lower=0)
    x["WICK_PCT"] = ((x["UPPER_WICK"] + x["LOWER_WICK"]) / candle_range) * 100

    return x


# =========================================================
# SCALPING ENGINE (TIGHT & STRICT)
# =========================================================
def get_trend(close_, ma20, ma50):
    if pd.isna(close_) or pd.isna(ma20) or pd.isna(ma50):
        return "NEUTRAL"
    if close_ > ma20 > ma50:
        return "BULL"
    if close_ < ma20 < ma50:
        return "BEAR"
    return "NEUTRAL"


def get_rsi_signal(rsi, macd, macd_signal):
    if pd.isna(rsi) or pd.isna(macd) or pd.isna(macd_signal):
        return "WAIT"
    if 52 <= rsi <= 60 and macd > macd_signal:
        return "UP"
    if rsi >= 60 and macd > macd_signal:
        return "HOT"
    if rsi < 48 and macd < macd_signal:
        return "DEAD"
    return "WAIT"


def get_scalp_signal(close_, ema9, ma20, rsi, macd, macd_signal, vol, vol_ma5, vol_ma20, resistance, wick):
    if any(pd.isna(v) for v in [close_, ema9, ma20, rsi, macd, macd_signal, vol, vol_ma5, vol_ma20, resistance, wick]):
        return "WAIT"

    breakout_near = close_ >= resistance * 0.985
    trend_ok = close_ > ema9 > ma20
    macd_ok = macd > macd_signal
    rsi_ok = 55 <= rsi <= 72
    vol_ok = vol > vol_ma5 and vol > vol_ma20
    wick_ok = wick < 30

    if trend_ok and macd_ok and rsi_ok and vol_ok and breakout_near and wick_ok:
        return "SCALP STRONG"

    if trend_ok and macd_ok and 52 <= rsi <= 68 and vol > vol_ma5 and wick < 35:
        return "SCALP READY"

    if trend_ok and macd_ok and 50 <= rsi <= 65:
        return "WATCH"

    if rsi >= 74 or wick >= 40:
        return "OVERHEAT"

    return "WAIT"


def get_scalp_action(signal_label, close_, entry):
    if signal_label == "SCALP STRONG":
        if not pd.isna(entry) and close_ <= entry * 1.01:
            return "ENTRY NOW"
        return "CHASE LIGHT"
    if signal_label == "SCALP READY":
        return "WAIT TRIGGER"
    if signal_label == "WATCH":
        return "WATCH"
    if signal_label == "OVERHEAT":
        return "AVOID"
    return "WAIT"


def compute_scalp_score(close_, ema9, ma20, rsi, macd, macd_signal, vol, vol_ma5, vol_ma20, resistance, wick):
    score = 0

    if not pd.isna(close_) and not pd.isna(ema9) and not pd.isna(ma20) and close_ > ema9 > ma20:
        score += 25

    if not pd.isna(rsi):
        if 56 <= rsi <= 68:
            score += 20
        elif 52 <= rsi < 56:
            score += 12
        elif rsi > 72:
            score -= 8

    if not pd.isna(macd) and not pd.isna(macd_signal) and macd > macd_signal:
        score += 18

    if not pd.isna(vol) and not pd.isna(vol_ma5) and vol > vol_ma5:
        score += 12

    if not pd.isna(vol) and not pd.isna(vol_ma20) and vol > vol_ma20:
        score += 10

    if not pd.isna(resistance) and close_ >= resistance * 0.985:
        score += 10

    if not pd.isna(wick):
        if wick < 20:
            score += 8
        elif wick < 30:
            score += 4
        elif wick >= 40:
            score -= 10

    return max(min(score, 100), 0)


# =========================================================
# ROW BUILDER
# =========================================================
def build_row(symbol: str, daily_df: pd.DataFrame, intraday_5m: pd.DataFrame, live_price: float):
    df = calc_indicators(daily_df)
    if len(df) < 30:
        return None

    last_close_hist = latest(df["Close"])
    close_ = live_price if not pd.isna(live_price) else last_close_hist
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close_hist

    gain = ((close_ - prev_close) / prev_close * 100) if prev_close else 0.0

    wick = latest(df["WICK_PCT"])
    rsi = latest(df["RSI"])
    macd = latest(df["MACD"])
    macd_signal = latest(df["MACD_SIGNAL"])
    vol = latest(df["Volume"])
    vol_ma5 = latest(df["VOL_MA5"])
    vol_ma20 = latest(df["VOL_MA20"])
    ma20 = latest(df["MA20"])
    ma50 = latest(df["MA50"])
    ema9 = latest(df["EMA9"])
    resistance = latest(df["RESIST20"])
    atr = latest(df["ATR14"])

    rvol = (vol / vol_ma20 * 100) if not pd.isna(vol_ma20) and vol_ma20 > 0 else np.nan

    entry = round(max(ema9, ma20)) if not pd.isna(ema9) and not pd.isna(ma20) else round(close_)
    now_price = close_

    tp1 = round(close_ + (atr * 0.8)) if not pd.isna(atr) else round(close_ * 1.02)
    tp2 = round(close_ + (atr * 1.5)) if not pd.isna(atr) else round(close_ * 1.035)
    sl = round(close_ - (atr * 0.6)) if not pd.isna(atr) else round(close_ * 0.985)

    profit = ((now_price - entry) / entry * 100) if entry else 0.0
    to_tp = ((tp1 - now_price) / now_price * 100) if now_price else 0.0

    intraday_rsi = np.nan
    if not intraday_5m.empty and "Close" in intraday_5m.columns:
        intra = calc_indicators(intraday_5m)
        intraday_rsi = latest(intra["RSI"])

    trend = get_trend(close_, ma20, ma50)
    rsi_sig = get_rsi_signal(rsi, macd, macd_signal)
    sinyal = get_scalp_signal(close_, ema9, ma20, rsi, macd, macd_signal, vol, vol_ma5, vol_ma20, resistance, wick)
    aksi = get_scalp_action(sinyal, close_, entry)

    scalp_score = compute_scalp_score(close_, ema9, ma20, rsi, macd, macd_signal, vol, vol_ma5, vol_ma20, resistance, wick)
    val = close_ * vol if not pd.isna(close_) and not pd.isna(vol) else np.nan

    return {
        "symbol": symbol.replace(".JK", ""),
        "full_symbol": symbol,
        "gain": gain,
        "wick": wick,
        "aksi": aksi,
        "sinyal": sinyal,
        "rvol": rvol,
        "entry": entry,
        "now": now_price,
        "tp": tp1,
        "tp2": tp2,
        "sl": sl,
        "profit": profit,
        "to_tp": to_tp,
        "rsi": rsi,
        "rsi_sig": rsi_sig,
        "rsi_5m": intraday_rsi,
        "val": val,
        "trend": trend,
        "score_scalp": scalp_score,
        "daily_df": df
    }


def run_live_monitor(symbols, provider, api_key):
    rows = []
    for symbol in symbols:
        try:
            daily = get_daily_data(symbol, period="6mo", interval="1d")
            if daily.empty:
                continue

            intra5 = get_intraday_5m(symbol)
            live_price = get_live_price(symbol, provider, api_key)
            row = build_row(symbol, daily, intra5, live_price)

            if row is not None and not pd.isna(row["now"]):
                rows.append(row)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["score_scalp", "rvol", "gain"],
        ascending=[False, False, False]
    ).reset_index(drop=True)


# =========================================================
# CELL COLORS
# =========================================================
def bg_gain(v):
    if pd.isna(v):
        return "#243244"
    if v > 2:
        return "#10b981"
    if v > 0:
        return "#15803d"
    if v > -1.5:
        return "#dc2626"
    return "#991b1b"


def bg_wick(v):
    if pd.isna(v):
        return "#243244"
    if v < 20:
        return "#0f766e"
    if v < 30:
        return "#2563eb"
    if v < 40:
        return "#d97706"
    return "#dc2626"


def bg_aksi(v):
    mapping = {
        "ENTRY NOW": "#7c3aed",
        "CHASE LIGHT": "#1d4ed8",
        "WAIT TRIGGER": "#b45309",
        "WATCH": "#334155",
        "AVOID": "#b91c1c",
        "WAIT": "#111827"
    }
    return mapping.get(v, "#334155")


def bg_sinyal(v):
    mapping = {
        "SCALP STRONG": "#7e22ce",
        "SCALP READY": "#16a34a",
        "WATCH": "#2563eb",
        "OVERHEAT": "#ea580c",
        "WAIT": "#111827"
    }
    return mapping.get(v, "#334155")


def bg_rvol(v):
    if pd.isna(v):
        return "#243244"
    if v >= 250:
        return "#9333ea"
    if v >= 150:
        return "#f97316"
    if v >= 100:
        return "#2563eb"
    return "#374151"


def bg_price(kind):
    mapping = {"entry": "#1d4ed8", "now": "#2563eb", "tp": "#16a34a", "sl": "#b91c1c"}
    return mapping.get(kind, "#243244")


def bg_profit(v):
    if pd.isna(v):
        return "#243244"
    if v > 1.5:
        return "#16a34a"
    if v > 0:
        return "#0f766e"
    if v > -1:
        return "#92400e"
    return "#b91c1c"


def bg_to_tp(v):
    if pd.isna(v):
        return "#243244"
    if v <= 0.8:
        return "#f97316"
    if v <= 2:
        return "#16a34a"
    return "#0f766e"


def bg_rsi_sig(v):
    mapping = {"UP": "#16a34a", "HOT": "#7c3aed", "DEAD": "#dc2626", "WAIT": "#111827"}
    return mapping.get(v, "#334155")


def bg_rsi(v):
    if pd.isna(v):
        return "#243244"
    if v >= 72:
        return "#f59e0b"
    if v >= 55:
        return "#16a34a"
    if v >= 48:
        return "#2563eb"
    return "#7c3aed"


def bg_trend(v):
    mapping = {"BULL": "#16a34a", "BEAR": "#dc2626", "NEUTRAL": "#6b7280"}
    return mapping.get(v, "#334155")


def bg_scalp_score(v):
    if pd.isna(v):
        return "#243244"
    if v >= 80:
        return "#9333ea"
    if v >= 65:
        return "#16a34a"
    if v >= 50:
        return "#2563eb"
    return "#374151"


# =========================================================
# HTML TABLE
# =========================================================
def make_html_table(df: pd.DataFrame, title: str, sub: str):
    html = textwrap.dedent(f"""
    <html>
    <head>
    <style>
    body {{
        margin: 0;
        background: #07111b;
        color: white;
        font-family: Arial, Helvetica, sans-serif;
    }}
    .screen-box {{
        border: 1px solid #17324d;
        border-radius: 10px;
        padding: 8px;
        background: #07111b;
        box-sizing: border-box;
        width: 100%;
    }}
    .screener-title {{
        text-align: center;
        font-weight: 800;
        font-size: 13px;
        color: #eaf2ff;
        margin-bottom: 4px;
        letter-spacing: 0.3px;
    }}
    .screener-sub {{
        text-align: center;
        color: #9fb5d1;
        font-size: 10px;
        margin-bottom: 6px;
    }}
    .table-wrap {{
        width: 100%;
        overflow-x: auto;
    }}
    .custom-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 11px;
        min-width: 1450px;
    }}
    .custom-table th {{
        background: #184574;
        color: #ffffff;
        border: 1px solid #2a527b;
        padding: 5px 3px;
        text-align: center;
        white-space: nowrap;
        font-weight: 800;
    }}
    .custom-table td {{
        border: 1px solid #20364e;
        padding: 4px 3px;
        text-align: center;
        white-space: nowrap;
        font-weight: 700;
    }}
    .footer-line {{
        margin-top: 6px;
        text-align: center;
        color: #ffd451;
        font-size: 10px;
        font-weight: 700;
    }}
    </style>
    </head>
    <body>
    <div class="screen-box">
      <div class="screener-title">{title}</div>
      <div class="screener-sub">{sub}</div>
      <div class="table-wrap">
      <table class="custom-table">
        <thead>
          <tr>
            <th>RANK</th>
            <th>EMITEN</th>
            <th>SCALP SCORE</th>
            <th>GAIN</th>
            <th>WICK</th>
            <th>AKSI</th>
            <th>SINYAL</th>
            <th>RVOL</th>
            <th>ENTRY</th>
            <th>NOW</th>
            <th>TP1</th>
            <th>TP2</th>
            <th>SL</th>
            <th>PROFIT</th>
            <th>%TO TP1</th>
            <th>RSI SIG</th>
            <th>RSI</th>
            <th>RSI 5M</th>
            <th>VAL</th>
            <th>TREND</th>
          </tr>
        </thead>
        <tbody>
    """)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        html += f"""
        <tr>
            <td style="background:#0f172a;color:#fff;">{i}</td>
            <td style="background:#1d4ed8;color:#fff;">{row['symbol']}</td>
            <td style="background:{bg_scalp_score(row['score_scalp'])};color:#fff;">{int(row['score_scalp'])}</td>
            <td style="background:{bg_gain(row['gain'])};color:#fff;">{fmt_pct(row['gain'])}</td>
            <td style="background:{bg_wick(row['wick'])};color:#fff;">{fmt_pct(row['wick'])}</td>
            <td style="background:{bg_aksi(row['aksi'])};color:#fff;">{row['aksi']}</td>
            <td style="background:{bg_sinyal(row['sinyal'])};color:#fff;">{row['sinyal']}</td>
            <td style="background:{bg_rvol(row['rvol'])};color:#fff;">{fmt_pct(row['rvol'])}</td>
            <td style="background:{bg_price('entry')};color:#fff;">{fmt_price(row['entry'])}</td>
            <td style="background:{bg_price('now')};color:#fff;">{fmt_price(row['now'])}</td>
            <td style="background:{bg_price('tp')};color:#fff;">{fmt_price(row['tp'])}</td>
            <td style="background:{bg_price('tp')};color:#fff;">{fmt_price(row['tp2'])}</td>
            <td style="background:{bg_price('sl')};color:#fff;">{fmt_price(row['sl'])}</td>
            <td style="background:{bg_profit(row['profit'])};color:#fff;">{fmt_pct(row['profit'])}</td>
            <td style="background:{bg_to_tp(row['to_tp'])};color:#fff;">{fmt_pct(row['to_tp'])}</td>
            <td style="background:{bg_rsi_sig(row['rsi_sig'])};color:#fff;">{row['rsi_sig']}</td>
            <td style="background:{bg_rsi(row['rsi'])};color:#fff;">{rsi_cell_text(row['rsi'])}</td>
            <td style="background:{bg_rsi(row['rsi_5m'])};color:#fff;">{rsi_cell_text(row['rsi_5m'])}</td>
            <td style="background:#183b69;color:#fff;">{human_value(row['val'])}</td>
            <td style="background:{bg_trend(row['trend'])};color:#fff;">{row['trend']}</td>
        </tr>
        """

    html += """
        </tbody>
      </table>
      </div>
      <div class="footer-line">SCALPING TIGHT MODE | 1-5 saham | fokus breakout, EMA9, volume, RSI, MACD | live monitor</div>
    </div>
    </body>
    </html>
    """
    return html


# =========================================================
# CHART DETAIL
# =========================================================
def show_detail_chart(df: pd.DataFrame, symbol_name: str):
    st.subheader(f"Chart Detail: {symbol_name}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candlestick"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], mode="lines", name="EMA9"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["RESIST20"], mode="lines", name="RESIST20"))
    fig.update_layout(
        height=520,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# HEADER
# =========================================================
st.title("SCALPING TIGHT REAL-TIME")
st.markdown(
    '<div class="small-note">fokus scalping ketat | 1-5 saham pilihan | EMA9 + MA20 + volume + RSI + MACD | live monitor</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Pengaturan")

    search_input = st.text_input(
        "Masukkan saham (pisahkan koma, max 5)",
        value=DEFAULT_SYMBOLS
    )

    live_provider = st.selectbox(
        "Live Price Provider",
        ["Fallback yfinance", "Twelve Data"],
        index=0
    )

    twelve_api_key = st.text_input("Twelve Data API Key", type="password")
    auto_refresh = st.checkbox("Auto Refresh Live", value=True)
    refresh_sec = st.selectbox("Refresh tiap", [2, 3, 5, 10], index=1)

    st.markdown("---")
    st.subheader("Telegram Bot")
    telegram_enabled = st.checkbox("Aktifkan notifikasi Telegram", value=False)
    telegram_bot_token = st.text_input("Bot Token", type="password")
    telegram_chat_id = st.text_input("Chat ID")

    send_test_btn = st.button("Tes Kirim Telegram", use_container_width=True)
    if send_test_btn:
        now_text = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        test_message = (
            "🤖 <b>Test Notifikasi Berhasil</b>\n"
            "✅ Bot Telegram sudah terhubung ke scalping monitor.\n\n"
            f"🕒 <b>Waktu:</b> {now_text}\n"
            "📡 <b>Status:</b> ONLINE\n"
            "🔥 Sistem siap kirim alert scalping."
        )
        ok, msg = send_telegram_message(telegram_bot_token, telegram_chat_id, test_message)
        if ok:
            st.success("Pesan test berhasil dikirim.")
        else:
            st.error(f"Gagal kirim test: {msg}")

# =========================================================
# PARSE SYMBOLS
# =========================================================
symbols = [normalize_jk_symbol(x) for x in search_input.split(",") if x.strip()]
symbols = list(dict.fromkeys(symbols))[:MAX_SELECTED]

if not symbols:
    st.warning("Masukkan minimal 1 saham.")
    st.stop()

# =========================================================
# MAIN RENDER
# =========================================================
def render_live_panel():
    df = run_live_monitor(symbols, live_provider, twelve_api_key)

    if df.empty:
        st.error("Tidak ada data yang berhasil diambil.")
        return

    display_df = df.head(TOP_N).reset_index(drop=True)
    last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MONITOR", ", ".join([x.replace(".JK", "") for x in symbols]))
    m2.metric("TOP PICK", display_df.iloc[0]["symbol"])
    m3.metric("TOP SCORE", int(display_df.iloc[0]["score_scalp"]))
    m4.metric("LAST UPDATE", last_run)

    st.subheader("Scalping Live Table")
    components.html(
        make_html_table(
            display_df,
            "SCALPING TIGHT REAL-TIME",
            f"Update: {last_run} | Provider: {live_provider}"
        ),
        height=560,
        scrolling=True
    )

    st.subheader("Ranking Live")
    rank_df = display_df[[
        "symbol", "now", "gain", "rvol", "rsi", "rsi_5m",
        "trend", "score_scalp", "sinyal", "aksi"
    ]].copy()
    rank_df.columns = [
        "EMITEN", "PRICE", "GAIN", "RVOL", "RSI", "RSI 5M",
        "TREND", "SCALP SCORE", "SIGNAL", "AKSI"
    ]
    rank_df["PRICE"] = rank_df["PRICE"].apply(fmt_price)
    rank_df["GAIN"] = rank_df["GAIN"].apply(fmt_pct)
    rank_df["RVOL"] = rank_df["RVOL"].apply(fmt_pct)
    rank_df["RSI"] = rank_df["RSI"].apply(rsi_cell_text)
    rank_df["RSI 5M"] = rank_df["RSI 5M"].apply(rsi_cell_text)
    st.dataframe(rank_df, use_container_width=True, height=260)

    selected_symbol = st.selectbox(
        "Pilih saham untuk detail",
        display_df["full_symbol"].tolist(),
        key="detail_symbol"
    )
    selected_row = display_df[display_df["full_symbol"] == selected_symbol].iloc[0]
    selected_df = selected_row["daily_df"]

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("EMITEN", selected_row["symbol"])
    d2.metric("PRICE", fmt_price(selected_row["now"]))
    d3.metric("GAIN", fmt_pct(selected_row["gain"]))
    d4.metric("RVOL", fmt_pct(selected_row["rvol"]))
    d5.metric("SCALP SCORE", int(selected_row["score_scalp"]))
    d6.metric("RSI 5M", rsi_cell_text(selected_row["rsi_5m"]))

    show_detail_chart(selected_df, selected_row["symbol"])

    st.subheader("Analisa Scalping Ketat")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**Signal:** {selected_row['sinyal']}")
        st.write(f"**Aksi:** {selected_row['aksi']}")
        st.write(f"**Trend:** {selected_row['trend']}")
    with c2:
        st.write(f"**Entry:** {fmt_price(selected_row['entry'])}")
        st.write(f"**TP1:** {fmt_price(selected_row['tp'])}")
        st.write(f"**TP2:** {fmt_price(selected_row['tp2'])}")
    with c3:
        st.write(f"**SL:** {fmt_price(selected_row['sl'])}")
        st.write(f"**RSI:** {rsi_cell_text(selected_row['rsi'])}")
        st.write(f"**RSI 5M:** {rsi_cell_text(selected_row['rsi_5m'])}")

    if telegram_enabled and telegram_bot_token and telegram_chat_id:
        top_row = display_df.iloc[0]
        alert_key = f"{top_row['symbol']}-{int(top_row['score_scalp'])}-{top_row['sinyal']}-{fmt_price(top_row['now'])}"
        last_alert_key = st.session_state.get("last_alert_key", "")

        if top_row["score_scalp"] >= 70 and top_row["sinyal"] in ["SCALP STRONG", "SCALP READY"]:
            if alert_key != last_alert_key:
                message = (
                    f"🚨 <b>SCALPING ALERT</b>\n"
                    f"🕒 <b>{last_run}</b>\n\n"
                    f"<b>{top_row['symbol']}</b>\n"
                    f"💰 Price: <b>{fmt_price(top_row['now'])}</b>\n"
                    f"📍 Signal: <b>{top_row['sinyal']}</b>\n"
                    f"⚡ Action: <b>{top_row['aksi']}</b>\n"
                    f"🏆 Scalp Score: <b>{int(top_row['score_scalp'])}</b>\n"
                    f"⚡ RVOL: <b>{fmt_pct(top_row['rvol'])}</b>\n"
                    f"📈 Trend: <b>{top_row['trend']}</b>\n"
                    f"🎯 Entry: <b>{fmt_price(top_row['entry'])}</b>\n"
                    f"🏁 TP1: <b>{fmt_price(top_row['tp'])}</b>\n"
                    f"🏁 TP2: <b>{fmt_price(top_row['tp2'])}</b>\n"
                    f"🛑 SL: <b>{fmt_price(top_row['sl'])}</b>"
                )
                ok, _ = send_telegram_message(telegram_bot_token, telegram_chat_id, message)
                if ok:
                    st.session_state["last_alert_key"] = alert_key
                    st.success("Alert Telegram terkirim")

# =========================================================
# AUTO REFRESH
# =========================================================
if auto_refresh:
    @st.fragment(run_every=f"{refresh_sec}s")
    def live_fragment():
        render_live_panel()
    live_fragment()
else:
    render_live_panel()
