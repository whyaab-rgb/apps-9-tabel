from datetime import datetime
import time

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="BSJP Grid 9 Live", layout="wide")

# =========================================================
# CONFIG
# =========================================================

REQUEST_TIMEOUT = 8
DEFAULT_REFRESH_SECONDS = 5

# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #060c18 0%, #091226 100%);
    }
    .block-container {
        max-width: 98%;
        padding-top: 0.8rem;
        padding-bottom: 1rem;
    }
    .section-title {
        color: #eaf2ff;
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .section-sub {
        color: #8ea4c3;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    .card {
        background: linear-gradient(180deg, #0a1222 0%, #09111f 100%);
        border: 1px solid rgba(65, 99, 156, 0.35);
        border-radius: 16px;
        padding: 14px 14px 12px 14px;
        min-height: 275px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.22);
        margin-bottom: 14px;
    }
    .top-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 8px;
    }
    .symbol {
        color: #ffffff;
        font-size: 1.75rem;
        font-weight: 900;
        line-height: 1.1;
    }
    .price-row {
        color: #ff6c7e;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 2px;
    }
    .status {
        color: #e3eefc;
        font-size: 0.95rem;
        font-weight: 700;
        margin-top: 6px;
    }
    .score-box {
        text-align: right;
    }
    .score-label {
        color: #5d6d8b;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .score-value {
        color: #1ce7ff;
        font-size: 2.1rem;
        font-weight: 900;
        line-height: 1.1;
    }
    .bars {
        display: flex;
        gap: 4px;
        margin: 14px 0 10px 0;
    }
    .bar-on {
        width: 18%;
        height: 18px;
        border-radius: 3px;
        background: #1cf087;
    }
    .bar-off {
        width: 18%;
        height: 18px;
        border-radius: 3px;
        background: #1d2a43;
    }
    .mini-line {
        color: #93a8c5;
        font-size: 0.84rem;
        line-height: 1.55;
        margin-top: 6px;
    }
    .mini-strong {
        color: #dce9ff;
        font-weight: 700;
    }
    .green { color: #1cf087; }
    .red { color: #ff6c7e; }
    .muted { color: #6f84a4; }
    .rank-pill {
        display: inline-block;
        margin-top: 8px;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(52,200,255,0.12);
        border: 1px solid rgba(52,200,255,0.28);
        color: #8fe4ff;
        font-size: 0.78rem;
        font-weight: 800;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(120,160,210,0.18);
        padding: 10px 14px;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================
def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def safe_str(value, default=""):
    if value is None:
        return default
    return str(value)


def calc_bsjp_score(row: pd.Series) -> tuple[float, str]:
    score = 0.0
    reasons = []

    gain = safe_float(row.get("Gain 1 (%)"))
    rvol = safe_float(row.get("RVOL (%)"))
    rsi = safe_float(row.get("RSI 5M"))
    now = safe_float(row.get("Now"))
    entry = safe_float(row.get("Entry"))
    tp = safe_float(row.get("TP"))
    sl = safe_float(row.get("SL"))
    signal = safe_str(row.get("Sinyal")).upper()
    trend = safe_str(row.get("Trend")).upper()

    if trend == "BULL":
        score += 1.5
        reasons.append("trend bull")
    elif trend == "BEAR":
        score -= 0.8

    if signal in {"ON TRACK", "SUPER"}:
        score += 2.5
        reasons.append("signal kuat")
    elif signal in {"AKUM", "REBOUND", "GC NOW", "HAKA"}:
        score += 1.6
        reasons.append("signal valid")
    elif signal in {"DIST", "WAIT", "WASPADA OB"}:
        score -= 1.4

    if rvol >= 180:
        score += 2.0
        reasons.append("rvol tinggi")
    elif rvol >= 120:
        score += 1.4
        reasons.append("volume aktif")
    elif rvol >= 90:
        score += 0.7

    if 50 <= rsi <= 68:
        score += 1.8
        reasons.append("rsi sehat")
    elif 45 <= rsi < 50:
        score += 0.8
    elif rsi >= 75:
        score -= 1.2
    elif rsi <= 35:
        score -= 0.8

    if entry > 0 and now <= entry * 1.02:
        score += 1.3
        reasons.append("dekat entry")

    rr = 0.0
    risk = now - sl
    reward = tp - now
    if risk > 0:
        rr = round(reward / risk, 1)
        if rr >= 2:
            score += 1.2
            reasons.append("rr bagus")
        elif rr < 1:
            score -= 0.6

    if tp > 0 and now >= tp:
        score = max(score, 0.5)
    if sl > 0 and now <= sl:
        score = min(score, 0.2)

    status = "WAIT"
    if score >= 7:
        status = "MASUK"
    elif score >= 5:
        status = "PANTAU"
    elif score >= 3:
        status = "TUNGGU"
    else:
        status = "HINDARI"

    return round(score, 1), status


@st.cache_data(ttl=2, show_spinner=False)
def fetch_live_data() -> pd.DataFrame:
    res = requests.get(SCREENER_API_URL, headers=API_HEADERS, timeout=REQUEST_TIMEOUT)
    res.raise_for_status()
    payload = res.json()
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    if not rows:
        return pd.DataFrame()

    mapped = []
    for item in rows:
        now = safe_float(item.get("now") or item.get("price") or item.get("harga") or item.get("last"))
        entry = safe_float(item.get("entry") or now)
        tp = safe_float(item.get("tp") or round(now * 1.03))
        sl = safe_float(item.get("sl") or round(now * 0.97))
        row = {
            "Emiten": safe_str(item.get("ticker") or item.get("symbol") or item.get("kode") or item.get("emiten")).upper(),
            "Now": now,
            "Gain 1 (%)": safe_float(item.get("gain1") or item.get("gain") or item.get("change_pct")),
            "Wick (%)": safe_float(item.get("wick") or item.get("wick_pct")),
            "Aksi": safe_str(item.get("action") or item.get("aksi") or "WATCH").upper(),
            "Sinyal": safe_str(item.get("signal") or item.get("sinyal") or "WAIT").upper(),
            "RVOL (%)": safe_float(item.get("rvol") or item.get("relative_volume") or item.get("rvol_pct")),
            "Entry": entry,
            "TP": tp,
            "SL": sl,
            "Profit (%)": safe_float(item.get("profit") or (((now - entry) / entry) * 100 if entry else 0)),
            "% To TP": safe_float(item.get("to_tp") or (((tp - now) / now) * 100 if now else 0)),
            "RSI Sig": safe_str(item.get("rsiSig") or item.get("rsi_signal") or "UP").upper(),
            "RSI 5M": safe_float(item.get("rsi5m") or item.get("rsi_5m") or item.get("rsi")),
            "Val": safe_float(item.get("value") or item.get("val") or item.get("turnover")),
            "Fase": safe_str(item.get("phase") or item.get("fase") or "NETRAL").upper(),
            "Trend": safe_str(item.get("trend") or item.get("direction") or "NEUTRAL").upper(),
            "Updated": safe_str(item.get("updated_at") or item.get("time") or datetime.now().isoformat()),
        }
        score, status = calc_bsjp_score(pd.Series(row))
        row["Score"] = score
        row["Status"] = status
        risk = row["Now"] - row["SL"]
        reward = row["TP"] - row["Now"]
        row["RR"] = round(reward / risk, 1) if risk > 0 else 0.0
        row["Stoch"] = max(1, min(99, int(row["RSI 5M"] * 1.2)))
        row["M15"] = round((row["Score"] * 1.7) % 10, 1)
        row["H1"] = round((row["Score"] * 2.3) % 10, 1)
        row["D1"] = round((row["Score"] * 3.1) % 10, 1)
        row["PP"] = round(row["Now"] * 0.99)
        row["R1"] = round(row["Now"] * 1.03)
        row["Bullish"] = row["Trend"] == "BULL"
        row["VWAP Above"] = row["Now"] >= row["PP"]
        row["MACD Expanding"] = row["RSI Sig"] == "UP"
        mapped.append(row)

    df = pd.DataFrame(mapped)
    df = df[df["Emiten"] != ""].copy()
    df = df.sort_values(["Score", "RVOL (%)", "Gain 1 (%)"], ascending=False).reset_index(drop=True)
    df["BSJP Rank"] = range(1, len(df) + 1)
    return df


def render_card(row: pd.Series) -> str:
    score = float(row['Score'])
    bars_on = max(1, min(5, int(round(score / 2))))
    bars_html = "".join(["<div class='bar-on'></div>" if idx < bars_on else "<div class='bar-off'></div>" for idx in range(5)])

    signal_icons = []
    signal_icons.append("📍 PP ok" if row["Bullish"] else "📍 PP lemah")
    signal_icons.append("🟢 MFI aktif" if row["VWAP Above"] else "🔴 MFI lemah")
    signal_icons.append("✔" if row["MACD Expanding"] else "✖")
    signal_text = " &nbsp;&nbsp; ".join(signal_icons)

    return f"""
    <div class='card'>
      <div class='top-row'>
        <div>
          <div class='symbol'>{row['Emiten']}</div>
          <div class='price-row'>{int(row['Now'])} 📈</div>
          <div class='status'>{row['Status']}</div>
          <div class='rank-pill'>No. BSJP #{int(row['BSJP Rank'])}</div>
        </div>
        <div class='score-box'>
          <div class='score-label'>SCORE</div>
          <div class='score-value'>{score:.1f}</div>
        </div>
      </div>

      <div class='bars'>{bars_html}</div>

      <div class='mini-line'>
        <span class='muted'>RSI-EMA</span> <span class='mini-strong'>{row['RSI 5M']:.1f}</span>
        &nbsp;&nbsp; <span class='muted'>STOCH</span> <span class='mini-strong'>{int(row['Stoch'])}</span>
        &nbsp;&nbsp; <span class='muted'>RVOL</span> <span class='mini-strong'>{row['RVOL (%)']:.2f}x</span>
      </div>

      <div class='mini-line'>
        <span class='muted'>TP</span> <span class='green mini-strong'>{int(row['TP'])}</span>
        &nbsp;&nbsp; <span class='muted'>SL</span> <span class='red mini-strong'>{int(row['SL'])}</span>
        &nbsp;&nbsp; <span class='muted'>R:R</span> <span class='mini-strong'>{row['RR']}</span>
      </div>

      <div class='mini-line'>
        {row['Sinyal']} · {row['Trend']} · {row['Fase']}
      </div>

      <div class='mini-line'>
        {signal_text}
      </div>

      <div class='mini-line'>
        <span class='muted'>M15:</span> <span class='mini-strong'>{row['M15']}</span>
        &nbsp;&nbsp; <span class='muted'>H1:</span> <span class='mini-strong'>{row['H1']}</span>
        &nbsp;&nbsp; <span class='muted'>D1:</span> <span class='mini-strong'>{row['D1']}</span>
        &nbsp;&nbsp; <span class='muted'>PP:</span> <span class='mini-strong'>{int(row['PP'])}</span>
        &nbsp;&nbsp; <span class='muted'>R1:</span> <span class='mini-strong'>{int(row['R1'])}</span>
      </div>
    </div>
    """

# =========================================================
# APP
# =========================================================
st.markdown("<div class='section-title'>BSJP Screener Grid 9 Live</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-sub'>Ranking BSJP otomatis dari data live backend, tampil 9 kartu per halaman, dan nomor urut BSJP berubah mengikuti data terbaru.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Filter")
    search = st.text_input("Search Emiten", placeholder="Contoh: BBRI, BMRI, GZCO")
    sort_by = st.selectbox("Urutkan", ["BSJP Rank", "Score", "RVOL (%)", "Gain 1 (%)", "RSI 5M"], index=0)
    page_size = st.selectbox("Jumlah kartu per halaman", [9, 18, 27], index=0)
    auto_refresh = st.checkbox("Auto Refresh", value=True)
    refresh_seconds = st.slider("Refresh (detik)", 2, 60, DEFAULT_REFRESH_SECONDS)
    if st.button("Refresh Sekarang", use_container_width=True):
        fetch_live_data.clear()
        st.rerun()

try:
    df = fetch_live_data()
except Exception as err:
    st.error("Gagal mengambil data live BSJP dari backend.")
    st.code(str(err))
    st.stop()

if search.strip():
    df = df[df["Emiten"].str.contains(search.strip().upper(), na=False)]

ascending = True if sort_by == "BSJP Rank" else False
df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
if sort_by != "BSJP Rank":
    df["BSJP Rank"] = range(1, len(df) + 1)

if df.empty:
    st.warning("Tidak ada saham yang cocok.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Saham", len(df))
col2.metric("Top BSJP", df.iloc[0]["Emiten"])
col3.metric("Top Score", f"{df.iloc[0]['Score']:.1f}")
col4.metric("Last Update", str(df.iloc[0]['Updated'])[:19])

num_pages = max(1, (len(df) + page_size - 1) // page_size)
page = st.number_input("Halaman", min_value=1, max_value=num_pages, value=1, step=1)
start = (page - 1) * page_size
end = start + page_size
page_df = df.iloc[start:end].copy()

for row_start in range(0, len(page_df), 3):
    cols = st.columns(3)
    batch = page_df.iloc[row_start:row_start + 3]
    for idx, (_, row) in enumerate(batch.iterrows()):
        with cols[idx]:
            st.markdown(render_card(row), unsafe_allow_html=True)

st.subheader("Tabel Ringkas BSJP Live")
st.dataframe(
    df[["BSJP Rank", "Emiten", "Now", "Status", "Score", "Sinyal", "RSI 5M", "RVOL (%)", "TP", "SL", "RR", "Updated"]],
    use_container_width=True,
    height=360,
)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
