# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from binance import Client

# ==========================
# 工具：抓 K 線（python-binance）
# ==========================
INTERVAL_MAP = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "3m": Client.KLINE_INTERVAL_3MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "2h": Client.KLINE_INTERVAL_2HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "6h": Client.KLINE_INTERVAL_6HOUR,
    "8h": Client.KLINE_INTERVAL_8HOUR,
    "12h": Client.KLINE_INTERVAL_12HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
    "3d": Client.KLINE_INTERVAL_3DAY,
    "1w": Client.KLINE_INTERVAL_1WEEK,
    "1M": Client.KLINE_INTERVAL_1MONTH,
}

def fetch_klines(symbol: str, interval: str, start_str: str) -> pd.DataFrame:
    client = Client(api_key="", api_secret="")  # 公開端點，不需要key
    raw = client.get_historical_klines(symbol, INTERVAL_MAP[interval], start_str, "now UTC")
    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])
    if df.empty:
        return df
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["open_time","open","high","low","close","volume"]].astype({
        "open":"float","high":"float","low":"float","close":"float","volume":"float"
    })
    df.rename(columns={"open_time":"time"}, inplace=True)
    return df.sort_values("time").reset_index(drop=True)

# ==========================
# 指標
# ==========================
def rsi(series: pd.Series, period=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_dn = pd.Series(dn, index=series.index).rolling(period).mean()
    rs = roll_up / roll_dn
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close: pd.Series, n=20, k=2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    up = mid + k * std
    lo = mid - k * std
    return mid, up, lo

def atr(df: pd.DataFrame, n=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def pivot_points_from_prev(df: pd.DataFrame):
    """以『前一根 K 線』計算 PP/R1/R2/S1/S2（Standard）"""
    if len(df) < 2:
        return [np.nan]*5
    H = df["high"].iloc[-2]
    L = df["low"].iloc[-2]
    C = df["close"].iloc[-2]
    PP = (H + L + C) / 3
    R1 = 2*PP - L
    S1 = 2*PP - H
    R2 = PP + (H - L)
    S2 = PP - (H - L)
    return PP, R1, R2, S1, S2

# ==========================
# 建議 Strike 價位（Sell Call / Sell Put）
# ==========================
def suggest_strikes(df: pd.DataFrame, bb_up, bb_lo, rsi_ser, atr_ser, short_ma, long_ma,
                    pp, r1, r2, s1, s2, atr_mult_call=0.5, atr_mult_put=0.5):
    """回傳 (call_low, call_high), (put_low, put_high)"""
    close = df["close"].iloc[-1]
    up = bb_up.iloc[-1]; lo = bb_lo.iloc[-1]
    atr_now = atr_ser.iloc[-1]
    rsi_now = rsi_ser.iloc[-1]
    ma_ok = short_ma.iloc[-1] > long_ma.iloc[-1]

    # Sell Call：動能過熱 + 靠近上緣；以 R1/R2 與布林上軌/ATR 做區間
    call_low = max(x for x in [r1, up, close] if pd.notna(x))
    call_high = max(x for x in [r2, up + atr_mult_call*atr_now, call_low] if pd.notna(x))
    # 若動能不強或短MA未高於長MA，稍微保守提高下限
    if not ma_ok or (pd.notna(rsi_now) and rsi_now < 65):
        call_low = max(call_low, close + 0.25*atr_now)

    # Sell Put：動能偏弱 + 靠近下緣；以 S1/S2 與布林下軌/ATR 做區間
    put_high = min(x for x in [s1, lo, close] if pd.notna(x))
    put_low = min(x for x in [s2, lo - atr_mult_put*atr_now, put_high] if pd.notna(x))
    # 若動能不弱或短MA>長MA，保守降低上限
    if ma_ok or (pd.notna(rsi_now) and rsi_now > 35):
        put_high = min(put_high, close - 0.25*atr_now)

    return (round(call_low, 2), round(call_high, 2)), (round(put_low, 2), round(put_high, 2))

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="短週期多指標目標價 Dashboard", layout="wide")

st.sidebar.title("參數設定")
symbol = st.sidebar.text_input("交易對 (symbol)", "BTCUSDT").upper()
interval = st.sidebar.selectbox("時間週期 (interval)", list(INTERVAL_MAP.keys()), index=6)  # 預設 4h
lookback_days = st.sidebar.number_input("下載歷史天數", min_value=5, max_value=3650, value=365)

SHORT_MA = st.sidebar.number_input("短期MA", 3, 60, 5)
LONG_MA  = st.sidebar.number_input("長期MA", 5, 200, 20)
RSI_P = st.sidebar.number_input("RSI 週期", 5, 50, 14)
BB_N  = st.sidebar.number_input("布林 n", 5, 60, 20)
BB_K  = st.sidebar.number_input("布林 k", 1.0, 3.0, 2.0, step=0.1)
ATR_N = st.sidebar.number_input("ATR 週期", 5, 50, 14)
ATR_M_CALL = st.sidebar.number_input("Call 區間 ATR 係數", 0.0, 2.0, 0.5, step=0.1)
ATR_M_PUT  = st.sidebar.number_input("Put 區間 ATR 係數", 0.0, 2.0, 0.5, step=0.1)

start_str = (datetime.utcnow() - timedelta(days=int(lookback_days))).strftime("%Y-%m-%d")

st.title("短週期多指標目標價預測 Dashboard（含 Pivot Points）")
st.caption("MA / RSI / Bollinger / MACD / ATR / Pivot Points（PP, R1, R2, S1, S2）")

with st.spinner("下載資料中…"):
    df = fetch_klines(symbol, interval, start_str)

if df.empty:
    st.error("抓不到資料，請確認交易對或週期。")
    st.stop()

# 指標計算
df["MA_S"] = df["close"].rolling(SHORT_MA).mean()
df["MA_L"] = df["close"].rolling(LONG_MA).mean()
df["RSI"]  = rsi(df["close"], RSI_P)
bb_mid, bb_up, bb_lo = bollinger(df["close"], BB_N, BB_K)
df["ATR"]  = atr(df, ATR_N)
macd_line, macd_sig, macd_hist = macd(df["close"], 12, 26, 9)

# Pivot（用前一根 K 計算）
pp, r1, r2, s1, s2 = pivot_points_from_prev(df)

# 建議 Strike 區間
(call_low, call_high), (put_low, put_high) = suggest_strikes(
    df, bb_up, bb_lo, df["RSI"], df["ATR"], df["MA_S"], df["MA_L"],
    pp, r1, r2, s1, s2, ATR_M_CALL, ATR_M_PUT
)

# ==========================
# Plotly 視覺化
# ==========================
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.03)

# K 線
fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"],
                             low=df["low"], close=df["close"], name="K"), row=1, col=1)
# MA
fig.add_trace(go.Scatter(x=df["time"], y=df["MA_S"], name=f"MA{SHORT_MA}", mode="lines"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["time"], y=df["MA_L"], name=f"MA{LONG_MA}", mode="lines"), row=1, col=1)
# 布林
fig.add_trace(go.Scatter(x=df["time"], y=bb_up, name="BB Upper", mode="lines"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["time"], y=bb_lo, name="BB Lower", mode="lines"), row=1, col=1)
# Pivot 水平線
for val, name, color in [(pp,"PP","gold"), (r1,"R1","red"), (r2,"R2","red"),
                         (s1,"S1","green"), (s2,"S2","green")]:
    if not np.isnan(val):
        fig.add_hline(y=val, line_dash="dot", line_color=color, row=1, col=1, annotation_text=name)

# RSI
fig.add_trace(go.Scatter(x=df["time"], y=df["RSI"], name=f"RSI({RSI_P})"), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

# MACD
fig.add_trace(go.Bar(x=df["time"], y=macd_hist, name="MACD Hist"), row=3, col=1)
fig.add_trace(go.Scatter(x=df["time"], y=macd_line, name="MACD"), row=3, col=1)
fig.add_trace(go.Scatter(x=df["time"], y=macd_sig, name="Signal"), row=3, col=1)

fig.update_layout(title=f"{symbol}  {interval}  |  PP/R1/R2/S1/S2＋多指標",
                  xaxis_rangeslider_visible=False, height=800, legend=dict(orientation="h"))

st.plotly_chart(fig, use_container_width=True)

# ==========================
# 指標摘要 & 建議
# ==========================
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Pivot Points")
    st.write(f"PP: **{pp:.2f}**")
    st.write(f"R1: **{r1:.2f}**   |   R2: **{r2:.2f}**")
    st.write(f"S1: **{s1:.2f}**   |   S2: **{s2:.2f}**")

with col2:
    st.subheader("即時指標")
    st.write(f"Close: **{df['close'].iloc[-1]:.2f}**")
    st.write(f"RSI({RSI_P}): **{df['RSI'].iloc[-1]:.2f}**")
    st.write(f"ATR({ATR_N}): **{df['ATR'].iloc[-1]:.2f}**")

with col3:
    st.subheader("建議區間（非投資建議）")
    st.markdown(f"**Sell Call 區間**：`{call_low}  ~  {call_high}`")
    st.markdown(f"**Sell Put  區間**：`{put_low}   ~  {put_high}`")
    st.caption("邏輯：以 Pivot (R1/R2、S1/S2) 為主，綜合布林上/下軌與 ATR 微調；"
               "若短MA未強於長MA或動能不足，系統會保守上調/下調區間。")

st.info("提示：Pivot 以『前一根 K 線』計算。短週期（≤7天）建議搭配 1h / 4h 觀察，"
        "並以資金分散與到期日分散降低尾部風險。")
