# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 15:48:25 2025

@author: 63191
"""

import os
import sys
import math
import time
import json
from datetime import datetime
import requests
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 使用者可調參數
# -------------------------------
API_KEY = "8e89f1784b79042278aa7598de029f73"  # 也可以直接填字串：API_KEY = "YOUR_KEY"
START_DATE = "1999-01-01"                        # 抓資料起始日
USE_SP500 = True                                 # 是否抓 SP500 畫金價/股指比值
USE_CPI_ADJUST = False                           # 是否顯示經 CPI 平減之金價（可視需要開啟）

# 全球地上黃金存量（公噸）。常見估值使用 205,000 公噸（WGC 近年估算量級）
ABOVE_GROUND_GOLD_TONNES = 205_000

# 覆蓋率（0~1）。若假設「100% 貨幣以黃金完全背書」用 1.0；
# 也可測試 0.4（40% 金本位覆蓋率）；可輸入多個做情境比較。
COVERAGE_RATIOS = [1.0, 0.6, 0.4]

# -------------------------------
# FRED 參數
# -------------------------------
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
SERIES = {
    "M2NS": "M2NS",                  # 月頻
    "GOLD": "GOLDPMGBD228NLBM",      # 先用 PM Fix（FRED）
    "SP500": "SP500",                # 日頻
    "CPI": "CPIAUCSL",               # 月頻
}

# 備註：若你偏好 PM 報價，可改用 GOLDPMGBD228NLBM


def fred_series(series_id, start_date=START_DATE, api_key=API_KEY, frequency=None):
    """抓取 FRED 某一系列資料，回傳 pd.Series（index=datetime, values=float）"""
    if not api_key:
        raise RuntimeError("請先在環境變數 FRED_API_KEY 設定你的 API Key，或直接將 API_KEY 字串填入。")

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }
    if frequency:
        params["frequency"] = frequency

    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    obs = data.get("observations", [])
    if not obs:
        raise RuntimeError(f"FRED 回傳空資料：{series_id}")

    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    # 有些系列會出現"." 或空值
    df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    s = df.set_index("date")["value"].astype(float)
    return s

def fred_series_with_fallback(series_candidates, start_date=START_DATE, api_key=API_KEY, frequency=None):
    last_err = None
    for sid in series_candidates:
        try:
            return fred_series(sid, start_date=start_date, api_key=api_key, frequency=frequency)
        except Exception as e:
            last_err = e
            print(f"[warn] 嘗試 {sid} 失敗：{e}")
    raise RuntimeError(f"所有候選 series 皆失敗：{series_candidates}") from last_err


def tonnes_to_troy_ounces(tonnes: float) -> float:
    # 1 公噸 = 32,150.7466 金衡盎司
    return tonnes * 32150.7466


def compute_theoretical_gold_price(m2_usd_billion: pd.Series,
                                   above_ground_tonnes: float,
                                   coverage: float) -> pd.Series:
    """
    理論金價 = (M2 總額 / 地上黃金總盎司) * 覆蓋率
    M2NS 單位是「十億美元」，需乘以 1e9
    """
    gold_oz_total = tonnes_to_troy_ounces(above_ground_tonnes)
    m2_usd = m2_usd_billion * 1e9
    theo_price = (m2_usd / gold_oz_total) * coverage
    return theo_price


def main():
    print("下載 FRED 資料中 ...")
    m2 = fred_series(SERIES["M2NS"])                   # 月頻
    gold = fred_series_with_fallback(
    ["GOLDPMGBD228NLBM", "GOLDAMGBD228NLBM"]
    )
    spx = fred_series(SERIES["SP500"]) if USE_SP500 else None
    cpi = fred_series(SERIES["CPI"]) if USE_CPI_ADJUST else None

    # 將日金價轉為月頻（取月均價，比較平滑；你也可改 'last' 或 'mean'）
    gold_m = gold.resample("M").mean()

    # 將不同頻率合併
    df = pd.DataFrame({"M2NS_bil": m2})
    df["Gold_USD"] = gold_m
    if spx is not None:
        df["SP500"] = spx.resample("M").last()
    if cpi is not None:
        df["CPI"] = cpi  # CPI 已是月頻

    # 計算理論金價（多個覆蓋率情境）
    for cov in COVERAGE_RATIOS:
        col = f"Theo_M2_c{int(cov*100)}"
        df[col] = compute_theoretical_gold_price(df["M2NS_bil"], ABOVE_GROUND_GOLD_TONNES, cov)

    # 計算比值：實際/理論
    for cov in COVERAGE_RATIOS:
        theo_col = f"Theo_M2_c{int(cov*100)}"
        ratio_col = f"Gold_to_Theo_c{int(cov*100)}"
        df[ratio_col] = df["Gold_USD"] / df[theo_col]

    # 金價 / SP500 比值（可用來對照股市昂貴或便宜）
    if USE_SP500:
        df["Gold_over_SP500"] = df["Gold_USD"] / df["SP500"]

    # 依需要做 CPI 平減（把金價換成不變美元）
    if USE_CPI_ADJUST and "CPI" in df:
        df["Gold_Real"] = df["Gold_USD"] / (df["CPI"] / df["CPI"].iloc[-1])
        for cov in COVERAGE_RATIOS:
            theo_col = f"Theo_M2_c{int(cov*100)}"
            df[f"{theo_col}_Real"] = df[theo_col] / (df["CPI"] / df["CPI"].iloc[-1])

    # 輸出 CSV
    out_csv = "gold_m2_valuation.csv"
    df.to_csv(out_csv, float_format="%.6f")
    print(f"已輸出：{out_csv}")

    # 繪圖
    plt.figure(figsize=(11,6))
    plt.plot(df.index, df["Gold_USD"], label="Gold (USD) monthly avg")
    for cov in COVERAGE_RATIOS:
        theo_col = f"Theo_M2_c{int(cov*100)}"
        plt.plot(df.index, df[theo_col], label=f"M2 Theoretical (coverage {int(cov*100)}%)")
    plt.title("Gold vs. M2-Theoretical Value")
    plt.xlabel("Date")
    plt.ylabel("USD / oz")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 比值圖：實際/理論（100% 覆蓋）
    cov = 1.0
    ratio_col = f"Gold_to_Theo_c{int(cov*100)}"
    plt.figure(figsize=(11,4))
    plt.plot(df.index, df[ratio_col])
    plt.axhline(1.0, linestyle="--")
    plt.title(f"Gold / Theoretical (Coverage {int(cov*100)}%)")
    plt.xlabel("Date")
    plt.ylabel("Ratio")
    plt.tight_layout()
    plt.show()

    # 金價 / SP500（可選）
    if USE_SP500:
        plt.figure(figsize=(11,4))
        plt.plot(df.index, df["Gold_over_SP500"])
        plt.title("Gold / S&P 500")
        plt.xlabel("Date")
        plt.ylabel("Ratio")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("發生錯誤：", e, file=sys.stderr)
        sys.exit(1)
