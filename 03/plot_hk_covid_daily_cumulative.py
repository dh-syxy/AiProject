#!/usr/bin/env python
# coding: utf-8
"""读取香港各区疫情 Excel，按日期汇总全港新增确诊与累计确诊并绘图。

数据说明：每条记录为「某日某区」，全港口径 = 当日各区数值之和。

依赖：pip install pandas openpyxl matplotlib

输出：同目录下 PNG（默认 香港疫情_新增与累计确诊.png）
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "香港各区疫情数据_20250322.xlsx"
OUT_IMAGE = BASE_DIR / "香港疫情_新增与累计确诊.png"


def _setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_agg_hk_daily(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df["报告日期"] = pd.to_datetime(df["报告日期"], errors="coerce")
    df = df.dropna(subset=["报告日期"])

    agg = (
        df.groupby("报告日期", sort=True)
        .agg(全港新增确诊=("新增确诊", "sum"), 全港累计确诊=("累计确诊", "sum"))
        .reset_index()
    )
    return agg


def plot_daily_and_cumulative(agg: pd.DataFrame, out_path: Path) -> None:
    _setup_chinese_font()

    dates = agg["报告日期"]
    daily = agg["全港新增确诊"]
    cum = agg["全港累计确诊"]

    fig, ax1 = plt.subplots(figsize=(12, 6), constrained_layout=True)

    ax1.bar(
        dates,
        daily,
        width=0.8,
        color="steelblue",
        alpha=0.65,
        label="每日新增确诊",
        zorder=1,
    )
    ax1.set_ylabel("每日新增确诊", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_title("香港确诊病例：每日新增与累计（全港，18 区汇总）")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_xlabel("报告日期")

    ax2 = ax1.twinx()
    ax2.plot(
        dates,
        cum,
        color="darkred",
        linewidth=1.8,
        marker=".",
        markersize=3,
        label="累计确诊",
        zorder=2,
    )
    ax2.fill_between(dates, cum, alpha=0.1, color="darkred", zorder=0)
    ax2.set_ylabel("累计确诊", color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=35, ha="right")

    h1, lab1 = ax1.get_legend_handles_labels()
    h2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, lab1 + lab2, loc="upper left")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not DATA_FILE.is_file():
        raise SystemExit(f"未找到数据文件: {DATA_FILE}")

    agg = load_agg_hk_daily(DATA_FILE)
    print(f"日期范围: {agg['报告日期'].min().date()} ~ {agg['报告日期'].max().date()}")
    print(f"天数: {len(agg)}")
    plot_daily_and_cumulative(agg, OUT_IMAGE)
    print(f"图表已保存: {OUT_IMAGE}")


if __name__ == "__main__":
    main()
