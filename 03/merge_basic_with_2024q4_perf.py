#!/usr/bin/env python
# coding: utf-8
"""在「员工基本信息表」基础上合并 2024 年第 4 季度绩效评分，输出新 Excel。

放置在同一目录下：
  - 员工基本信息表.xlsx   （须含列：员工ID）
  - 员工绩效表.xlsx       （须含列：员工ID、绩效评分；若有「年度」「季度」则只取 2024 年第 4 季度）

输出：员工基本信息_含2024Q4绩效.xlsx

依赖：pip install pandas openpyxl
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FILE_BASIC = BASE_DIR / "员工基本信息表.xlsx"
FILE_PERF = BASE_DIR / "员工绩效表.xlsx"
OUT_FILE = BASE_DIR / "员工基本信息_含2024Q4绩效.xlsx"

YEAR = 2024
QUARTER = 4

REQUIRED_BASIC = ("员工ID",)
REQUIRED_PERF = ("员工ID", "绩效评分")
SCORE_COL_OUT = "2024年第四季度绩效评分"


def _require_columns(df: pd.DataFrame, name: str, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{name} 缺少列 {missing}。实际列名: {list(df.columns)}"
        )


def extract_q4_perf(perf: pd.DataFrame) -> pd.DataFrame:
    """取出 2024Q4 绩效；若无年度/季度列则假定整张表即为该季度数据。"""
    if "年度" in perf.columns and "季度" in perf.columns:
        q = perf[(perf["年度"] == YEAR) & (perf["季度"] == QUARTER)].copy()
        if q.empty:
            print(
                f"提示: 绩效表中未找到 年度={YEAR} 且 季度={QUARTER} 的行，"
                "合并后绩效列为空。"
            )
    else:
        print("提示: 绩效表无「年度」「季度」列，将使用整张表作为 2024Q4 绩效来源。")
        q = perf.copy()

    q = q[["员工ID", "绩效评分"]].rename(columns={"绩效评分": SCORE_COL_OUT})
    if q["员工ID"].duplicated().any():
        print("警告: 存在重复员工ID，已去重保留首条")
        q = q.drop_duplicates(subset=["员工ID"], keep="first")
    return q


def main() -> None:
    for p in (FILE_BASIC, FILE_PERF):
        if not p.is_file():
            raise SystemExit(f"未找到文件: {p}（请与本脚本放在同一目录）")

    basic = pd.read_excel(FILE_BASIC, engine="openpyxl")
    perf = pd.read_excel(FILE_PERF, engine="openpyxl")

    _require_columns(basic, "员工基本信息表", REQUIRED_BASIC)
    _require_columns(perf, "员工绩效表", REQUIRED_PERF)

    q4 = extract_q4_perf(perf)
    merged = basic.merge(q4, on="员工ID", how="left")
    merged.to_excel(OUT_FILE, index=False, engine="openpyxl")

    print(f"已写入: {OUT_FILE}")
    print(f"行数: {len(merged)}, 列数: {len(merged.columns)}")
    missing = merged[SCORE_COL_OUT].isna().sum()
    if missing:
        print(f"提示: {int(missing)} 名员工未匹配到 2024Q4 绩效（{SCORE_COL_OUT} 为空）")


if __name__ == "__main__":
    main()
