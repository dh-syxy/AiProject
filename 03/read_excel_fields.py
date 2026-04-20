#!/usr/bin/env python
# coding: utf-8
"""读取本目录下的 Excel，打印工作表名、字段（列名）、类型与行数。"""

from pathlib import Path

import pandas as pd


def describe_excel(path: Path) -> None:
    print("=" * 60)
    print(f"文件: {path.name}")
    xl = pd.ExcelFile(path)
    print(f"工作表: {xl.sheet_names}")
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        print(f"\n  --- {sheet} ---")
        print(f"  行数: {len(df)}  列数: {len(df.columns)}")
        print("  字段:")
        for col in df.columns:
            print(f"    {col!r}  ->  {df[col].dtype}")


def main() -> None:
    base = Path(__file__).resolve().parent
    files = sorted(base.glob("*.xlsx"))
    if not files:
        print("未在 03 目录下找到 .xlsx 文件")
        return
    for p in files:
        describe_excel(p)
    print("=" * 60)


if __name__ == "__main__":
    main()
