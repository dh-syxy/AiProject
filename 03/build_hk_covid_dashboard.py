#!/usr/bin/env python
# coding: utf-8
"""从「香港各区疫情数据」Excel 生成 ECharts 可视化大屏 HTML。

包含：1) 每日新增 + 7 日滑动平均 + 累计确诊  2) 地图  3) 按日各区新增确诊堆叠柱  4) 环比增长率

首次运行会从网络下载香港 18 区 GeoJSON（阿里云 DataV），缓存到同目录 hk_geo_810000.json。

依赖：pip install pandas openpyxl

用法：python build_hk_covid_dashboard.py
输出：hk_covid_dashboard.html
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_XLSX = BASE_DIR / "香港各区疫情数据_20250322.xlsx"
GEO_URL = "https://geo.datav.aliyun.com/areas_v3/bound/810000_full.json"
GEO_CACHE = BASE_DIR / "hk_geo_810000.json"
OUT_HTML = BASE_DIR / "hk_covid_dashboard.html"

# 地图：黄框高风险区 = 选定「报告日期」当日新增确诊数前 N 名的地区
HOTSPOT_TOP_N = 3


def _ensure_geojson() -> dict:
    if GEO_CACHE.is_file():
        return json.loads(GEO_CACHE.read_text(encoding="utf-8"))
    req = urllib.request.Request(GEO_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    GEO_CACHE.write_text(raw, encoding="utf-8")
    return json.loads(raw)


def load_district_daily(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df["报告日期"] = pd.to_datetime(df["报告日期"], errors="coerce")
    df = df.dropna(subset=["报告日期"])
    return df


def aggregate_all(df: pd.DataFrame) -> tuple[list, list, list]:
    """全港按日：新增、累计。"""
    g = (
        df.groupby("报告日期", sort=True)
        .agg(新增=("新增确诊", "sum"), 累计=("累计确诊", "sum"))
        .reset_index()
    )
    dates = g["报告日期"].dt.strftime("%Y-%m-%d").tolist()
    daily = [int(x) for x in g["新增"].tolist()]
    cumul = [int(x) for x in g["累计"].tolist()]
    return dates, daily, cumul


def moving_avg(values: list[int], window: int = 7) -> list:
    s = pd.Series(values, dtype="float64")
    m = s.rolling(window=window, min_periods=1).mean()
    return [round(float(v), 2) for v in m.tolist()]


def growth_rate_vs_prev_day(daily: list[int]) -> list:
    """环比：(今日新增 - 昨日新增) / 昨日新增；昨日为 0 则为 null。"""
    out: list[float | None] = [None]
    for i in range(1, len(daily)):
        prev, cur = daily[i - 1], daily[i]
        if prev and prev > 0:
            out.append(round((cur - prev) / prev, 4))
        else:
            out.append(None)
    return out


def latest_date_map_payload(
    df: pd.DataFrame,
) -> tuple[str, list[dict], list[str]]:
    """返回：地图用日期、series.data（含高风险区描边样式）、高风险区名称列表。"""
    latest = df["报告日期"].max()
    d = df[df["报告日期"] == latest].copy()
    by_name = d.groupby("地区名称")["新增确诊"].sum()
    # GeoJSON properties.name 与 Excel 地区名称一致
    items = []
    hot_names: set[str] = set(
        by_name.nlargest(HOTSPOT_TOP_N).index.astype(str).tolist()
    )
    for name in by_name.index.astype(str):
        val = int(by_name[name])
        hot = name in hot_names
        item = {"name": name, "value": val}
        if hot:
            item["itemStyle"] = {
                "borderColor": "#fffc00",
                "borderWidth": 2,
                "shadowBlur": 18,
                "shadowColor": "rgba(255,200,0,0.8)",
            }
        items.append(item)
    return latest.strftime("%Y-%m-%d"), items, list(hot_names)


def build_html_payload(daily: list, cumul: list, dates: list) -> dict:
    ma7 = moving_avg(daily, 7)
    mom = growth_rate_vs_prev_day(daily)
    return {
        "dates": dates,
        "dailyNew": daily,
        "cumulative": cumul,
        "ma7": ma7,
        "momRate": mom,
    }


def district_daily_stack(df: pd.DataFrame) -> dict:
    """横轴为每日，序列：各区当日新增确诊（堆叠柱状图）。"""
    g = df.groupby(["报告日期", "地区名称"], as_index=False)["新增确诊"].sum()
    piv = g.pivot(index="报告日期", columns="地区名称", values="新增确诊").fillna(0).sort_index()
    dates = piv.index.strftime("%Y-%m-%d").tolist()
    cols = sorted(piv.columns.astype(str).tolist())
    series = [{"name": c, "data": [int(x) for x in piv[c].tolist()]} for c in cols]
    return {"districtDailyDates": dates, "districtSeries": series}


def render_html(geo_obj: dict, chart_data: dict, map_block: dict) -> str:
    geo_json_str = json.dumps(geo_obj, ensure_ascii=False)
    data_json = json.dumps(chart_data, ensure_ascii=False)
    map_json = json.dumps(map_block, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh-Hans">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>香港疫情可视化大屏</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #0a1628;
      color: #e8f4ff;
      font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
    }}
    .header {{
      text-align: center;
      padding: 12px 16px 8px;
      font-size: 22px;
      font-weight: 600;
      letter-spacing: 4px;
      background: linear-gradient(90deg, #0a1628, #132a45, #0a1628);
      border-bottom: 1px solid rgba(64,156,255,0.35);
    }}
    .sub {{
      font-size: 12px;
      color: #7eb8ff;
      letter-spacing: 1px;
      margin-top: 6px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      grid-template-rows: 320px 340px 300px;
      gap: 10px;
      padding: 10px;
      max-width: 1600px;
      margin: 0 auto;
    }}
    .panel {{
      background: rgba(13, 32, 58, 0.85);
      border: 1px solid rgba(64,156,255,0.25);
      border-radius: 8px;
      padding: 8px;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }}
    .panel-title {{
      font-size: 14px;
      color: #9fd0ff;
      padding: 4px 8px 8px;
      border-bottom: 1px solid rgba(64,156,255,0.15);
    }}
    .chart {{
      flex: 1;
      min-height: 0;
      width: 100%;
    }}
    .full {{ grid-column: 1 / -1; }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: 1fr; grid-template-rows: auto; }}
    }}
  </style>
</head>
<body>
  <div class="header">
    香港疫情可视化大屏
    <div class="sub">数据来源：香港各区疫情明细（按日汇总全港 &amp; 分区地图）</div>
  </div>
  <div class="grid">
    <div class="panel full">
      <div class="panel-title">1）确诊病例 · 全港每日新增、7日滑动平均与累计确诊</div>
      <div id="chart-overview" class="chart"></div>
    </div>
    <div class="panel">
      <div class="panel-title">2）地理分布 · 各区疫情（按日新增确诊填色：蓝→黄为低到高风险；黄框标识高风险地区）</div>
      <div id="chart-map" class="chart"></div>
    </div>
    <div class="panel">
      <div class="panel-title">3）各区每日确诊病例 · 横轴为日期，堆叠柱为各区当日新增确诊（人）</div>
      <div id="chart-district" class="chart"></div>
    </div>
    <div class="panel full">
      <div class="panel-title">4）增长率 · 每日新增相对前一日环比（%）</div>
      <div id="chart-mom" class="chart"></div>
    </div>
  </div>

  <script type="application/json" id="geo-json">{geo_json_str}</script>
  <script>
    const GEO_JSON = JSON.parse(document.getElementById('geo-json').textContent);
    const CHART_DATA = {data_json};
    const MAP_BLOCK = {map_json};
  </script>
  <script>
    function pct(x) {{
      if (x === null || x === undefined) return '';
      return (x * 100).toFixed(2) + '%';
    }}

    function initOverview() {{
      const el = document.getElementById('chart-overview');
      const chart = echarts.init(el);
      const d = CHART_DATA;
      chart.setOption({{
        backgroundColor: 'transparent',
        tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'cross' }} }},
        legend: {{
          data: ['每日新增确诊', '7日滑动平均', '累计确诊'],
          textStyle: {{ color: '#bde0ff' }},
          top: 8
        }},
        grid: {{ left: 56, right: 56, top: 56, bottom: 80 }},
        xAxis: {{
          type: 'category',
          data: d.dates,
          axisLabel: {{ color: '#8ab4d9', rotate: 30 }}
        }},
        yAxis: [
          {{ type: 'value', name: '新增', axisLabel: {{ color: '#7eb8ff' }}, splitLine: {{ lineStyle: {{ color: 'rgba(64,156,255,0.15)' }} }} }},
          {{ type: 'value', name: '累计', axisLabel: {{ color: '#ffb4a2' }}, splitLine: {{ show: false }} }}
        ],
        dataZoom: [
          {{ type: 'inside', start: 0, end: 100 }},
          {{ type: 'slider', start: 0, end: 100, bottom: 24, height: 22 }}
        ],
        series: [
          {{
            name: '每日新增确诊',
            type: 'bar',
            yAxisIndex: 0,
            data: d.dailyNew,
            itemStyle: {{ color: 'rgba(80,160,240,0.85)' }}
          }},
          {{
            name: '7日滑动平均',
            type: 'line',
            yAxisIndex: 0,
            data: d.ma7,
            smooth: true,
            symbol: 'circle',
            symbolSize: 3,
            lineStyle: {{ width: 2.2, color: '#f4d03f' }},
            z: 3
          }},
          {{
            name: '累计确诊',
            type: 'line',
            yAxisIndex: 1,
            data: d.cumulative,
            smooth: true,
            symbol: 'none',
            lineStyle: {{ width: 2, color: '#ff7e6b' }},
            areaStyle: {{ color: 'rgba(255,100,80,0.12)' }},
            z: 2
          }}
        ]
      }});
      return chart;
    }}

    function initMap() {{
      echarts.registerMap('HK18', GEO_JSON);
      const el = document.getElementById('chart-map');
      const chart = echarts.init(el);
      const mb = MAP_BLOCK;
      const vals = mb.seriesData.map(function (it) {{ return it.value; }});
      const vmax = Math.max.apply(null, vals.concat([1]));
      chart.setOption({{
        backgroundColor: 'transparent',
        tooltip: {{
          trigger: 'item',
          formatter: function (p) {{
            return p.name + '<br/>当日新增确诊: ' + (p.value || 0)
              + (mb.hotspots.indexOf(p.name) >= 0 ? '<br/><span style="color:#fffc00">高风险地区</span>' : '');
          }}
        }},
        visualMap: {{
          min: 0,
          max: vmax,
          text: ['高风险（黄）', '低风险（蓝）'],
          realtime: true,
          calculable: true,
          inRange: {{
            color: ['#0d47a1', '#1565c0', '#42a5f5', '#90caf9', '#fff9c4', '#fff176', '#ffeb3b', '#ffc107']
          }},
          textStyle: {{ color: '#bde0ff' }},
          left: 'left',
          bottom: 24
        }},
        series: [{{
          name: '新增确诊',
          type: 'map',
          map: 'HK18',
          roam: true,
          emphasis: {{
            label: {{ show: true, color: '#fff' }},
            itemStyle: {{ areaColor: '#4a9eff' }}
          }},
          data: mb.seriesData,
          nameMap: {{}}
        }}]
      }});
      return chart;
    }}

    function initDistrictCompare() {{
      const el = document.getElementById('chart-district');
      const chart = echarts.init(el);
      const d = CHART_DATA;
      const series = (d.districtSeries || []).map(function (s) {{
        return {{
          name: s.name,
          type: 'bar',
          stack: 'district',
          emphasis: {{ focus: 'series' }},
          data: s.data
        }};
      }});
      chart.setOption({{
        backgroundColor: 'transparent',
        tooltip: {{
          trigger: 'axis',
          axisPointer: {{ type: 'shadow' }},
          confine: true
        }},
        legend: {{
          type: 'scroll',
          orient: 'horizontal',
          bottom: 0,
          textStyle: {{ color: '#bde0ff', fontSize: 10 }},
          pageTextStyle: {{ color: '#7eb8ff' }},
          itemWidth: 12,
          itemHeight: 8
        }},
        grid: {{ left: 52, right: 20, top: 28, bottom: 56, containLabel: false }},
        xAxis: {{
          type: 'category',
          data: d.districtDailyDates || [],
          axisLabel: {{ color: '#8ab4d9', rotate: 28, fontSize: 9 }},
          boundaryGap: true
        }},
        yAxis: {{
          type: 'value',
          name: '新增确诊（人）',
          nameTextStyle: {{ color: '#9fd0ff', fontSize: 11 }},
          axisLabel: {{ color: '#7eb8ff' }},
          splitLine: {{ lineStyle: {{ color: 'rgba(64,156,255,0.12)' }} }}
        }},
        dataZoom: [
          {{ type: 'inside', xAxisIndex: 0 }},
          {{ type: 'slider', bottom: 28, height: 18 }}
        ],
        series: series
      }});
      return chart;
    }}

    function initMom() {{
      const el = document.getElementById('chart-mom');
      const chart = echarts.init(el);
      const d = CHART_DATA;
      const ratePct = d.momRate.map(function (x) {{ return x === null ? null : +(x * 100).toFixed(2); }});
      const colors = ratePct.map(function (v) {{
        if (v === null) return 'rgba(128,128,128,0.35)';
        return v >= 0 ? 'rgba(217,83,79,0.85)' : 'rgba(92,184,92,0.85)';
      }});
      chart.setOption({{
        backgroundColor: 'transparent',
        tooltip: {{
          trigger: 'axis',
          formatter: function (params) {{
            var p = params[0];
            if (p.value === null || p.value === undefined) return p.axisValue + '<br/>无环比（昨日新增为0）';
            return p.axisValue + '<br/>环比：' + p.value + '%';
          }}
        }},
        grid: {{ left: 52, right: 28, top: 36, bottom: 72 }},
        xAxis: {{
          type: 'category',
          data: d.dates,
          axisLabel: {{ color: '#8ab4d9', rotate: 28 }}
        }},
        yAxis: {{
          type: 'value',
          name: '环比 %',
          axisLabel: {{ color: '#7eb8ff', formatter: '{{value}}%' }},
          splitLine: {{ lineStyle: {{ color: 'rgba(64,156,255,0.12)' }} }}
        }},
        dataZoom: [{{ type: 'inside' }}, {{ type: 'slider', bottom: 8, height: 20 }}],
        series: [{{
          name: '新增环比',
          type: 'bar',
          data: ratePct.map(function (v, i) {{
            return {{ value: v, itemStyle: {{ color: colors[i] }} }};
          }}),
          barMaxWidth: 16
        }}]
      }});
      return chart;
    }}

    const charts = [];
    window.addEventListener('load', function () {{
      charts.push(initOverview());
      charts.push(initMap());
      charts.push(initDistrictCompare());
      charts.push(initMom());
    }});
    window.addEventListener('resize', function () {{
      charts.forEach(function (c) {{ c && c.resize(); }});
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    if not DATA_XLSX.is_file():
        raise SystemExit(f"未找到数据文件: {DATA_XLSX}")

    print("加载 GeoJSON …")
    geo_obj = _ensure_geojson()

    df = load_district_daily(DATA_XLSX)
    dates, daily, cumul = aggregate_all(df)

    chart_data = build_html_payload(daily, cumul, dates)
    chart_data.update(district_daily_stack(df))
    map_date, series_data, hotspots = latest_date_map_payload(df)
    map_block = {
        "reportDate": map_date,
        "seriesData": series_data,
        "hotspots": hotspots,
    }

    html = render_html(geo_obj, chart_data, map_block)
    OUT_HTML.write_text(html, encoding="utf-8")

    print(f"已生成: {OUT_HTML}")
    print(f"日期范围: {dates[0]} ~ {dates[-1]} ，共 {len(dates)} 日")
    print(f"地图使用日期: {map_date} ，高风险地区（黄框）: {', '.join(hotspots)}")


if __name__ == "__main__":
    main()
