# AI 量化配对交易系统

基于协整检验和Z-score均值回归的A股配对交易策略，覆盖全行业扫描、三年域外验证、实时监控与交易确认。

## 核心能力

1. **全行业配对发现** — 33个行业200+只股票，自动筛选协整配对（P<0.10）
2. **样本外两段验证** — 2023-2024 两年（24个月）做协整初筛，2025 全年 + 2026 至今两段做样本外回测，两段全赢才入选
3. **实时监控** — Z-score仪表盘、买卖信号生成、持仓状态管理
4. **单边轮动策略（方案三）** — 不做空，Z极端时买入被低估一方，回归时平仓

## 目录结构

```
AI_Economic/
├── pairs_scanner.py         # 全行业配对扫描器（发现+验证）
├── pairs_monitor.py         # 实时监控工具（信号+持仓管理）
├── trades.py                # 交易记录管理器（买卖+统计）
├── pairs_config.json        # 已验证配对配置（动态维护）
├── archive/                 # 归档（教学实验代码）
│   ├── lession1.py          # 双均线策略实验
│   ├── lession1.md          # 策略学习笔记
│   └── lession2_pairs_trading.py # 配对交易回测框架
├── monitor_states/          # 各配对独立持仓状态
├── data_cache/              # 股票日K数据缓存
└── charts/                  # 回测结果图表
```

## 快速开始

### 环境准备

```bash
cd AI_Economic
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy requests statsmodels
```

### 1. 扫描发现配对

```bash
python pairs_scanner.py
```

扫描 33 个行业 200+ 只股票，自动完成：
- 腾讯财经API批量获取 2023-01-01 ~ 2025-01-01（24个月）日K数据做初筛
- 同行业内两两协整检验（Engle-Granger, P<0.10, corr>0.5）
- 对通过的候选做 2025 全年 + 2026 年至今两段样本外回测
- 两段全部正收益的配对写入 `pairs_config.json`

### 2. 监控信号

```bash
# 列出所有可用配对
python pairs_monitor.py list

# 扫描默认配对
python pairs_monitor.py scan

# 扫描全部配对
python pairs_monitor.py scan --all

# 扫描指定配对
python pairs_monitor.py scan --pair 2
```

### 3. 确认交易

```bash
# 确认买入第0号配对
python pairs_monitor.py buy 0

# 确认卖出第0号配对
python pairs_monitor.py sell 0

# 重置所有持仓
python pairs_monitor.py reset
```

### 4. 交易记录管理

使用 `trades.py` 规范管理所有买卖记录：

```bash
# 按金额买入（自动计算股数）
python trades.py buy --pair 0 --date 2026-05-13 --price 34.58 --amount 10000 --stock 1

# 按股数买入
python trades.py buy --pair 1 --date 2026-05-13 --price 80.67 --shares 100 --stock 2 --note "Z=2.53"

# 卖出（不指定shares则全部卖出）
python trades.py sell --pair 0 --date 2026-05-20 --price 36.50 --shares 200 --stock 1

# 查看持仓
python trades.py list

# 查看历史交易
python trades.py history

# 统计汇总（胜率、盈亏等）
python trades.py summary

# 导出CSV
python trades.py export
```

参数说明：
- `--pair N`: 配对编号（通过 `pairs_monitor.py list` 查看）
- `--stock 1|2`: 买入/卖出配对中的哪只（1=Stock1, 2=Stock2）
- `--date`: 交易日期 `YYYY-MM-DD`
- `--price`: 成交价格
- `--shares` / `--amount`: 二选一，股数或金额

### 5. 自动模式

```bash
# 每个交易日15:05自动扫描
python pairs_monitor.py auto
```

## 策略逻辑

### 信号规则

| 条件 | 动作 | 含义 |
|------|------|------|
| Z-score < -2.0 | 买入Stock1 | Stock1相对低估 |
| Z-score > +2.0 | 买入Stock2 | Stock2相对低估 |
| \|Z-score\| < 0.5 | 平仓 | 价差回归均值 |

### 计算流程

1. **OLS回归** → 求对冲比率(hedge ratio)
2. **价差序列** → spread = P1 - (hedge × P2 + intercept)
3. **滚动Z-score** → (spread - 60日均值) / 60日标准差
4. **信号生成** → Z超阈值时触发买入/平仓

## 当前已验证配对

> 扫描日期 2026-05-13 17:15；股票池 **33 行业 / 200+ 只**；初筛窗口 2023-01-01~2025-01-01（24个月）；样本外验证 2025 全年 + 2026-01-01~2026-05-13。协整通过 86 对，两段全赢 **18 对**。

| # | 行业 | 配对 | 平均收益/段 | 最大回撤 | 夏普估算 | 2025收益 | 2026收益 |
|---|------|------|----------|----------|---------|---------|---------|
| 1 | 钢铁 | 河钢股份 vs 包钢股份 | +28.90% | 3.67% | 7.87 | +51.69% | +6.11% |
| 2 | 家电 | 九阳股份 vs 飞科电器 | +16.46% | 0.49% | 33.65 | +13.07% | +19.84% |
| 3 | 港口航运 | 中远海发 vs 中远海控 | +16.16% | 0.00% | — | +31.99% | +0.33% |
| 4 | 港口航运 | 招商轮船 vs 中远海能 | +14.73% | 3.31% | 4.45 | +29.34% | +0.13% |
| 5 | 保险 | 中国平安 vs 中国人寿 | +12.53% | 3.07% | 4.08 | +22.55% | +2.51% |
| 6 | 新能源 | 晶澳科技 vs 赣锋锂业 | +10.18% | 11.28% | 0.90 | +14.14% | +6.23% |
| 7 | 电力 | 华电国际 vs 深圳能源 | +9.83% | 0.20% | 48.86 | +11.70% | +7.96% |
| 8 | 通信 | 中国联通 vs 中兴通讯 | +7.80% | 1.10% | 7.12 | +7.01% | +8.60% |
| 9 | 券商 | 光大证券 vs 东兴证券 | +6.10% | 9.51% | 0.64 | +6.31% | +5.90% |
| 10 | 高速公路 | 山东高速 vs 赣粤高速 | +5.75% | 8.63% | 0.67 | +6.83% | +4.67% |
| 11 | 券商 | 光大证券 vs 长江证券 | +5.51% | 2.31% | 2.39 | +7.79% | +3.24% |
| 12 | 高速公路 | 山东高速 vs 四川成渝 | +5.43% | 0.00% | — | +9.11% | +1.75% |
| 13 | 券商 | 中信证券 vs 东兴证券 | +4.11% | 1.76% | 2.33 | +5.42% | +2.79% |
| 14 | 通信 | 中国联通 vs 星网锐捷 | +4.03% | 3.22% | 1.25 | +3.19% | +4.87% |
| 15 | 航空运输 | 南方航空 vs 海南航空 | +4.03% | 3.65% | 1.10 | +4.54% | +3.51% |
| 16 | 港口航运 | 上港集团 vs 中远海控 | +3.94% | 0.00% | — | +5.64% | +2.25% |
| 17 | 银行 | 农业银行 vs 南京银行 | +3.29% | 6.47% | 0.51 | +6.22% | +0.35% |
| 18 | 券商 | 华泰证券 vs 光大证券 | +2.29% | 7.47% | 0.31 | +1.41% | +3.17% |

## 配置文件说明

`pairs_config.json` 结构：

```json
{
  "scan_date": "2026-05-13 16:59",
  "scan_params": {
    "zscore_window": 60,
    "entry_z": 2.0,
    "exit_z": 0.5,
    "cointegration_threshold": 0.10,
    "train_window": "2023-01-01 ~ 2025-01-01",
    "validate_windows": [
      "2025-01-01 ~ 2026-01-01",
      "2026-01-01 ~ 2026-05-13"
    ]
  },
  "cointegration_passed": 34,
  "validated_count": 5,
  "pairs": [
    {
      "rank": 1,
      "sector": "钢铁",
      "c1": "000709", "c2": "600010",
      "n1": "河钢股份", "n2": "包钢股份",
      "p_value_train": 0.085653,
      "correlation_train": 0.8628,
      "avg_return_pct": 28.90,
      "max_drawdown_pct": 3.67,
      "sharpe_est": 7.87,
      "year_2025_return": 51.69,
      "year_2026_return": 6.11
    }
  ]
}
```

- `enabled`: 设为 `false` 可禁用某配对（如标的退市）
- 运行 `pairs_scanner.py` 可重新生成此文件
- `pairs_monitor.py` 启动时自动加载此配置

## 数据源

- **日K线数据**: 腾讯财经 `web.ifzq.gtimg.cn`（前复权）
- **实时行情**: 腾讯实时接口 `qt.gtimg.cn`
- **刷新时间**: 收盘后(15:00)逐步更新，建议16:30后使用

## 技术栈

- Python 3.12+
- pandas / numpy — 数据处理
- statsmodels — 协整检验(Engle-Granger) + OLS回归
- requests — 数据获取
- argparse — CLI管理

## 风险提示

- 本策略基于历史统计规律，不保证未来收益
- 协整关系可能随时间失效，建议每季度重新扫描验证
- 实际交易需考虑佣金、滑点、涨跌停等因素
- 建议单配对仓位不超过总资金的20%
