"""
配对交易监控工具 - 方案三：单边轮动策略
========================================

信号规则：
  Z-score < -2.0  → 买入Stock1（Stock1相对低估）
  Z-score > +2.0  → 买入Stock2（Stock2相对低估）
  |Z-score| < 0.5 → 平仓

使用方法：
  python pairs_monitor.py scan              # 扫描默认配对
  python pairs_monitor.py scan --all        # 扫描全部配对
  python pairs_monitor.py scan --pair 1     # 扫描指定配对
  python pairs_monitor.py list              # 列出所有配对
  python pairs_monitor.py auto              # 每日15:05自动扫描

交易记录由 trades.py 独立管理：
  python trades.py buy  --code 601899 --date 2026-05-13 --price 34.58 --shares 200
  python trades.py sell --code 601899 --date 2026-05-20 --price 36.50
  python trades.py list
"""

import requests
import pandas as pd
import numpy as np
import os
import time
import json
from datetime import datetime, timedelta
import statsmodels.api as sm

# ============================================================
# 配置
# ============================================================

# 从 pairs_config.json 动态加载配对组合
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pairs_config.json')

def load_pairs_config():
    """从配置文件加载经过验证的配对组合"""
    if not os.path.exists(CONFIG_FILE):
        print(f"⚠️ 配置文件不存在: {CONFIG_FILE}")
        print(f"   请先运行 pairs_scanner.py 生成配置")
        return []
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    pairs = []
    for p in config.get('pairs', []):
        if not p.get('enabled', True):
            continue
        pairs.append({
            'c1': p['c1'], 'c2': p['c2'],
            'n1': p['n1'], 'n2': p['n2'],
            'sector': p['sector'],
            'avg_ret': p['avg_return_pct'],
            'note': p.get('note', f"平均+{p['avg_return_pct']}%/年, 回撤{p['max_drawdown_pct']}%"),
        })
    return pairs

PAIRS = load_pairs_config()

# 默认监控对（可通过 --pair 0/1/2/3 切换）
ACTIVE_PAIR_IDX = 0

ZSCORE_WINDOW = 60       # Z-score滚动窗口
ENTRY_Z = 2.0            # 开仓阈值
EXIT_Z = 0.5             # 平仓阈值
LOOKBACK_DAYS = 320      # 获取最近多少天数据

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')

# ============================================================
# 数据获取（腾讯财经接口，实时更新）
# ============================================================

def fetch_latest_data(code, name):
    """获取最新日K数据"""
    prefix = 'sz' if code.startswith('0') or code.startswith('3') else 'sh'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')

    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{prefix}{code},day,{start_date},{end_date},{LOOKBACK_DAYS},qfq"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            data = r.json()
            key = f"{prefix}{code}"
            klines = data["data"][key].get("qfqday") or data["data"][key].get("day")
            if not klines:
                raise ValueError("无K线数据")
            rows = [{"date": k[0], "close": float(k[2])} for k in klines]
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                raise RuntimeError(f"{name}({code}) 数据获取失败: {e}")


def get_realtime_price(code):
    """获取实时价格（盘中用）"""
    prefix = 'sz' if code.startswith('0') or code.startswith('3') else 'sh'
    url = f"https://qt.gtimg.cn/q={prefix}{code}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        parts = r.text.split('~')
        if len(parts) > 3:
            return float(parts[3])
    except:
        pass
    return None


# ============================================================
# 核心计算
# ============================================================

def calculate_zscore(df1, df2, window=60):
    """计算当前Z-score"""
    # 对齐日期
    common = df1.index.intersection(df2.index)
    p1 = df1.loc[common, 'close']
    p2 = df2.loc[common, 'close']

    # OLS回归求对冲比率
    X = sm.add_constant(p2)
    model = sm.OLS(p1, X).fit()
    hedge_ratio = model.params.iloc[1]
    intercept = model.params.iloc[0]

    # 价差
    spread = p1 - (hedge_ratio * p2 + intercept)

    # 滚动Z-score
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    z_score = (spread - spread_mean) / spread_std

    return z_score, hedge_ratio, intercept, spread


# ============================================================
# 持仓状态管理（从 trades_data.json 读取）
# ============================================================

TRADES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trades_data.json')

def get_position_for_pair(pair):
    """从trades_data.json读取某配对的持仓信息"""
    if not os.path.exists(TRADES_FILE):
        return 'empty', None, None, None
    with open(TRADES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    positions = data.get('positions', {})
    # 检查配对的两只股票是否有持仓
    c1, c2 = pair['c1'], pair['c2']
    for code in [c1, c2]:
        pos = positions.get(code, {})
        if pos.get('shares', 0) > 0:
            which = 'long_stock1' if code == c1 else 'long_stock2'
            return which, pos.get('avg_cost'), pos.get('shares'), pos.get('first_buy_date')
    return 'empty', None, None, None


# ============================================================
# 信号判断
# ============================================================

def generate_signal(z_score, current_position, stock1_name, stock2_name):
    """
    根据Z-score和当前持仓生成操作信号

    返回: (action, target, reason)
        action: 'buy', 'sell', 'hold'
        target: 'stock1', 'stock2', None
        reason: 信号原因描述
    """
    z = z_score

    if current_position == 'empty':
        if z < -ENTRY_Z:
            return 'buy', 'stock1', f'Z={z:.2f} < -{ENTRY_Z}，{stock1_name}相对低估'
        elif z > ENTRY_Z:
            return 'buy', 'stock2', f'Z={z:.2f} > +{ENTRY_Z}，{stock2_name}相对低估'
        else:
            return 'hold', None, f'Z={z:.2f}，无信号，继续观望'

    elif current_position == 'long_stock1':
        if z > -EXIT_Z:
            return 'sell', 'stock1', f'Z={z:.2f} > -{EXIT_Z}，价差回归，平仓{stock1_name}'
        else:
            return 'hold', None, f'Z={z:.2f}，继续持有{stock1_name}'

    elif current_position == 'long_stock2':
        if z < EXIT_Z:
            return 'sell', 'stock2', f'Z={z:.2f} < +{EXIT_Z}，价差回归，平仓{stock2_name}'
        else:
            return 'hold', None, f'Z={z:.2f}，继续持有{stock2_name}'

    return 'hold', None, '未知状态'


# ============================================================
# 主监控逻辑
# ============================================================

def run_monitor(pair_idx=0):
    """运行一次监控检测"""
    pair = PAIRS[pair_idx]
    STOCK1_CODE = pair['c1']
    STOCK2_CODE = pair['c2']
    STOCK1_NAME = pair['n1']
    STOCK2_NAME = pair['n2']
    pair_key = f"{STOCK1_CODE}_{STOCK2_CODE}"

    print("=" * 60)
    print(f"  📊 配对交易监控 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  [{pair['sector']}] {STOCK1_NAME}({STOCK1_CODE}) vs {STOCK2_NAME}({STOCK2_CODE})")
    print(f"  历史表现: {pair['note']}")
    print("=" * 60)

    # 获取数据
    print(f"\n[1] 获取最新数据...")
    df1 = fetch_latest_data(STOCK1_CODE, STOCK1_NAME)
    time.sleep(2)
    df2 = fetch_latest_data(STOCK2_CODE, STOCK2_NAME)

    latest_date = min(df1.index[-1], df2.index[-1])
    print(f"  数据截止: {latest_date.date()}", end='')
    if df1.index[-1] != df2.index[-1]:
        print(f"  ⚠️ {STOCK1_NAME}→{df1.index[-1].date()}, {STOCK2_NAME}→{df2.index[-1].date()}")
    else:
        print()
    print(f"  {STOCK1_NAME}最新价: ¥{df1['close'].iloc[-1]:.2f} ({df1.index[-1].date()})")
    print(f"  {STOCK2_NAME}最新价: ¥{df2['close'].iloc[-1]:.2f} ({df2.index[-1].date()})")

    # 计算Z-score
    print(f"\n[2] 计算Z-score (窗口={ZSCORE_WINDOW}天)...")
    z_series, hedge_ratio, intercept, spread = calculate_zscore(df1, df2, ZSCORE_WINDOW)

    current_z = z_series.iloc[-1]
    prev_z = z_series.iloc[-2] if len(z_series) > 1 else current_z
    z_change = current_z - prev_z

    print(f"  当前Z-score: {current_z:.3f} (较昨日 {z_change:+.3f})")
    print(f"  对冲比率: {hedge_ratio:.4f}")

    # Z-score仪表盘
    print(f"\n[3] Z-score 仪表盘:")
    print(f"  ─────────────────────────────────────────────")
    print(f"  买入{STOCK1_NAME}区  ←──── 观望区 ────→  买入{STOCK2_NAME}区")
    print(f"  [-3  -2  -1  0  +1  +2  +3]")

    # 可视化当前位置
    pos_indicator = int((current_z + 3) / 6 * 40)
    pos_indicator = max(0, min(39, pos_indicator))
    bar = '·' * 40
    bar = bar[:pos_indicator] + '▼' + bar[pos_indicator+1:]
    print(f"   {bar}")
    print(f"  ─────────────────────────────────────────────")

    # 加载持仓状态（从trades_data.json）
    current_position, entry_price, shares, entry_date = get_position_for_pair(pair)
    print(f"\n[4] 当前持仓: ", end='')
    if current_position == 'empty':
        print("空仓 🔲")
    elif current_position == 'long_stock1':
        print(f"持有{STOCK1_NAME} 🟢 ({shares}股, 均价¥{entry_price:.2f})")
    elif current_position == 'long_stock2':
        print(f"持有{STOCK2_NAME} 🟢 ({shares}股, 均价¥{entry_price:.2f})")

    # 生成信号
    action, target, reason = generate_signal(current_z, current_position, STOCK1_NAME, STOCK2_NAME)

    print(f"\n[5] 📢 交易信号:")
    print(f"  ─────────────────────────────────────────────")

    if action == 'buy':
        stock_name = STOCK1_NAME if target == 'stock1' else STOCK2_NAME
        stock_code = STOCK1_CODE if target == 'stock1' else STOCK2_CODE
        price = df1['close'].iloc[-1] if target == 'stock1' else df2['close'].iloc[-1]
        print(f"  🔔 【买入信号】{stock_name}({stock_code})")
        print(f"  💰 当前价格: ¥{price:.2f}")
        print(f"  📐 原因: {reason}")
        print(f"  ⚡ 建议: 明日开盘买入{stock_name}")
        print(f"  💡 确认: python trades.py buy --code {stock_code} --date <日期> --price <价格> --shares <股数>")

    elif action == 'sell':
        stock_name = STOCK1_NAME if target == 'stock1' else STOCK2_NAME
        stock_code = STOCK1_CODE if target == 'stock1' else STOCK2_CODE
        price = df1['close'].iloc[-1] if target == 'stock1' else df2['close'].iloc[-1]
        pnl = (price - entry_price) / entry_price * 100 if entry_price else 0

        print(f"  🔔 【卖出信号】{stock_name}({stock_code})")
        print(f"  💰 当前价格: ¥{price:.2f} (均价¥{entry_price:.2f}, 盈亏{pnl:+.2f}%)")
        print(f"  📐 原因: {reason}")
        print(f"  ⚡ 建议: 明日开盘卖出{stock_name}")
        print(f"  💡 确认: python trades.py sell --code {stock_code} --date <日期> --price <价格>")

    else:  # hold
        print(f"  ⏸️  【无操作】")
        print(f"  📐 {reason}")
        # 提示距离阈值的距离
        if current_position == 'empty':
            dist_buy1 = -ENTRY_Z - current_z
            dist_buy2 = current_z - ENTRY_Z
            if dist_buy1 > dist_buy2:
                print(f"  📏 距买入{STOCK2_NAME}信号: Z还差 {ENTRY_Z - current_z:.2f}")
            else:
                print(f"  📏 距买入{STOCK1_NAME}信号: Z还差 {abs(-ENTRY_Z - current_z):.2f}")

    print(f"  ─────────────────────────────────────────────")

    # 近5日Z-score趋势
    print(f"\n[6] 近5日Z-score趋势:")
    recent = z_series.tail(5)
    z_shifted = z_series.shift(1)
    for date, z in recent.items():
        prev = z_shifted.loc[date] if date in z_shifted.index and pd.notna(z_shifted.loc[date]) else z
        direction = "↑" if z > prev else "↓"
        print(f"  {date.date()}  Z={z:+.3f} {direction}")

    print(f"\n{'=' * 60}")
    return current_z, action


# ============================================================
# 定时运行模式
# ============================================================

def run_auto_mode():
    """每日定时运行模式"""
    import sched
    import time as t

    print("🔄 启动每日自动监控模式...")
    print("   每个交易日 15:05 自动检测")
    print("   按 Ctrl+C 退出\n")

    while True:
        now = datetime.now()
        # 判断是否交易日（简单判断：周一到周五）
        if now.weekday() < 5:
            # 15:05执行
            target = now.replace(hour=15, minute=5, second=0, microsecond=0)
            if now > target:
                target += timedelta(days=1)
            # 跳过周末
            while target.weekday() >= 5:
                target += timedelta(days=1)

            wait_seconds = (target - now).total_seconds()
            print(f"⏰ 下次检测: {target.strftime('%Y-%m-%d %H:%M')} (等待 {wait_seconds/3600:.1f}小时)")
            t.sleep(min(wait_seconds, 3600))  # 每小时醒来检查一次

            if datetime.now() >= target and datetime.now() < target + timedelta(minutes=5):
                run_monitor()
                t.sleep(300)  # 执行后等5分钟避免重复
        else:
            # 周末等待到周一
            t.sleep(3600)


# ============================================================
# 入口 (argparse)
# ============================================================

def build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog='pairs_monitor',
        description='配对交易监控工具 - 方案三单边轮动策略',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', help='子命令')

    p_scan = sub.add_parser('scan', help='扫描配对信号')
    p_scan.add_argument('--all', action='store_true', help='扫描全部配对')
    p_scan.add_argument('--pair', type=int, default=None, metavar='N', help='指定配对编号')

    sub.add_parser('list', help='列出所有配对组合')
    sub.add_parser('auto', help='每日15:05自动扫描模式')

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'scan':
        if args.all:
            print("=" * 60)
            print("  📊 全部配对快速扫描")
            print("=" * 60)
            for i in range(len(PAIRS)):
                print(f"\n{'─' * 60}")
                run_monitor(pair_idx=i)
        else:
            idx = args.pair if args.pair is not None else ACTIVE_PAIR_IDX
            run_monitor(pair_idx=idx)

    elif args.command == 'list':
        print("可用配对组合:")
        for i, p in enumerate(PAIRS):
            position, _, shares, _ = get_position_for_pair(p)
            if position == 'empty':
                pos_str = '空仓'
            else:
                held = p['n1'] if 'stock1' in position else p['n2']
                pos_str = f"持有{held}({shares}股)"
            print(f"  [{i}] [{p['sector']}] {p['n1']} vs {p['n2']} | {pos_str} | {p['note']}")
        print(f"\n用法:")
        print(f"  python pairs_monitor.py scan --pair <N>   扫描指定配对")
        print(f"  python pairs_monitor.py scan --all        扫描全部")
        print(f"  python trades.py buy --code <代码> ...    记录买入")
        print(f"  python trades.py sell --code <代码> ...   记录卖出")
        print(f"  python trades.py list                     查看持仓")

    elif args.command == 'auto':
        run_auto_mode()

    else:
        parser.print_help()
