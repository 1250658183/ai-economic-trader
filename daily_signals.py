"""
每日信号摘要脚本 - 供 GitHub Actions 调用
========================================

复用 pairs_monitor.py 的核心函数，对 pairs_config.json 中所有启用的配对：
  1. 拉取最新行情
  2. 计算当前 Z-score
  3. 调用 generate_signal 判定 buy / sell / hold
  4. 输出三种结果：
     - stdout: 人读 markdown 表格
     - daily_signals.md: 当日报告（供 commit 入库）
     - daily_signals.json: 机器可读，含 has_signal 字段供 workflow 判断

退出码：始终 0（除非脚本异常）。是否有信号通过 has_signal 字段传递。
"""

import json
import os
import sys
from datetime import datetime

# 复用 pairs_monitor.py 的核心函数
from pairs_monitor import (
    PAIRS,
    ZSCORE_WINDOW,
    ENTRY_Z,
    EXIT_Z,
    fetch_latest_data,
    calculate_zscore,
    generate_signal,
    get_position_for_pair,
)

ROOT = os.path.dirname(os.path.abspath(__file__))


def evaluate_pair(pair):
    """对单个配对计算 Z-score 和信号，返回 dict"""
    try:
        df1 = fetch_latest_data(pair['c1'], pair['n1'])
        df2 = fetch_latest_data(pair['c2'], pair['n2'])
        z_series, hedge_ratio, _, _ = calculate_zscore(df1, df2, window=ZSCORE_WINDOW)
        z = z_series.dropna().iloc[-1]
        last_date = z_series.dropna().index[-1].strftime('%Y-%m-%d')
        p1 = df1['close'].iloc[-1]
        p2 = df2['close'].iloc[-1]

        position, _, _, _ = get_position_for_pair(pair)
        action, target, reason = generate_signal(z, position, pair['n1'], pair['n2'])

        return {
            'sector': pair['sector'],
            'n1': pair['n1'], 'c1': pair['c1'], 'p1': round(float(p1), 3),
            'n2': pair['n2'], 'c2': pair['c2'], 'p2': round(float(p2), 3),
            'z': round(float(z), 3),
            'date': last_date,
            'position': position,
            'action': action,
            'target': target,
            'reason': reason,
            'avg_ret': pair['avg_ret'],
            'error': None,
        }
    except Exception as e:
        return {
            'sector': pair['sector'],
            'n1': pair['n1'], 'c1': pair['c1'],
            'n2': pair['n2'], 'c2': pair['c2'],
            'error': str(e),
            'action': 'error',
        }


def render_markdown(results, generated_at):
    """生成 markdown 报告"""
    signals = [r for r in results if r['action'] in ('buy', 'sell')]
    holds = [r for r in results if r['action'] == 'hold']
    errors = [r for r in results if r['action'] == 'error']

    lines = []
    lines.append(f"# 📊 配对交易每日信号 · {generated_at}")
    lines.append("")
    lines.append(f"- 监控配对总数：**{len(results)}**")
    lines.append(f"- 触发买卖信号：**{len(signals)}**")
    lines.append(f"- 观望中：{len(holds)}")
    if errors:
        lines.append(f"- ⚠️ 数据异常：{len(errors)}")
    lines.append("")

    # 信号区
    if signals:
        lines.append("## 🚨 触发信号")
        lines.append("")
        lines.append("| 行业 | 配对 | Z-score | 操作 | 目标股 | 现价 | 原因 |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in signals:
            target_name = r['n1'] if r['target'] == 'stock1' else r['n2']
            target_code = r['c1'] if r['target'] == 'stock1' else r['c2']
            target_price = r['p1'] if r['target'] == 'stock1' else r['p2']
            emoji = '🟢' if r['action'] == 'buy' else '🔴'
            lines.append(
                f"| {r['sector']} | {r['n1']}({r['c1']})×{r['n2']}({r['c2']}) "
                f"| **{r['z']:+.2f}** | {emoji} {r['action'].upper()} "
                f"| {target_name}({target_code}) | ¥{target_price} | {r['reason']} |"
            )
        lines.append("")

    # 观望区（折叠）
    if holds:
        lines.append("<details><summary>📋 观望中配对（点击展开）</summary>")
        lines.append("")
        lines.append("| 行业 | 配对 | Z-score | 距入场 | 历史均收 |")
        lines.append("|---|---|---|---|---|")
        for r in holds:
            dist = ENTRY_Z - abs(r['z']) if abs(r['z']) < ENTRY_Z else 0
            lines.append(
                f"| {r['sector']} | {r['n1']}×{r['n2']} | {r['z']:+.2f} "
                f"| {dist:.2f} | +{r['avg_ret']}% |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # 异常区
    if errors:
        lines.append("## ⚠️ 数据异常")
        lines.append("")
        for r in errors:
            lines.append(f"- {r['sector']} {r['n1']}×{r['n2']}: `{r['error']}`")
        lines.append("")

    return "\n".join(lines)


def main():
    if not PAIRS:
        print("❌ pairs_config.json 中无启用的配对")
        sys.exit(1)

    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"开始扫描 {len(PAIRS)} 对配对...")

    results = [evaluate_pair(p) for p in PAIRS]

    md = render_markdown(results, generated_at)
    print(md)

    # 写文件
    with open(os.path.join(ROOT, 'daily_signals.md'), 'w', encoding='utf-8') as f:
        f.write(md)

    summary = {
        'generated_at': generated_at,
        'total': len(results),
        'has_signal': any(r['action'] in ('buy', 'sell') for r in results),
        'signals': [r for r in results if r['action'] in ('buy', 'sell')],
        'all': results,
    }
    with open(os.path.join(ROOT, 'daily_signals.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 写入 GitHub Actions Step Summary（如果在 CI 中）
    summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary_path:
        with open(summary_path, 'a', encoding='utf-8') as f:
            f.write(md)
        print(f"\n✓ 已写入 GITHUB_STEP_SUMMARY")


if __name__ == '__main__':
    main()
