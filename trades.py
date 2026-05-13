"""
配对交易 - 交易记录管理器
==========================
以股票代码为主键管理所有买入/卖出记录。

用法：
  python trades.py buy  --code 601899 --date 2026-05-13 --price 34.58 --amount 10000
  python trades.py buy  --code 601899 --date 2026-05-13 --price 34.58 --shares 200
  python trades.py sell --code 601899 --date 2026-05-20 --price 36.50 --shares 200
  python trades.py sell --code 601899 --date 2026-05-20 --price 36.50  (默认全部卖出)
  python trades.py list                     # 查看所有持仓
  python trades.py list --code 601899       # 查看指定股票
  python trades.py history                  # 查看所有历史交易
  python trades.py history --code 601899    # 查看指定股票历史
  python trades.py summary                  # 统计汇总
  python trades.py export                   # 导出CSV
"""

import argparse
import json
import os
import csv
from datetime import datetime

# ============================================================
# 数据存储
# ============================================================

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trades_data.json')
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pairs_config.json')


def load_config():
    """加载配对配置，构建股票代码→名称映射"""
    if not os.path.exists(CONFIG_FILE):
        return {}, []
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    code_map = {}  # code -> {name, sector, pair_partner_code, pair_partner_name}
    pairs = []
    for p in config.get('pairs', []):
        if not p.get('enabled', True):
            continue
        pairs.append(p)
        code_map[p['c1']] = {
            'name': p['n1'], 'sector': p['sector'],
            'partner_code': p['c2'], 'partner_name': p['n2'],
        }
        code_map[p['c2']] = {
            'name': p['n2'], 'sector': p['sector'],
            'partner_code': p['c1'], 'partner_name': p['n1'],
        }
    return code_map, pairs


def load_trades():
    """加载交易数据"""
    if not os.path.exists(DATA_FILE):
        return {'trades': [], 'positions': {}}
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_trades(data):
    """保存交易数据"""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_stock(code, code_map):
    """解析股票代码，返回 (code, name, sector) 或 None"""
    if code in code_map:
        info = code_map[code]
        return code, info['name'], info['sector']
    # 尝试在配置中模糊匹配（用户可能输入不带前缀的代码）
    return code, f"未知({code})", "未知"


# ============================================================
# 命令：买入
# ============================================================

def cmd_buy(args):
    code_map, _ = load_config()
    stock_code, stock_name, sector = resolve_stock(args.code, code_map)
    data = load_trades()

    # 计算份数或金额
    if args.shares:
        shares = args.shares
        amount = round(args.price * shares, 2)
    elif args.amount:
        shares = int(args.amount / args.price)
        amount = round(args.price * shares, 2)
    else:
        print("❌ 必须指定 --shares（股数）或 --amount（金额）")
        return

    trade = {
        'id': len(data['trades']) + 1,
        'type': 'buy',
        'date': args.date,
        'stock_code': stock_code,
        'stock_name': stock_name,
        'sector': sector,
        'price': args.price,
        'shares': shares,
        'amount': amount,
        'note': args.note or '',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    data['trades'].append(trade)

    # 更新持仓（以股票代码为key）
    pos_key = stock_code
    if pos_key not in data['positions']:
        data['positions'][pos_key] = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'sector': sector,
            'shares': 0,
            'cost_total': 0,
            'avg_cost': 0,
            'first_buy_date': args.date,
        }
    pos = data['positions'][pos_key]
    # 更新名称（可能之前是"未知"）
    pos['stock_name'] = stock_name
    pos['sector'] = sector
    pos['shares'] += shares
    pos['cost_total'] = round(pos['cost_total'] + amount, 2)
    pos['avg_cost'] = round(pos['cost_total'] / pos['shares'], 4) if pos['shares'] > 0 else 0

    save_trades(data)

    print(f"✅ 买入成功")
    print(f"  标的: {stock_name}({stock_code}) [{sector}]")
    print(f"  日期: {args.date}")
    print(f"  价格: ¥{args.price}")
    print(f"  数量: {shares}股")
    print(f"  金额: ¥{amount:.2f}")
    print(f"  持仓: {pos['shares']}股, 均价¥{pos['avg_cost']:.4f}, 成本¥{pos['cost_total']:.2f}")


# ============================================================
# 命令：卖出
# ============================================================

def cmd_sell(args):
    code_map, _ = load_config()
    stock_code, stock_name, sector = resolve_stock(args.code, code_map)
    data = load_trades()

    pos_key = stock_code
    if pos_key not in data['positions'] or data['positions'][pos_key]['shares'] <= 0:
        print(f"❌ 无 {stock_name}({stock_code}) 持仓")
        return

    pos = data['positions'][pos_key]

    # 计算卖出份数
    if args.shares:
        shares = args.shares
    elif args.amount:
        shares = int(args.amount / args.price)
    else:
        shares = pos['shares']  # 默认全部卖出

    if shares > pos['shares']:
        print(f"❌ 卖出数量({shares})超过持仓({pos['shares']})")
        return

    amount = round(args.price * shares, 2)
    cost_basis = round(pos['avg_cost'] * shares, 2)
    pnl = round(amount - cost_basis, 2)
    pnl_pct = round((args.price - pos['avg_cost']) / pos['avg_cost'] * 100, 2)

    trade = {
        'id': len(data['trades']) + 1,
        'type': 'sell',
        'date': args.date,
        'stock_code': stock_code,
        'stock_name': stock_name,
        'sector': sector,
        'price': args.price,
        'shares': shares,
        'amount': amount,
        'cost_basis': cost_basis,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'note': args.note or '',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    data['trades'].append(trade)

    # 更新持仓
    pos['shares'] -= shares
    pos['cost_total'] = round(pos['cost_total'] - cost_basis, 2)
    if pos['shares'] <= 0:
        pos['shares'] = 0
        pos['cost_total'] = 0
        pos['avg_cost'] = 0

    save_trades(data)

    print(f"✅ 卖出成功")
    print(f"  标的: {stock_name}({stock_code}) [{sector}]")
    print(f"  日期: {args.date}")
    print(f"  价格: ¥{args.price}")
    print(f"  数量: {shares}股")
    print(f"  金额: ¥{amount:.2f}")
    print(f"  盈亏: ¥{pnl:+.2f} ({pnl_pct:+.2f}%)")
    if pos['shares'] > 0:
        print(f"  剩余: {pos['shares']}股, 均价¥{pos['avg_cost']:.4f}")
    else:
        print(f"  剩余: 已清仓")


# ============================================================
# 命令：查看持仓
# ============================================================

def cmd_list(args):
    data = load_trades()
    positions = data.get('positions', {})

    active = {k: v for k, v in positions.items() if v['shares'] > 0}
    if args.code:
        active = {k: v for k, v in active.items() if v['stock_code'] == args.code}

    if not active:
        print("📋 当前无持仓")
        return

    print(f"{'─' * 70}")
    print(f"  {'代码':<8} {'名称':<8} {'行业':<8} {'持仓':<8} {'均价':<10} {'成本':<12}")
    print(f"{'─' * 70}")
    total_cost = 0
    for key, pos in active.items():
        print(f"  {pos['stock_code']:<8} {pos['stock_name']:<6} {pos['sector']:<6} "
              f"{pos['shares']:<8} ¥{pos['avg_cost']:<9.3f} ¥{pos['cost_total']:<10.2f}")
        total_cost += pos['cost_total']
    print(f"{'─' * 70}")
    print(f"  总投入: ¥{total_cost:.2f}")


# ============================================================
# 命令：历史交易
# ============================================================

def cmd_history(args):
    data = load_trades()
    trades = data.get('trades', [])

    if args.code:
        trades = [t for t in trades if t['stock_code'] == args.code]

    if not trades:
        print("📋 暂无交易记录")
        return

    print(f"{'─' * 80}")
    print(f"  {'ID':<4} {'日期':<12} {'类型':<5} {'代码':<8} {'名称':<6} "
          f"{'价格':<9} {'数量':<7} {'盈亏':<10}")
    print(f"{'─' * 80}")
    for t in trades:
        type_str = '买入' if t['type'] == 'buy' else '卖出'
        pnl_str = f"¥{t.get('pnl', 0):+.2f}" if t['type'] == 'sell' else '-'
        print(f"  {t['id']:<4} {t['date']:<12} {type_str:<4} "
              f"{t['stock_code']:<8} {t['stock_name']:<6} "
              f"¥{t['price']:<8.2f} {t['shares']:<7} {pnl_str}")
    print(f"{'─' * 80}")

    # 统计卖出盈亏
    sells = [t for t in trades if t['type'] == 'sell']
    if sells:
        total_pnl = sum(t.get('pnl', 0) for t in sells)
        print(f"  已实现盈亏: ¥{total_pnl:+.2f}")


# ============================================================
# 命令：统计汇总
# ============================================================

def cmd_summary(args):
    data = load_trades()
    trades = data.get('trades', [])
    positions = data.get('positions', {})

    buys = [t for t in trades if t['type'] == 'buy']
    sells = [t for t in trades if t['type'] == 'sell']
    active = {k: v for k, v in positions.items() if v['shares'] > 0}

    total_invested = sum(t['amount'] for t in buys)
    total_sold = sum(t['amount'] for t in sells)
    realized_pnl = sum(t.get('pnl', 0) for t in sells)
    holding_cost = sum(v['cost_total'] for v in active.values())

    print(f"{'═' * 50}")
    print(f"  📊 交易统计汇总")
    print(f"{'═' * 50}")
    print(f"  总买入次数: {len(buys)}")
    print(f"  总卖出次数: {len(sells)}")
    print(f"  总买入金额: ¥{total_invested:.2f}")
    print(f"  总卖出金额: ¥{total_sold:.2f}")
    print(f"  已实现盈亏: ¥{realized_pnl:+.2f}")
    print(f"  当前持仓成本: ¥{holding_cost:.2f}")
    print(f"  当前持仓标的: {len(active)}只")
    print(f"{'═' * 50}")

    if sells:
        win_trades = [t for t in sells if t.get('pnl', 0) > 0]
        print(f"  胜率: {len(win_trades)}/{len(sells)} = {len(win_trades)/len(sells)*100:.1f}%")
        if win_trades:
            print(f"  平均盈利: ¥{sum(t['pnl'] for t in win_trades)/len(win_trades):.2f}")
        lose_trades = [t for t in sells if t.get('pnl', 0) <= 0]
        if lose_trades:
            print(f"  平均亏损: ¥{sum(t['pnl'] for t in lose_trades)/len(lose_trades):.2f}")


# ============================================================
# 命令：导出CSV
# ============================================================

def cmd_export(args):
    data = load_trades()
    trades = data.get('trades', [])

    if not trades:
        print("📋 暂无交易记录可导出")
        return

    export_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trades_export.csv')
    fields = ['id', 'type', 'date', 'stock_code', 'stock_name', 'sector',
              'price', 'shares', 'amount', 'pnl', 'pnl_pct', 'note']

    with open(export_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for t in trades:
            writer.writerow(t)

    print(f"✅ 已导出 {len(trades)} 条记录至: trades_export.csv")


# ============================================================
# 入口
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        prog='trades',
        description='配对交易记录管理器（以股票代码为主键）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', help='子命令')

    # buy
    p_buy = sub.add_parser('buy', help='记录买入')
    p_buy.add_argument('--code', type=str, required=True, help='股票代码 (如 601899)')
    p_buy.add_argument('--date', type=str, required=True, help='买入日期 (YYYY-MM-DD)')
    p_buy.add_argument('--price', type=float, required=True, help='买入价格')
    p_buy.add_argument('--shares', type=int, default=None, help='买入股数')
    p_buy.add_argument('--amount', type=float, default=None, help='买入金额（自动计算股数）')
    p_buy.add_argument('--note', type=str, default='', help='备注')

    # sell
    p_sell = sub.add_parser('sell', help='记录卖出')
    p_sell.add_argument('--code', type=str, required=True, help='股票代码')
    p_sell.add_argument('--date', type=str, required=True, help='卖出日期')
    p_sell.add_argument('--price', type=float, required=True, help='卖出价格')
    p_sell.add_argument('--shares', type=int, default=None, help='卖出股数（默认全部）')
    p_sell.add_argument('--amount', type=float, default=None, help='卖出金额')
    p_sell.add_argument('--note', type=str, default='', help='备注')

    # list
    p_list = sub.add_parser('list', help='查看当前持仓')
    p_list.add_argument('--code', type=str, default=None, help='筛选股票代码')

    # history
    p_hist = sub.add_parser('history', help='查看历史交易')
    p_hist.add_argument('--code', type=str, default=None, help='筛选股票代码')

    # summary
    sub.add_parser('summary', help='统计汇总')

    # export
    sub.add_parser('export', help='导出CSV')

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'buy':
        cmd_buy(args)
    elif args.command == 'sell':
        cmd_sell(args)
    elif args.command == 'list':
        cmd_list(args)
    elif args.command == 'history':
        cmd_history(args)
    elif args.command == 'summary':
        cmd_summary(args)
    elif args.command == 'export':
        cmd_export(args)
    else:
        parser.print_help()
