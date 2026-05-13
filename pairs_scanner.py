"""
配对交易组合扫描器 - 批量探查可行配对
=====================================
1. 批量获取候选股票数据
2. 同行业内两两做协整检验
3. 对通过的配对做2023/2024/2025三年回测
4. 将可用组合保存至 pairs_config.json
"""

import requests
import pandas as pd
import numpy as np
import os
import time
import json
from datetime import datetime, timedelta
from itertools import combinations
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# ============================================================
# 候选股票池（按行业分组）- 大幅扩展
# ============================================================

STOCK_POOL = {
    '银行': [
        ('601398', '工商银行'), ('601939', '建设银行'), ('601288', '农业银行'),
        ('601988', '中国银行'), ('600036', '招商银行'), ('601166', '兴业银行'),
        ('600000', '浦发银行'), ('000001', '平安银行'), ('601328', '交通银行'),
        ('002142', '宁波银行'), ('600016', '民生银行'), ('601169', '北京银行'),
        ('601818', '光大银行'), ('601998', '中信银行'), ('600015', '华夏银行'),
        ('601009', '南京银行'), ('002807', '江阴银行'),
    ],
    '券商': [
        ('600030', '中信证券'), ('601688', '华泰证券'), ('601211', '国泰君安'),
        ('600999', '招商证券'), ('000776', '广发证券'),
        ('601788', '光大证券'), ('600837', '海通证券'), ('000166', '申万宏源'),
        ('601377', '兴业证券'), ('601555', '东吴证券'), ('000783', '长江证券'),
        ('601198', '东兴证券'), ('002736', '国信证券'), ('601099', '太平洋'),
    ],
    '保险': [
        ('601318', '中国平安'), ('601628', '中国人寿'), ('601601', '中国太保'),
        ('601336', '新华保险'), ('601319', '中国人保'),
    ],
    '白酒': [
        ('600519', '贵州茅台'), ('000858', '五粮液'), ('000596', '古井贡酒'),
        ('002304', '洋河股份'), ('600809', '山西汾酒'), ('000568', '泸州老窖'),
        ('603369', '今世缘'), ('000799', '酒鬼酒'), ('603589', '口子窖'),
        ('000860', '顺鑫农业'),
    ],
    '家电': [
        ('000651', '格力电器'), ('000333', '美的集团'), ('002032', '苏泊尔'),
        ('600690', '海尔智家'), ('002050', '三花智控'), ('000521', '长虹美菱'),
        ('000100', 'TCL科技'), ('600839', '四川长虹'),
        ('002242', '九阳股份'), ('603868', '飞科电器'),
    ],
    '消费电子': [
        ('002475', '立讯精密'), ('002241', '歌尔股份'), ('000725', '京东方A'),
        ('002938', '鹏鼎控股'), ('002456', '欧菲光'), ('002436', '兴森科技'),
    ],
    '通信': [
        ('600050', '中国联通'), ('601728', '中国电信'), ('600941', '中国移动'),
        ('000063', '中兴通讯'), ('002396', '星网锐捷'),
    ],
    '石油化工': [
        ('600028', '中国石化'), ('601857', '中国石油'), ('600938', '中国海油'),
    ],
    '煤炭': [
        ('601088', '中国神华'), ('601898', '中煤能源'), ('600188', '兖矿能源'),
        ('601001', '大同煤业'), ('600395', '盘江股份'),
        ('601225', '陕西煤业'), ('601699', '潞安环能'), ('600157', '永泰能源'),
    ],
    '钢铁': [
        ('600019', '宝钢股份'), ('000709', '河钢股份'), ('600010', '包钢股份'),
        ('000898', '鞍钢股份'), ('600022', '山东钢铁'), ('002075', '沙钢股份'),
        ('600808', '马钢股份'), ('000825', '太钢不锈'), ('000932', '华菱钢铁'),
    ],
    '汽车': [
        ('600104', '上汽集团'), ('601238', '广汽集团'), ('000625', '长安汽车'),
        ('002594', '比亚迪'), ('601633', '长城汽车'), ('600733', '北汽蓝谷'),
        ('601127', '赛力斯'), ('000550', '江铃汽车'),
    ],
    '电力': [
        ('600900', '长江电力'), ('600886', '国投电力'), ('001289', '龙源电力'),
        ('600025', '华能水电'), ('600023', '浙能电力'), ('600011', '华能国际'),
        ('600795', '国电电力'), ('600027', '华电国际'), ('000027', '深圳能源'),
        ('601985', '中国核电'),
    ],
    '房地产': [
        ('001979', '招商蛇口'), ('600048', '保利发展'), ('000002', '万科A'),
        ('600383', '金地集团'), ('601155', '新城控股'),
        ('000069', '华侨城A'), ('000656', '金科股份'),
    ],
    '医药': [
        ('600276', '恒瑞医药'), ('000538', '云南白药'), ('600196', '复星医药'),
        ('002007', '华兰生物'), ('000963', '华东医药'), ('600436', '片仔癀'),
        ('600867', '通化东宝'), ('000739', '普洛药业'), ('000999', '华润三九'),
    ],
    '医疗器械': [
        ('300760', '迈瑞医疗'), ('002223', '鱼跃医疗'),
        ('300595', '欧普康视'), ('300406', '九强生物'),
    ],
    '食品饮料': [
        ('600887', '伊利股份'), ('002714', '牧原股份'), ('603288', '海天味业'),
        ('600600', '青岛啤酒'), ('000895', '双汇发展'), ('002557', '洽洽食品'),
        ('603198', '迎驾贡酒'),
    ],
    '航空运输': [
        ('600029', '南方航空'), ('601111', '中国国航'), ('600115', '东方航空'),
        ('002928', '华夏航空'), ('601021', '春秋航空'), ('600221', '海南航空'),
    ],
    '港口航运': [
        ('601872', '招商轮船'), ('600018', '上港集团'), ('601866', '中远海发'),
        ('601919', '中远海控'), ('601018', '宁波港'), ('600026', '中远海能'),
    ],
    '有色金属': [
        ('601899', '紫金矿业'), ('600489', '中金黄金'), ('601600', '中国铝业'),
        ('000630', '铜陵有色'), ('603993', '洛阳钼业'),
        ('600111', '北方稀土'), ('600547', '山东黄金'), ('000060', '中金岭南'),
        ('000878', '云南铜业'), ('600456', '宝钛股份'),
    ],
    '军工': [
        ('600893', '航发动力'), ('601989', '中国重工'), ('600760', '中航沈飞'),
        ('002179', '中航光电'), ('600150', '中国船舶'),
        ('000768', '中航西飞'), ('600879', '航天电子'),
    ],
    '半导体': [
        ('688981', '中芯国际'), ('002371', '北方华创'), ('603501', '韦尔股份'),
        ('002049', '紫光国微'), ('603986', '兆易创新'),
        ('300782', '卓胜微'), ('688012', '中微公司'), ('300661', '圣邦股份'),
    ],
    '新能源': [
        ('300750', '宁德时代'), ('601012', '隆基绿能'), ('600438', '通威股份'),
        ('002459', '晶澳科技'), ('688599', '天合光能'),
        ('002466', '天齐锂业'), ('002460', '赣锋锂业'), ('300274', '阳光电源'),
    ],
    '风电': [
        ('002202', '金风科技'), ('601016', '节能风电'), ('300129', '泰胜风能'),
        ('002080', '中材科技'),
    ],
    '化工': [
        ('600309', '万华化学'), ('600426', '华鲁恒升'), ('000792', '盐湖股份'),
        ('600352', '浙江龙盛'), ('002648', '卫星化学'), ('600989', '宝丰能源'),
    ],
    '建材': [
        ('600585', '海螺水泥'), ('000877', '天山股份'), ('600801', '华新水泥'),
        ('000401', '冀东水泥'), ('002233', '塔牌集团'),
    ],
    '工程机械': [
        ('000157', '中联重科'), ('600031', '三一重工'), ('600761', '安徽合力'),
        ('000425', '徐工机械'), ('603338', '浙江鼎力'),
    ],
    '家居': [
        ('002572', '索菲亚'), ('603833', '欧派家居'),
        ('603816', '顾家家居'), ('002818', '富森美'),
    ],
    '基建': [
        ('601668', '中国建筑'), ('601390', '中国中铁'), ('601186', '中国铁建'),
        ('601669', '中国电建'), ('601800', '中国交建'), ('600820', '隧道股份'),
    ],
    '农林牧渔': [
        ('000876', '新希望'), ('300498', '温氏股份'), ('002311', '海大集团'),
        ('000998', '隆平高科'),
    ],
    '传媒': [
        ('300413', '芒果超媒'), ('002027', '分众传媒'), ('000681', '视觉中国'),
        ('002292', '奥飞娱乐'),
    ],
    '物流': [
        ('002352', '顺丰控股'), ('600233', '圆通速递'), ('002120', '韵达股份'),
        ('603128', '华贸物流'),
    ],
    '旅游酒店': [
        ('600138', '中青旅'), ('600754', '锦江酒店'), ('600258', '首旅酒店'),
    ],
    '高速公路': [
        ('600350', '山东高速'), ('600020', '中原高速'), ('601107', '四川成渝'),
        ('600269', '赣粤高速'), ('600035', '楚天高速'),
    ],
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 数据获取
# ============================================================

def fetch_stock_data(code, name, start_date, end_date):
    """从腾讯财经获取日K数据"""
    prefix = 'sz' if code.startswith(('0', '3')) else 'sh'
    # 688开头也是上交所
    if code.startswith('688'):
        prefix = 'sh'

    cache_file = os.path.join(DATA_DIR, f"scan_{code}_{start_date}_{end_date}.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if len(df) > 20:
            return df

    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{prefix}{code},day,{start_date},{end_date},500,qfq"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            data = r.json()
            key = f"{prefix}{code}"
            klines = data["data"][key].get("qfqday") or data["data"][key].get("day")
            if not klines or len(klines) < 20:
                return None
            rows = [{"date": k[0], "close": float(k[2])} for k in klines]
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.to_csv(cache_file)
            return df
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"    ⚠️ {name}({code}) 获取失败: {e}")
                return None
    return None


# ============================================================
# 协整检验
# ============================================================

def test_cointegration(df1, df2):
    """协整检验，返回(p_value, correlation, hedge_ratio)"""
    common = df1.index.intersection(df2.index)
    if len(common) < 60:
        return None, None, None
    p1 = df1.loc[common, 'close'].values
    p2 = df2.loc[common, 'close'].values
    corr = np.corrcoef(p1, p2)[0, 1]
    try:
        _, p_value, _ = coint(p1, p2)
    except:
        return None, None, None
    # 对冲比率
    X = sm.add_constant(p2)
    model = sm.OLS(p1, X).fit()
    hedge_ratio = model.params[1]
    return p_value, corr, hedge_ratio


# ============================================================
# 回测引擎
# ============================================================

def backtest_pair(df1, df2, window=60, entry_z=2.0, exit_z=0.5):
    """简化版配对交易回测，返回年化收益和最大回撤"""
    common = df1.index.intersection(df2.index)
    if len(common) < window + 20:
        return None, None, None
    p1 = df1.loc[common, 'close']
    p2 = df2.loc[common, 'close']

    # OLS
    X = sm.add_constant(p2)
    model = sm.OLS(p1, X).fit()
    hedge_ratio = model.params.iloc[1]
    intercept = model.params.iloc[0]
    spread = p1 - (hedge_ratio * p2 + intercept)

    # Z-score
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    z_score = (spread - spread_mean) / spread_std
    z_score = z_score.dropna()

    if len(z_score) < 20:
        return None, None, None

    # 模拟单边轮动
    position = 'empty'  # 'empty', 'long1', 'long2'
    entry_price = 0
    capital = 1.0
    peak = 1.0
    max_dd = 0

    for i in range(len(z_score)):
        date = z_score.index[i]
        z = z_score.iloc[i]
        price1 = p1.loc[date]
        price2 = p2.loc[date]

        if position == 'empty':
            if z < -entry_z:
                position = 'long1'
                entry_price = price1
            elif z > entry_z:
                position = 'long2'
                entry_price = price2
        elif position == 'long1':
            if z > -exit_z:
                pnl = (price1 - entry_price) / entry_price
                capital *= (1 + pnl)
                position = 'empty'
        elif position == 'long2':
            if z < exit_z:
                pnl = (price2 - entry_price) / entry_price
                capital *= (1 + pnl)
                position = 'empty'

        peak = max(peak, capital)
        dd = (peak - capital) / peak
        max_dd = max(max_dd, dd)

    # 处理未平仓
    if position == 'long1':
        pnl = (p1.iloc[-1] - entry_price) / entry_price
        capital *= (1 + pnl)
    elif position == 'long2':
        pnl = (p2.iloc[-1] - entry_price) / entry_price
        capital *= (1 + pnl)

    peak = max(peak, capital)
    dd = (peak - capital) / peak
    max_dd = max(max_dd, dd)

    total_return = (capital - 1) * 100
    # 年化（按天数估算）
    days = (z_score.index[-1] - z_score.index[0]).days
    if days > 0:
        annual_return = total_return * 365 / days
    else:
        annual_return = total_return

    return total_return, max_dd * 100, annual_return


# ============================================================
# 主流程
# ============================================================

def scan_all_pairs():
    """扫描所有行业配对"""
    print("=" * 70)
    print("  🔍 配对交易全行业扫描器")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # 统计
    total_stocks = sum(len(v) for v in STOCK_POOL.values())
    total_sectors = len(STOCK_POOL)
    print(f"\n  候选池: {total_sectors}个行业, {total_stocks}只股票")

    # ===== 阶段1: 获取2023-2024两年数据做协整筛选 =====
    print(f"\n{'─' * 70}")
    print("  📥 阶段1: 获取2023-01~2024-12 两年数据 & 协整筛选 (24个月样本)")
    print(f"{'─' * 70}")

    sector_data_train = {}
    for sector, stocks in STOCK_POOL.items():
        print(f"\n  [{sector}] 获取 {len(stocks)} 只股票...")
        sector_data_train[sector] = {}
        for code, name in stocks:
            df = fetch_stock_data(code, name, '2023-01-01', '2025-01-01')
            if df is not None and len(df) > 60:
                sector_data_train[sector][code] = (name, df)
                print(f"    ✓ {name}({code}) {len(df)}条")
            else:
                print(f"    ✗ {name}({code}) 数据不足")
            time.sleep(0.5)

    # 协整检验
    print(f"\n{'─' * 70}")
    print("  🧪 阶段2: 同行业协整检验 (P<0.10, 基于2023-2024两年数据)")
    print(f"{'─' * 70}")

    candidates = []
    for sector, stock_dict in sector_data_train.items():
        codes = list(stock_dict.keys())
        if len(codes) < 2:
            continue
        pairs_tested = 0
        for c1, c2 in combinations(codes, 2):
            n1, df1 = stock_dict[c1]
            n2, df2 = stock_dict[c2]
            p_val, corr, hedge = test_cointegration(df1, df2)
            pairs_tested += 1
            if p_val is not None and p_val < 0.10 and corr is not None and corr > 0.5:
                candidates.append({
                    'sector': sector, 'c1': c1, 'c2': c2,
                    'n1': n1, 'n2': n2,
                    'p_value': p_val, 'correlation': corr, 'hedge_ratio': hedge,
                })
        print(f"  [{sector}] 测试{pairs_tested}对, 通过: {sum(1 for c in candidates if c['sector']==sector)}对")

    candidates.sort(key=lambda x: x['p_value'])
    print(f"\n  📊 协整通过(P<0.10): {len(candidates)}对")
    for c in candidates:
        print(f"    {c['sector']:6s} | {c['n1']} vs {c['n2']} | P={c['p_value']:.4f} | r={c['correlation']:.3f}")

    # ===== 阶段3: 样本外两段验证 (2025全年 + 2026至今) =====
    print(f"\n{'─' * 70}")
    print("  📈 阶段3: 样本外两段验证 (2025-01~2025-12 / 2026-01~2026-05)")
    print(f"{'─' * 70}")

    validated_pairs = []
    for idx, cand in enumerate(candidates):
        c1, c2, n1, n2 = cand['c1'], cand['c2'], cand['n1'], cand['n2']
        print(f"\n  [{idx+1}/{len(candidates)}] {n1} vs {n2} ({cand['sector']})")

        results = {}
        all_positive = True
        for period_name, start, end in [
            ('2025', '2025-01-01', '2026-01-01'),
            ('2026', '2026-01-01', '2026-05-13'),
        ]:
            df1 = fetch_stock_data(c1, n1, start, end)
            time.sleep(0.5)
            df2 = fetch_stock_data(c2, n2, start, end)
            time.sleep(0.5)
            if df1 is None or df2 is None:
                print(f"    {period_name}: 数据缺失 ✗")
                all_positive = False
                results[period_name] = None
                continue
            ret, dd, annual = backtest_pair(df1, df2)
            if ret is None:
                print(f"    {period_name}: 回测失败 ✗")
                all_positive = False
                results[period_name] = None
            else:
                results[period_name] = {'return': ret, 'max_dd': dd, 'annual': annual}
                status = '✓' if ret > 0 else '✗'
                print(f"    {period_name}: {ret:+.2f}% (回撤{dd:.2f}%) {status}")
                if ret <= 0:
                    all_positive = False

        if all_positive and all(v is not None for v in results.values()):
            avg_ret = np.mean([r['return'] for r in results.values()])
            max_dd = max(r['max_dd'] for r in results.values())
            validated_pairs.append({
                'sector': cand['sector'],
                'c1': c1, 'c2': c2, 'n1': n1, 'n2': n2,
                'p_value_train': cand['p_value'],
                'correlation_train': cand['correlation'],
                'results': results,
                'avg_return': round(avg_ret, 2),
                'max_drawdown': round(max_dd, 2),
                'sharpe_est': round(avg_ret / max_dd, 2) if max_dd > 0 else 99,
            })
            print(f"    ✅ 两段全赢！平均+{avg_ret:.2f}%/段, 最大回撤{max_dd:.2f}%")

    # ===== 阶段4: 保存结果 =====
    print(f"\n{'─' * 70}")
    print("  💾 阶段4: 保存结果")
    print(f"{'─' * 70}")

    # 按平均收益排序
    validated_pairs.sort(key=lambda x: -x['avg_return'])

    output = {
        'scan_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'scan_params': {
            'zscore_window': 60,
            'entry_z': 2.0,
            'exit_z': 0.5,
            'cointegration_threshold': 0.10,
            'train_window': '2023-01-01 ~ 2025-01-01',
            'validate_windows': ['2025-01-01 ~ 2026-01-01', '2026-01-01 ~ 2026-05-13'],
        },
        'total_sectors': total_sectors,
        'total_stocks': total_stocks,
        'cointegration_passed': len(candidates),
        'validated_count': len(validated_pairs),
        'pairs': [],
    }

    for i, p in enumerate(validated_pairs):
        pair_entry = {
            'rank': i + 1,
            'sector': p['sector'],
            'c1': p['c1'],
            'c2': p['c2'],
            'n1': p['n1'],
            'n2': p['n2'],
            'p_value_train': round(p['p_value_train'], 6),
            'correlation_train': round(p['correlation_train'], 4),
            'avg_return_pct': p['avg_return'],
            'max_drawdown_pct': p['max_drawdown'],
            'sharpe_est': p['sharpe_est'],
            'year_2025_return': round(p['results']['2025']['return'], 2),
            'year_2026_return': round(p['results']['2026']['return'], 2),
        }
        output['pairs'].append(pair_entry)

    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pairs_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ 结果已保存至: pairs_config.json")
    print(f"\n  📊 最终结果: {len(validated_pairs)} 对通过两段样本外验证")
    print(f"  {'─' * 50}")
    print(f"  {'排名':<4} {'行业':<6} {'配对':<20} {'平均收益':<10} {'最大回撤':<10}")
    print(f"  {'─' * 50}")
    for p in output['pairs']:
        print(f"  {p['rank']:<4} {p['sector']:<6} {p['n1']}vs{p['n2']:<14} +{p['avg_return_pct']:.2f}%    {p['max_drawdown_pct']:.2f}%")
    print(f"  {'─' * 50}")

    return output


if __name__ == '__main__':
    scan_all_pairs()
