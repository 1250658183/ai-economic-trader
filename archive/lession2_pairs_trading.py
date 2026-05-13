"""
Lesson 2: 配对交易策略（统计套利）
====================================
策略原理：
- 茅台(600519)与五粮液(000858)高度相关（同属白酒板块）
- 当它们的价格比率偏离历史均值时，做反向操作
- 等待比率回归均值时获利平仓

核心指标：Z-score = (当前价差 - 均值) / 标准差
- Z > +2: 价差过大 → 做空价差（卖茅台，买五粮液）
- Z < -2: 价差过小 → 做多价差（买茅台，卖五粮液）
- |Z| < 0.5: 价差回归 → 平仓
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# statsmodels 用于协整检验
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

# ============================================================
# 数据缓存目录
# ============================================================
DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')


# ============================================================
# 1. 加载数据
# ============================================================

def load_pair_data(stock1_file, stock2_file):
    """加载配对股票数据，对齐日期"""
    df1 = pd.read_csv(os.path.join(DATA_CACHE_DIR, stock1_file),
                      index_col='date', parse_dates=True)
    df2 = pd.read_csv(os.path.join(DATA_CACHE_DIR, stock2_file),
                      index_col='date', parse_dates=True)

    # 对齐日期（取交集）
    common_dates = df1.index.intersection(df2.index)
    df1 = df1.loc[common_dates].sort_index()
    df2 = df2.loc[common_dates].sort_index()

    return df1, df2


# ============================================================
# 2. 协整检验
# ============================================================

def cointegration_test(price1, price2):
    """
    Engle-Granger 协整检验
    - p值 < 0.05: 两只股票存在长期均衡关系，适合配对交易
    """
    score, pvalue, _ = coint(price1, price2)
    print(f"  协整检验统计量: {score:.4f}")
    print(f"  P值: {pvalue:.4f}")
    if pvalue < 0.05:
        print(f"  ✓ P值 < 0.05，存在协整关系，适合配对交易！")
        return True
    else:
        print(f"  ⚠️ P值 >= 0.05，协整关系不显著（仍可尝试）")
        return False


# ============================================================
# 3. 计算对冲比率和价差
# ============================================================

def calculate_spread(price1, price2, window=60):
    """
    计算价差和Z-score

    参数:
        price1: 股票1价格序列
        price2: 股票2价格序列
        window: 滚动窗口（用于计算动态均值和标准差）

    返回: DataFrame包含 spread, z_score
    """
    # OLS回归求对冲比率
    X = sm.add_constant(price2)
    model = sm.OLS(price1, X).fit()
    hedge_ratio = model.params.iloc[1]
    intercept = model.params.iloc[0]

    print(f"  对冲比率(hedge ratio): {hedge_ratio:.4f}")
    print(f"  即: 茅台 ≈ {hedge_ratio:.2f} × 五粮液 + {intercept:.1f}")

    # 计算价差（残差）
    spread = price1 - (hedge_ratio * price2 + intercept)

    # 滚动Z-score
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    z_score = (spread - spread_mean) / spread_std

    result = pd.DataFrame({
        'price1': price1,
        'price2': price2,
        'spread': spread,
        'spread_mean': spread_mean,
        'z_score': z_score
    }, index=price1.index)

    return result, hedge_ratio


# ============================================================
# 4. 生成交易信号
# ============================================================

def generate_pair_signals(df, entry_z=2.0, exit_z=0.5):
    """
    基于Z-score生成配对交易信号

    参数:
        entry_z: 开仓阈值（Z-score绝对值超过此值开仓）
        exit_z:  平仓阈值（Z-score绝对值小于此值平仓）

    信号:
        +1: 做多价差（买茅台，卖五粮液）→ 价差偏低时
        -1: 做空价差（卖茅台，买五粮液）→ 价差偏高时
         0: 空仓
    """
    df = df.copy()
    df['signal'] = 0.0

    position = 0  # 当前持仓状态
    for i in range(len(df)):
        z = df['z_score'].iloc[i]
        if pd.isna(z):
            continue

        if position == 0:  # 空仓
            if z > entry_z:  # 价差过大，做空价差
                position = -1
            elif z < -entry_z:  # 价差过小，做多价差
                position = 1
        elif position == 1:  # 持多头
            if z > -exit_z:  # 价差回归，平仓
                position = 0
        elif position == -1:  # 持空头
            if z < exit_z:  # 价差回归，平仓
                position = 0

        df.iloc[i, df.columns.get_loc('signal')] = position

    # 记录开仓/平仓动作
    df['position_change'] = df['signal'].diff().fillna(0)
    return df


# ============================================================
# 5. 回测引擎
# ============================================================

def backtest_pairs(df, hedge_ratio, initial_capital=100000.0):
    """
    配对交易回测

    做多价差(signal=+1): 买入茅台 + 卖出(hedge_ratio份)五粮液
    做空价差(signal=-1): 卖出茅台 + 买入(hedge_ratio份)五粮液

    使用价差的每日变化来计算盈亏
    """
    df = df.copy()
    capital = initial_capital
    portfolio_values = []

    # 使用价差的日变化来计算收益
    df['spread_change'] = df['spread'].diff().fillna(0)

    position = 0
    position_size = 0  # 持仓份数

    for i in range(len(df)):
        signal = df['signal'].iloc[i]
        spread_chg = df['spread_change'].iloc[i]
        price1 = df['price1'].iloc[i]

        # 持仓盈亏
        if position != 0:
            # 每份价差的盈亏 × 持仓方向 × 份数
            pnl = position * spread_chg * position_size
            capital += pnl

        # 信号变化时调整持仓
        if signal != position:
            if signal != 0 and position == 0:  # 开仓
                # 用50%资金做配对（按茅台价格计算份数）
                position_size = (initial_capital * 0.5) / price1
            elif signal == 0:  # 平仓
                position_size = 0
            position = signal

        portfolio_values.append(capital)

    df['portfolio_value'] = portfolio_values
    df['returns'] = pd.Series(portfolio_values).pct_change().values
    df['cumulative_returns'] = (df['portfolio_value'] / initial_capital - 1) * 100

    return df


# ============================================================
# 6. 策略评估
# ============================================================

def evaluate_pairs_strategy(df, initial_capital=100000.0):
    """计算配对交易策略绩效指标"""
    final_value = df['portfolio_value'].iloc[-1]
    total_return = final_value / initial_capital - 1
    n_days = len(df)
    annual_return = (1 + total_return) ** (250 / n_days) - 1

    # 最大回撤
    peak = pd.Series(df['portfolio_value'].values).cummax()
    drawdown = (df['portfolio_value'].values - peak) / peak
    max_drawdown = drawdown.min()

    # 夏普比率
    daily_returns = df['returns'].dropna()
    sharpe = np.sqrt(250) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0

    # 交易统计
    trades = df['position_change']
    n_open = (trades.abs() > 0).sum() // 2  # 每次开平仓算一次

    metrics = {
        '策略总收益率': f'{total_return:.2%}',
        '年化收益率': f'{annual_return:.2%}',
        '最大回撤': f'{max_drawdown:.2%}',
        '夏普比率': f'{sharpe:.2f}',
        '交易次数(开平仓)': n_open,
        '最终资金': f'¥{final_value:,.0f}',
    }
    return metrics


# ============================================================
# 7. 可视化
# ============================================================

def plot_pairs_results(df, save_path='pairs_trading_result.png'):
    """绘制配对交易回测结果"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # 图1: 两只股票价格（归一化）
    ax1 = axes[0]
    norm1 = df['price1'] / df['price1'].iloc[0] * 100
    norm2 = df['price2'] / df['price2'].iloc[0] * 100
    ax1.plot(df.index, norm1, label='Maotai (600519)', color='red', linewidth=1)
    ax1.plot(df.index, norm2, label='Wuliangye (000858)', color='blue', linewidth=1)
    ax1.set_title('Normalized Stock Prices (Base=100)')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Price (normalized)')
    ax1.grid(True, alpha=0.3)

    # 图2: Z-score + 交易信号
    ax2 = axes[1]
    ax2.plot(df.index, df['z_score'], color='purple', linewidth=0.8, label='Z-Score')
    ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Entry +2')
    ax2.axhline(y=-2, color='green', linestyle='--', alpha=0.5, label='Entry -2')
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=-0.5, color='gray', linestyle=':', alpha=0.5, label='Exit ±0.5')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    # 标记持仓区间
    ax2.fill_between(df.index, -3, 3,
                     where=df['signal'] == 1, alpha=0.1, color='green', label='Long spread')
    ax2.fill_between(df.index, -3, 3,
                     where=df['signal'] == -1, alpha=0.1, color='red', label='Short spread')
    ax2.set_title('Z-Score & Trading Signals')
    ax2.set_ylabel('Z-Score')
    ax2.set_ylim(-3.5, 3.5)
    ax2.legend(loc='upper right', fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3)

    # 图3: 累计收益
    ax3 = axes[2]
    ax3.plot(df.index, df['cumulative_returns'], color='green', linewidth=1.2, label='Pairs Strategy')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_title('Cumulative Returns (%)')
    ax3.set_ylabel('Returns (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4: 价差
    ax4 = axes[3]
    ax4.plot(df.index, df['spread'], color='orange', linewidth=0.8, label='Spread')
    ax4.plot(df.index, df['spread_mean'], color='black', linewidth=1, linestyle='--', label='Mean')
    ax4.set_title('Price Spread (Maotai - hedge_ratio * Wuliangye)')
    ax4.set_ylabel('Spread')
    ax4.set_xlabel('Date')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 回测图表已保存至: {save_path}")


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Lesson 2: 配对交易策略（茅台 vs 五粮液）")
    print("=" * 60)

    # ========== 配置 ==========
    STOCK1_FILE = 'akshare_600519_2024-01-01_2025-05-01.csv'  # 茅台
    STOCK2_FILE = 'akshare_000858_2024-01-01_2025-05-01.csv'  # 五粮液
    STOCK1_NAME = '贵州茅台(600519)'
    STOCK2_NAME = '五粮液(000858)'
    INITIAL_CAPITAL = 100000
    ZSCORE_WINDOW = 60   # Z-score滚动窗口
    ENTRY_Z = 2.0        # 开仓阈值
    EXIT_Z = 0.5         # 平仓阈值
    # ==========================

    # Step 1: 加载数据
    print(f"\n[1/5] 加载配对数据...")
    print(f"  股票A: {STOCK1_NAME}")
    print(f"  股票B: {STOCK2_NAME}")
    df1, df2 = load_pair_data(STOCK1_FILE, STOCK2_FILE)
    print(f"  共同交易日: {len(df1)} 天")
    print(f"  日期范围: {df1.index[0].date()} ~ {df1.index[-1].date()}")

    # Step 2: 协整检验
    print(f"\n[2/5] 协整检验...")
    is_coint = cointegration_test(df1['close'], df2['close'])

    # Step 3: 计算价差和Z-score
    print(f"\n[3/5] 计算价差与Z-score (窗口={ZSCORE_WINDOW}天)...")
    df, hedge_ratio = calculate_spread(df1['close'], df2['close'], window=ZSCORE_WINDOW)
    print(f"  Z-score范围: [{df['z_score'].min():.2f}, {df['z_score'].max():.2f}]")

    # Step 4: 生成信号并回测
    print(f"\n[4/5] 生成交易信号 (入场Z>{ENTRY_Z}, 出场Z<{EXIT_Z})...")
    df = generate_pair_signals(df, entry_z=ENTRY_Z, exit_z=EXIT_Z)
    df = df.dropna()

    # 统计信号
    long_days = (df['signal'] == 1).sum()
    short_days = (df['signal'] == -1).sum()
    flat_days = (df['signal'] == 0).sum()
    print(f"  做多价差天数: {long_days}天")
    print(f"  做空价差天数: {short_days}天")
    print(f"  空仓天数: {flat_days}天")

    # 回测
    print(f"\n  执行回测 (初始资金: ¥{INITIAL_CAPITAL:,})...")
    df = backtest_pairs(df, hedge_ratio, initial_capital=INITIAL_CAPITAL)

    # Step 5: 评估
    print(f"\n[5/5] 策略评估结果:")
    print("-" * 45)
    metrics = evaluate_pairs_strategy(df, INITIAL_CAPITAL)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("-" * 45)

    # 可视化
    plot_pairs_results(df, save_path='/root/AI_Economic/pairs_trading_result.png')

    print("\n✅ 配对交易回测完成！")
    print("\n💡 配对交易 vs 趋势跟踪:")
    print("  • 配对交易是市场中性策略，不依赖大盘涨跌方向")
    print("  • 利用两只相关股票的价差回归特性获利")
    print("  • 在震荡市中表现通常优于趋势策略")
    print("  • 风险：价差可能持续扩大不回归（结构性变化）")
