"""
Lesson 1: 简单双均线交叉策略 + 回测
====================================
策略逻辑：
- 短期均线(MA5)上穿长期均线(MA20) → 买入信号
- 短期均线(MA5)下穿长期均线(MA20) → 卖出信号

使用模拟数据演示完整的量化回测流程。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 数据缓存目录
DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ============================================================
# 1. 获取股票数据（支持真实数据 + 本地缓存 + 模拟数据兜底）
# ============================================================

def get_real_stock_data(symbol='AAPL', start='2024-01-01', end='2025-01-01', source='yfinance'):
    """
    获取真实股票数据（优先从本地缓存读取）

    参数:
        symbol: 股票代码
            - yfinance: 美股如 'AAPL', 'TSLA', 'MSFT'; 港股如 '0700.HK'
            - akshare:  A股如 '000001'(平安银行), '600519'(贵州茅台)
        start: 起始日期 'YYYY-MM-DD'
        end: 结束日期 'YYYY-MM-DD'
        source: 数据源 'yfinance' 或 'akshare'

    返回: DataFrame，包含 'close' 列，以日期为索引
    """
    # --- 缓存机制：检查本地是否已有数据 ---
    cache_file = os.path.join(DATA_CACHE_DIR, f"{source}_{symbol}_{start}_{end}.csv")
    if os.path.exists(cache_file):
        print(f"  📁 从本地缓存加载: {os.path.basename(cache_file)}")
        df = pd.read_csv(cache_file, index_col='date', parse_dates=True)
        print(f"  ✓ 缓存加载成功！共 {len(df)} 条数据")
        return df
    if source == 'yfinance':
        import yfinance as yf
        print(f"  正在从 Yahoo Finance 获取 {symbol} 数据...")
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start, end=end)
        if hist.empty:
            raise ValueError(f"未获取到 {symbol} 的数据，请检查股票代码或网络")
        df = pd.DataFrame({'close': hist['Close']})
        df.index.name = 'date'
        df.index = df.index.tz_localize(None)  # 去掉时区信息

    elif source == 'akshare':
        import akshare as ak
        import time
        print(f"  正在从 akshare 获取 A股 {symbol} 数据...")
        # akshare 日期格式为 YYYYMMDD
        start_fmt = start.replace('-', '')
        end_fmt = end.replace('-', '')

        # 重试机制：最多重试3次，每次间隔递增
        hist = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                hist = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                           start_date=start_fmt, end_date=end_fmt, adjust="qfq")
                break  # 成功则跳出
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s 递增等待
                    print(f"  ⏳ 第{attempt+1}次请求失败，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise e  # 最后一次仍失败则抛出

        if hist is None or hist.empty:
            raise ValueError(f"未获取到 {symbol} 的数据，请检查股票代码")
        df = pd.DataFrame({
            'date': pd.to_datetime(hist['日期']),
            'close': hist['收盘'].values
        })
        df.set_index('date', inplace=True)
    else:
        raise ValueError(f"不支持的数据源: {source}")

    print(f"  ✓ 获取成功！共 {len(df)} 条数据")

    # --- 保存到本地缓存 ---
    df.to_csv(cache_file)
    print(f"  💾 已缓存至: {os.path.basename(cache_file)}")

    return df


def generate_stock_data(days=250, initial_price=100.0, seed=42):
    """生成模拟的每日股票价格数据（作为无网络时的兜底）"""
    np.random.seed(seed)
    daily_returns = np.random.normal(0.0005, 0.015, days)
    prices = initial_price * np.cumprod(1 + daily_returns)

    dates = pd.date_range(start='2025-01-01', periods=days, freq='B')
    df = pd.DataFrame({
        'date': dates[:len(prices)],
        'close': prices
    })
    df.set_index('date', inplace=True)
    return df


# ============================================================
# 2. 计算技术指标
# ============================================================

def compute_moving_averages(df, short_window=5, long_window=20):
    """计算短期和长期移动平均线"""
    df = df.copy()
    df['MA_short'] = df['close'].rolling(window=short_window).mean()
    df['MA_long'] = df['close'].rolling(window=long_window).mean()
    return df


def compute_rsi(df, period=14):
    """计算RSI（相对强弱指标）"""
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def compute_atr(df, period=14):
    """计算ATR（平均真实波幅），用于动态止损"""
    df = df.copy()
    high = df['close'].rolling(2).max()  # 简化：用收盘价模拟
    low = df['close'].rolling(2).min()
    tr = high - low
    df['ATR'] = tr.rolling(window=period).mean()
    return df


# ============================================================
# 3. 生成交易信号（基础版）
# ============================================================

def generate_signals(df):
    """
    基于双均线交叉生成交易信号
    signal:  1 = 持有多头, 0 = 空仓
    position: 信号变化时产生买卖动作
    """
    df = df.copy()
    df['signal'] = 0
    # 短期均线在长期均线之上时持有
    df.loc[df['MA_short'] > df['MA_long'], 'signal'] = 1
    # position变化: 1=买入, -1=卖出, 0=持有不变
    df['position'] = df['signal'].diff()
    return df


# ============================================================
# 3b. 生成交易信号（增强版：RSI过滤 + 趋势确认）
# ============================================================

def generate_signals_enhanced(df, rsi_buy_threshold=45, rsi_sell_threshold=70,
                               ma_gap_threshold=0.005):
    """
    增强版信号生成：
    - RSI过滤: 仅在RSI<45（非超买）时买入，RSI>70（超买）时加速卖出
    - 趋势确认: 短均线必须高于长均线至少0.5%才确认买入（避免假金叉）
    - 卖出条件: 死叉 或 RSI超买
    """
    df = df.copy()
    df['signal'] = 0

    for i in range(1, len(df)):
        prev_signal = df['signal'].iloc[i - 1]
        ma_short = df['MA_short'].iloc[i]
        ma_long = df['MA_long'].iloc[i]
        rsi = df['RSI'].iloc[i] if 'RSI' in df.columns else 50

        # 计算均线间距比例
        ma_gap = (ma_short - ma_long) / ma_long if ma_long != 0 else 0

        if prev_signal == 0:  # 当前空仓
            # 买入条件：金叉确认（间距>阈值）+ RSI未超买
            if ma_gap > ma_gap_threshold and rsi < rsi_buy_threshold:
                df.iloc[i, df.columns.get_loc('signal')] = 1
        else:  # 当前持仓
            # 卖出条件：死叉 或 RSI超买
            if ma_short < ma_long or rsi > rsi_sell_threshold:
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1

    df['position'] = df['signal'].diff()
    return df


# ============================================================
# 3c. 自适应策略（根据市场状态动态调整参数）
# ============================================================

def compute_market_regime(df, lookback=60):
    """
    判断市场状态（趋势/震荡）
    - 用长期均线斜率 + 波动率来判断
    - regime: 'trend_up'(上升趋势), 'trend_down'(下降趋势), 'range'(震荡)
    """
    df = df.copy()
    # 长均线斜率（20日均线的变化率）
    df['MA_slope'] = df['MA_long'].pct_change(periods=10) * 100  # 10日变化百分比
    # 波动率（20日收益率标准差）
    df['volatility'] = df['close'].pct_change().rolling(20).std() * 100
    # ADX简化版：用均线斜率绝对值衡量趋势强度
    df['trend_strength'] = df['MA_slope'].rolling(5).mean().abs()
    return df


def generate_signals_adaptive(df, base_rsi_buy=55, base_rsi_sell=75,
                               base_ma_gap=0.003):
    """
    自适应策略：根据市场状态动态调整买卖条件

    核心逻辑：
    - 上升趋势中：放宽买入条件（RSI阈值提高到60），紧跟趋势
    - 震荡/下降中：收紧买入条件（RSI阈值降到40），减少交易
    - 持仓中：根据趋势强度动态调整卖出容忍度
    """
    df = df.copy()
    df['signal'] = 0
    df['regime'] = 'range'  # 默认震荡

    for i in range(1, len(df)):
        prev_signal = df['signal'].iloc[i - 1]
        ma_short = df['MA_short'].iloc[i]
        ma_long = df['MA_long'].iloc[i]
        rsi = df['RSI'].iloc[i] if 'RSI' in df.columns else 50
        ma_slope = df['MA_slope'].iloc[i] if 'MA_slope' in df.columns else 0
        trend_str = df['trend_strength'].iloc[i] if 'trend_strength' in df.columns else 0

        # --- 判断市场状态 ---
        if ma_slope > 0.5 and trend_str > 0.3:
            regime = 'trend_up'
        elif ma_slope < -0.5 and trend_str > 0.3:
            regime = 'trend_down'
        else:
            regime = 'range'
        df.iloc[i, df.columns.get_loc('regime')] = regime

        # --- 动态参数调整 ---
        if regime == 'trend_up':
            # 上升趋势：放宽买入（更容易进场），收紧卖出（更不容易被震出）
            rsi_buy = base_rsi_buy + 10    # 65: 允许RSI较高时也买入
            rsi_sell = base_rsi_sell + 5   # 80: 不轻易因RSI卖出
            ma_gap = base_ma_gap * 0.5     # 间距要求降低，容易确认金叉
        elif regime == 'trend_down':
            # 下降趋势：极度保守，几乎不买入
            rsi_buy = base_rsi_buy - 20    # 35: 只有极度超卖才考虑
            rsi_sell = base_rsi_sell - 10  # 65: RSI稍高就卖
            ma_gap = base_ma_gap * 3       # 需要非常强的金叉才进场
        else:
            # 震荡市：适度保守
            rsi_buy = base_rsi_buy - 5     # 50
            rsi_sell = base_rsi_sell       # 75
            ma_gap = base_ma_gap * 1.5

        # --- 信号生成 ---
        ma_gap_actual = (ma_short - ma_long) / ma_long if ma_long != 0 else 0

        if prev_signal == 0:  # 空仓
            if ma_gap_actual > ma_gap and rsi < rsi_buy:
                df.iloc[i, df.columns.get_loc('signal')] = 1
        else:  # 持仓
            # 卖出条件
            should_sell = False
            if ma_short < ma_long:  # 死叉
                should_sell = True
            if rsi > rsi_sell:  # RSI超买
                should_sell = True
            # 趋势向上时，给更多容忍度（只有明确死叉才卖）
            if regime == 'trend_up' and not (ma_short < ma_long * 0.995):
                should_sell = False
            if rsi > 85:  # 极度超买无论如何都卖
                should_sell = True

            df.iloc[i, df.columns.get_loc('signal')] = 0 if should_sell else 1

    df['position'] = df['signal'].diff()
    return df


# ============================================================
# 4. 回测引擎（增强版：支持止损）
# ============================================================

def backtest(df, initial_capital=100000.0, stop_loss=None, trailing_stop=None):
    """
    回测引擎（支持止损）

    参数:
        stop_loss: 固定止损比例，如 0.05 表示亏损5%平仓
        trailing_stop: 移动止损比例，如 0.08 表示从最高点回撤8%平仓
    """
    df = df.copy()
    capital = initial_capital
    shares = 0
    buy_price = 0
    max_price_since_buy = 0  # 买入后的最高价（用于移动止损）
    portfolio_values = []
    stop_loss_triggered = []  # 记录止损触发

    for i, row in df.iterrows():
        price = row['close']
        pos = row['position']

        # --- 止损检查（持仓中） ---
        if shares > 0:
            # 更新持仓期间最高价
            max_price_since_buy = max(max_price_since_buy, price)

            triggered = False
            # 固定止损
            if stop_loss and (price - buy_price) / buy_price <= -stop_loss:
                triggered = True
            # 移动止损（从最高点回撤超过阈值）
            if trailing_stop and (price - max_price_since_buy) / max_price_since_buy <= -trailing_stop:
                triggered = True

            if triggered:
                capital += shares * price
                shares = 0
                stop_loss_triggered.append(i)
                # 覆盖position标记
                df.at[i, 'position'] = -1

        # --- 正常信号处理 ---
        if pos == 1 and shares == 0:  # 买入信号
            shares = capital // price
            capital -= shares * price
            buy_price = price
            max_price_since_buy = price
        elif pos == -1 and shares > 0:  # 卖出信号
            capital += shares * price
            shares = 0

        total_value = capital + shares * price
        portfolio_values.append(total_value)

    df['portfolio_value'] = portfolio_values
    df['returns'] = df['portfolio_value'].pct_change()
    df['cumulative_returns'] = (1 + df['returns'].fillna(0)).cumprod() - 1
    df['buy_hold_returns'] = df['close'] / df['close'].iloc[0] - 1

    if stop_loss_triggered:
        print(f"  ⚡ 止损触发 {len(stop_loss_triggered)} 次")

    return df


# ============================================================
# 5. 策略评估指标
# ============================================================

def evaluate_strategy(df, initial_capital=100000.0):
    """计算策略的关键绩效指标"""
    total_return = df['portfolio_value'].iloc[-1] / initial_capital - 1
    buy_hold_return = df['buy_hold_returns'].iloc[-1]

    # 年化收益率（假设250个交易日）
    n_days = len(df)
    annual_return = (1 + total_return) ** (250 / n_days) - 1

    # 最大回撤
    peak = df['portfolio_value'].cummax()
    drawdown = (df['portfolio_value'] - peak) / peak
    max_drawdown = drawdown.min()

    # 夏普比率（无风险利率假设3%）
    daily_rf = 0.03 / 250
    excess_returns = df['returns'].dropna() - daily_rf
    sharpe_ratio = np.sqrt(250) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0

    # 交易次数
    buy_signals = (df['position'] == 1).sum()
    sell_signals = (df['position'] == -1).sum()

    metrics = {
        '策略总收益率': f'{total_return:.2%}',
        '买入持有收益率': f'{buy_hold_return:.2%}',
        '年化收益率': f'{annual_return:.2%}',
        '最大回撤': f'{max_drawdown:.2%}',
        '夏普比率': f'{sharpe_ratio:.2f}',
        '买入次数': buy_signals,
        '卖出次数': sell_signals,
    }
    return metrics


# ============================================================
# 6. 可视化
# ============================================================

def plot_results(df, save_path='backtest_result.png'):
    """绘制回测结果图表"""
    # 尝试使用中文字体，若不可用则使用英文
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 图1：价格 + 均线 + 买卖信号
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Close Price', color='black', linewidth=0.8)
    ax1.plot(df.index, df['MA_short'], label='MA5 (Short)', color='blue', linewidth=0.8)
    ax1.plot(df.index, df['MA_long'], label='MA20 (Long)', color='orange', linewidth=0.8)

    # 标记买卖点
    buy_points = df[df['position'] == 1]
    sell_points = df[df['position'] == -1]
    ax1.scatter(buy_points.index, buy_points['close'], marker='^', color='green', s=80, label='Buy', zorder=5)
    ax1.scatter(sell_points.index, sell_points['close'], marker='v', color='red', s=80, label='Sell', zorder=5)
    ax1.set_title('Dual Moving Average Crossover Strategy - Price & Signals')
    ax1.legend(loc='upper left')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)

    # 图2：累计收益对比
    ax2 = axes[1]
    ax2.plot(df.index, df['cumulative_returns'] * 100, label='Strategy Returns', color='green', linewidth=1.2)
    ax2.plot(df.index, df['buy_hold_returns'] * 100, label='Buy & Hold Returns', color='gray', linewidth=1.2, linestyle='--')
    ax2.set_title('Cumulative Returns Comparison')
    ax2.legend(loc='upper left')
    ax2.set_ylabel('Returns (%)')
    ax2.grid(True, alpha=0.3)

    # 图3：回撤
    ax3 = axes[2]
    peak = df['portfolio_value'].cummax()
    drawdown = (df['portfolio_value'] - peak) / peak * 100
    ax3.fill_between(df.index, drawdown, 0, color='red', alpha=0.3)
    ax3.set_title('Strategy Drawdown')
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 回测图表已保存至: {save_path}")


# ============================================================
# 7. 策略诊断分析
# ============================================================

def diagnose_strategy(df, initial_capital=100000.0):
    """分析策略亏损原因，输出每笔交易的盈亏明细"""
    trades = []
    buy_price = None
    buy_date = None

    for i, row in df.iterrows():
        if row['position'] == 1:
            buy_price = row['close']
            buy_date = i
        elif row['position'] == -1 and buy_price is not None:
            sell_price = row['close']
            pnl_pct = (sell_price - buy_price) / buy_price * 100
            trades.append({
                'buy_date': buy_date.date(),
                'sell_date': i.date(),
                'buy_price': round(buy_price, 2),
                'sell_price': round(sell_price, 2),
                'pnl_pct': round(pnl_pct, 2),
                'hold_days': (i - buy_date).days
            })
            buy_price = None

    print("\n" + "=" * 60)
    print("  📋 策略诊断报告")
    print("=" * 60)

    # 逐笔交易明细
    print("\n【逐笔交易明细】")
    print(f"  {'序号':<4} {'买入日期':<12} {'卖出日期':<12} {'买入价':<9} {'卖出价':<9} {'盈亏%':<8} {'持仓天数'}")
    print("  " + "-" * 72)
    win_count = 0
    total_pnl = 0
    for idx, t in enumerate(trades, 1):
        flag = "✓" if t['pnl_pct'] > 0 else "✗"
        print(f"  {idx:<4} {str(t['buy_date']):<12} {str(t['sell_date']):<12} "
              f"{t['buy_price']:<9} {t['sell_price']:<9} {t['pnl_pct']:>+6.2f}%  {t['hold_days']:<4} {flag}")
        if t['pnl_pct'] > 0:
            win_count += 1
        total_pnl += t['pnl_pct']

    # 统计
    n_trades = len(trades)
    win_rate = win_count / n_trades * 100 if n_trades > 0 else 0
    avg_pnl = total_pnl / n_trades if n_trades > 0 else 0
    wins = [t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]
    losses = [t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    avg_hold = np.mean([t['hold_days'] for t in trades]) if trades else 0

    print(f"\n【诊断统计】")
    print(f"  总交易次数: {n_trades}")
    print(f"  胜率: {win_rate:.1f}% ({win_count}胜 / {n_trades - win_count}负)")
    print(f"  平均每笔盈亏: {avg_pnl:+.2f}%")
    print(f"  平均盈利(赢): {avg_win:+.2f}%")
    print(f"  平均亏损(输): {avg_loss:+.2f}%")
    print(f"  盈亏比: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  盈亏比: N/A")
    print(f"  平均持仓天数: {avg_hold:.1f}天")

    # 亏损原因诊断
    print(f"\n【亏损原因分析】")
    if win_rate < 50:
        print(f"  ⚠️ 胜率过低({win_rate:.0f}%): 短均线(MA5)过于灵敏，频繁产生假信号")
    if abs(avg_loss) > avg_win and avg_win > 0:
        print(f"  ⚠️ 亏损>盈利: 平均亏{abs(avg_loss):.1f}% vs 平均赚{avg_win:.1f}%，止损不及时")
    if avg_hold < 10:
        print(f"  ⚠️ 持仓过短({avg_hold:.0f}天): 频繁进出，受短期震荡影响大")
    if n_trades > 15:
        print(f"  ⚠️ 交易过于频繁({n_trades}次): 在震荡市中反复被割")

    # 市场环境判断
    total_change = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    if total_change < -10:
        print(f"  📉 市场环境: 整体下跌{total_change:.1f}%，趋势跟踪策略天然不利")
    elif abs(total_change) < 5:
        print(f"  📊 市场环境: 震荡市(变化{total_change:+.1f}%)，均线策略容易来回打脸")

    return trades


# ============================================================
# 8. 参数优化搜索
# ============================================================

def optimize_parameters(df_raw, initial_capital=100000.0):
    """遍历不同均线参数组合，找到最优参数"""
    print("\n" + "=" * 60)
    print("  🔍 参数优化搜索")
    print("=" * 60)

    results = []
    short_range = range(3, 15)     # 短期均线: 3~14
    long_range = range(15, 60, 5)  # 长期均线: 15~55

    for short_w in short_range:
        for long_w in long_range:
            df_test = compute_moving_averages(df_raw.copy(), short_w, long_w)
            df_test = generate_signals(df_test)
            df_test = df_test.dropna()
            if len(df_test) < 30:
                continue
            df_test = backtest(df_test, initial_capital)
            total_ret = df_test['portfolio_value'].iloc[-1] / initial_capital - 1
            peak = df_test['portfolio_value'].cummax()
            max_dd = ((df_test['portfolio_value'] - peak) / peak).min()
            n_trades = (df_test['position'] == 1).sum()
            results.append({
                'short': short_w, 'long': long_w,
                'return': total_ret, 'max_dd': max_dd, 'trades': n_trades
            })

    # 按收益排序
    results.sort(key=lambda x: x['return'], reverse=True)

    print(f"\n  测试了 {len(results)} 种参数组合")
    print(f"\n  {'排名':<4} {'短均线':<7} {'长均线':<7} {'收益率':<10} {'最大回撤':<10} {'交易次数'}")
    print("  " + "-" * 55)
    for i, r in enumerate(results[:10], 1):
        print(f"  {i:<4} MA{r['short']:<5} MA{r['long']:<5} {r['return']:>+7.2%}   {r['max_dd']:>+7.2%}   {r['trades']}")

    best = results[0]
    print(f"\n  🏆 最优参数: MA{best['short']} + MA{best['long']}")
    print(f"     收益率: {best['return']:+.2%} | 最大回撤: {best['max_dd']:.2%} | 交易{best['trades']}次")

    return best['short'], best['long']


# ============================================================
# 主程序：三只股票综合对比
# ============================================================

def run_single_stock(symbol, name, initial_capital=100000, short_w=5, long_w=20,
                     trailing_stop=0.08):
    """对单只股票运行所有策略并返回结果"""
    # 获取数据
    try:
        df_raw = get_real_stock_data(symbol=symbol, start='2024-01-01', end='2025-05-01', source='akshare')
    except:
        print(f"  ⚠️ {name}数据获取失败，跳过")
        return None

    results = {}

    # 策略A: 原始均线
    df_a = compute_moving_averages(df_raw.copy(), short_w, long_w)
    df_a = generate_signals(df_a)
    df_a = df_a.dropna()
    df_a = backtest(df_a, initial_capital)
    results['A:原始均线'] = evaluate_strategy(df_a, initial_capital)

    # 策略D: 增强版（RSI过滤+止损）
    df_d = compute_moving_averages(df_raw.copy(), short_w, long_w)
    df_d = compute_rsi(df_d, period=14)
    df_d = generate_signals_enhanced(df_d, rsi_buy_threshold=45, rsi_sell_threshold=70, ma_gap_threshold=0.005)
    df_d = df_d.dropna()
    df_d = backtest(df_d, initial_capital, trailing_stop=trailing_stop)
    results['D:保守增强'] = evaluate_strategy(df_d, initial_capital)

    # 策略E: 自适应策略（新）
    df_e = compute_moving_averages(df_raw.copy(), short_w, long_w)
    df_e = compute_rsi(df_e, period=14)
    df_e = compute_market_regime(df_e)
    df_e = generate_signals_adaptive(df_e, base_rsi_buy=55, base_rsi_sell=75, base_ma_gap=0.003)
    df_e = df_e.dropna()
    df_e = backtest(df_e, initial_capital, trailing_stop=trailing_stop)
    results['E:自适应策略'] = evaluate_strategy(df_e, initial_capital)

    # 基准
    results['基准:持有不动'] = {'策略总收益率': results['A:原始均线']['买入持有收益率']}

    return results, df_e  # 返回自适应策略的df用于可视化


if __name__ == '__main__':
    print("=" * 70)
    print("  Lesson 1: 自适应双均线策略 — 三只股票综合验证")
    print("=" * 70)

    # ========== 测试标的 ==========
    stocks = [
        ('600519', '贵州茅台(震荡下跌)'),
        ('601872', '招商轮船(先涨后跌)'),
        ('600938', '中国海油(强势上涨)'),
    ]

    # 策略参数
    SHORT_WINDOW = 5
    LONG_WINDOW = 20
    INITIAL_CAPITAL = 100000
    TRAILING_STOP = 0.08

    all_results = {}

    for symbol, name in stocks:
        print(f"\n{'─' * 70}")
        print(f"  📈 {name} ({symbol})")
        print(f"{'─' * 70}")

        result = run_single_stock(symbol, name, INITIAL_CAPITAL, SHORT_WINDOW, LONG_WINDOW, TRAILING_STOP)
        if result is None:
            continue

        results, df_e = result
        all_results[name] = results

        # 输出单股结果
        print(f"\n  {'策略':<14} {'收益率':<10} {'最大回撤':<10} {'夏普':<8} {'交易次数'}")
        print(f"  {'─' * 55}")
        for strat, m in results.items():
            if strat == '基准:持有不动':
                print(f"  {strat:<14} {m['策略总收益率']:<10}")
            else:
                print(f"  {strat:<14} {m['策略总收益率']:<10} {m['最大回撤']:<10} {m['夏普比率']:<8} {m['买入次数']}")

        # 保存自适应策略图表
        plot_results(df_e, save_path=f'/root/AI_Economic/backtest_{symbol}_adaptive.png')

    # ==========================================
    # 综合对比报告
    # ==========================================
    print("\n\n" + "=" * 70)
    print("  📊 综合对比报告：自适应策略 vs 其他策略")
    print("=" * 70)

    # 计算各策略在三只股票上的平均表现
    strategy_names = ['A:原始均线', 'D:保守增强', 'E:自适应策略']
    print(f"\n  {'股票':<18}", end='')
    for s in strategy_names:
        print(f" {s:<14}", end='')
    print(f" {'买入持有':<10}")
    print(f"  {'─' * 70}")

    avg_returns = {s: [] for s in strategy_names}
    avg_dd = {s: [] for s in strategy_names}

    for name, results in all_results.items():
        print(f"  {name:<18}", end='')
        for s in strategy_names:
            ret = results[s]['策略总收益率']
            print(f" {ret:<14}", end='')
            avg_returns[s].append(float(ret.strip('%')) / 100)
            avg_dd[s].append(float(results[s]['最大回撤'].strip('%')) / 100)
        print(f" {results['基准:持有不动']['策略总收益率']:<10}")

    # 平均值
    print(f"  {'─' * 70}")
    print(f"  {'【平均收益率】':<18}", end='')
    for s in strategy_names:
        avg = sum(avg_returns[s]) / len(avg_returns[s]) * 100
        print(f" {avg:>+6.2f}%{'':7}", end='')
    print()
    print(f"  {'【平均最大回撤】':<18}", end='')
    for s in strategy_names:
        avg = sum(avg_dd[s]) / len(avg_dd[s]) * 100
        print(f" {avg:>6.2f}%{'':7}", end='')
    print()

    # 找最优
    best_strat = max(strategy_names, key=lambda s: sum(avg_returns[s]) / len(avg_returns[s]))
    best_dd_strat = max(strategy_names, key=lambda s: sum(avg_dd[s]) / len(avg_dd[s]))  # 回撤最小的

    print(f"\n  🏆 综合最优策略: {best_strat}")
    print(f"     平均收益: {sum(avg_returns[best_strat])/len(avg_returns[best_strat])*100:+.2f}%")
    print(f"     平均回撤: {sum(avg_dd[best_strat])/len(avg_dd[best_strat])*100:.2f}%")

    print("\n" + "=" * 70)
    print("  💡 自适应策略核心逻辑")
    print("=" * 70)
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  市场状态判断（均线斜率 + 趋势强度）                      │
  ├──────────┬──────────────────────────────────────────────┤
  │ 上升趋势 │ 放宽买入(RSI<65) + 紧跟趋势(不轻易卖出)      │
  │ 震荡市   │ 适度保守(RSI<50) + 标准卖出                   │
  │ 下降趋势 │ 极度保守(RSI<35) + 快速止损                   │
  ├──────────┴──────────────────────────────────────────────┤
  │ + 移动止损8%: 从持仓最高点回撤超8%强制平仓               │
  │ + 极度超买保护: RSI>85 无条件卖出                         │
  └─────────────────────────────────────────────────────────┘
    """)
    print("✅ 分析完成！")
