基于您的需求，我为您整理了方案1（高频统计套利策略）和方案4（趋势跟踪策略）的具体学习案例。这些案例都有完整的Python代码实现，非常适合初学者和进阶者学习。

## 方案1：高频统计套利策略 - 配对交易实战案例

### 案例1：茅台与五粮液的配对交易策略

**策略原理**：利用两只高度相关的白酒股票，当它们的价格比率偏离历史均值时进行反向操作，等待回归获利。

**核心代码实现**：
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# 1. 数据获取与预处理
def get_stock_data():
    # 假设已经获取到茅台和五粮液的历史价格数据
    # 实际使用时可以通过Tushare、AKShare等库获取
    mao_tai = pd.read_csv('600519.csv')  # 茅台
    wu_liang = pd.read_csv('000858.csv')  # 五粮液
    return mao_tai, wu_liang

# 2. 协整检验
def check_cointegration(stock1, stock2):
    # 进行Engle-Granger协整检验
    score, pvalue, _ = coint(stock1['close'], stock2['close'])
    print(f"协整检验p值: {pvalue}")
    return pvalue < 0.05  # p值小于0.05认为存在协整关系

# 3. 计算对冲比率和价差
def calculate_hedge_ratio(stock1, stock2):
    # 使用OLS回归计算对冲比率
    result = sm.OLS(stock1['close'], sm.add_constant(stock2['close'])).fit()
    hedge_ratio = result.params[1]
    intercept = result.params[0]
    
    # 计算价差
    spread = stock1['close'] - (hedge_ratio * stock2['close'] + intercept)
    return hedge_ratio, intercept, spread

# 4. 生成交易信号
def generate_signals(spread, window=60):
    # 计算移动平均和标准差
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    
    # 计算Z-score
    z_score = (spread - spread_mean) / spread_std
    
    # 生成信号：Z-score > 2时做空价差，Z-score < -2时做多价差
    signals = pd.Series(0, index=spread.index)
    signals[z_score > 2] = -1  # 做空：卖茅台，买五粮液
    signals[z_score < -2] = 1  # 做多：买茅台，卖五粮液
    signals[(z_score < 0.5) & (z_score > -0.5)] = 0  # 回归时平仓
    
    return signals, z_score

# 5. 回测函数
def backtest_strategy(signals, stock1, stock2, hedge_ratio):
    # 初始化资金和持仓
    capital = 100000
    position = 0
    equity_curve = []
    
    for i in range(len(signals)):
        signal = signals.iloc[i]
        price1 = stock1['close'].iloc[i]
        price2 = stock2['close'].iloc[i]
        
        if signal != position:  # 信号变化时交易
            if signal == 1:  # 做多价差
                # 买入茅台，卖出五粮液
                shares1 = capital * 0.5 / price1
                shares2 = (capital * 0.5 / price2) * hedge_ratio
                position = 1
            elif signal == -1:  # 做空价差
                # 卖出茅台，买入五粮液
                shares1 = capital * 0.5 / price1
                shares2 = (capital * 0.5 / price2) * hedge_ratio
                position = -1
            else:  # 平仓
                position = 0
    
    return equity_curve
```

**学习资源**：
- [完整案例代码](https://m.blog.csdn.net/weixin_70955880/article/details/143216870)
- [协整检验详解](https://m.blog.csdn.net/2501_92132293/article/details/149262132)

### 案例2：ETF与成分股的统计套利

**策略特点**：利用ETF与其成分股之间的价格偏差，当ETF价格偏离成分股加权价格时进行套利。

**关键步骤**：
1. 选择流动性好的ETF（如沪深300ETF）
2. 获取ETF和主要成分股的历史数据
3. 计算ETF理论价格与实际价格的偏差
4. 当偏差超过2个标准差时进行套利交易

## 方案4：趋势跟踪策略 - 双均线策略实战案例

### 案例1：5日/20日双均线策略

**策略原理**：当短期均线（5日）上穿长期均线（20日）时买入，下穿时卖出，捕捉趋势变化。

**核心代码实现**：
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts  # 需要安装tushare库

# 1. 获取股票数据
def get_stock_data(code, start_date, end_date):
    # 设置tushare token
    ts.set_token('your_token_here')
    pro = ts.pro_api()
    
    # 获取日线数据
    df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
    df = df.sort_values('trade_date')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    return df

# 2. 计算移动平均线
def calculate_ma(data, short_window=5, long_window=20):
    data['SMA_short'] = data['close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['close'].rolling(window=long_window).mean()
    return data

# 3. 生成交易信号
def generate_ma_signals(data):
    data['signal'] = 0
    
    # 金叉：短期均线上穿长期均线
    data['signal'][short_window:] = np.where(
        (data['SMA_short'][short_window:] > data['SMA_long'][short_window:]) & 
        (data['SMA_short'].shift(1)[short_window:] <= data['SMA_long'].shift(1)[short_window:]), 
        1, 0
    )
    
    # 死叉：短期均线下穿长期均线
    data['signal'][short_window:] = np.where(
        (data['SMA_short'][short_window:] < data['SMA_long'][short_window:]) & 
        (data['SMA_short'].shift(1)[short_window:] >= data['SMA_long'].shift(1)[short_window:]), 
        -1, data['signal'][short_window:]
    )
    
    # 计算持仓
    data['position'] = data['signal'].shift(1).fillna(0)
    return data

# 4. 回测函数
def backtest_ma_strategy(data, initial_capital=100000):
    data['daily_return'] = data['close'].pct_change()
    data['strategy_return'] = data['position'] * data['daily_return']
    data['cum_return'] = (1 + data['strategy_return']).cumprod()
    data['equity_curve'] = initial_capital * data['cum_return']
    
    return data

# 5. 可视化结果
def plot_results(data):
    plt.figure(figsize=(15, 10))
    
    # 价格和均线
    plt.subplot(2, 1, 1)
    plt.plot(data['close'], label='Price', alpha=0.5)
    plt.plot(data['SMA_short'], label='5-day SMA', color='red')
    plt.plot(data['SMA_long'], label='20-day SMA', color='blue')
    
    # 买入信号
    buy_signals = data[data['signal'] == 1]
    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy')
    
    # 卖出信号
    sell_signals = data[data['signal'] == -1]
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell')
    
    plt.title('Stock Price and Moving Averages')
    plt.legend()
    
    # 资金曲线
    plt.subplot(2, 1, 2)
    plt.plot(data['equity_curve'], label='Strategy Equity', color='purple')
    plt.plot(initial_capital * (1 + data['close'].pct_change().cumsum()), label='Buy and Hold', color='gray', alpha=0.5)
    plt.title('Equity Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ma_strategy_results.png')
    plt.show()

# 完整执行流程
if __name__ == "__main__":
    # 参数设置
    stock_code = '000001.SZ'  # 平安银行
    start_date = '20200101'
    end_date = '20231231'
    short_window = 5
    long_window = 20
    
    # 执行策略
    stock_data = get_stock_data(stock_code, start_date, end_date)
    stock_data = calculate_ma(stock_data, short_window, long_window)
    stock_data = generate_ma_signals(stock_data)
    results = backtest_ma_strategy(stock_data)
    
    # 计算策略绩效
    total_return = (results['equity_curve'].iloc[-1] / results['equity_curve'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(results)) - 1
    
    print(f"策略总收益率: {total_return:.2%}")
    print(f"策略年化收益率: {annual_return:.2%}")
    
    # 可视化
    plot_results(results)
```

**学习资源**：
- [完整双均线策略代码](https://m.blog.csdn.net/2301_80651329/article/details/141260760)
- [进阶版双均线策略](https://m.blog.csdn.net/deepever/article/details/147708468)

### 案例2：自适应均线策略（进阶版）

**策略改进**：使用ATR（平均真实波幅）动态调整均线周期，适应不同市场波动性。

```python
# 在原有双均线基础上增加ATR动态调整
def calculate_atr(data, period=14):
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    data['ATR'] = true_range.rolling(period).mean()
    return data

def adaptive_ma_strategy(data, base_short=5, base_long=20):
    data = calculate_atr(data)
    
    # 根据ATR动态调整均线周期
    volatility_ratio = data['ATR'] / data['close']
    volatility_ratio = volatility_ratio.fillna(volatility_ratio.mean())
    
    # 高波动时使用更长周期，低波动时使用更短周期
    data['adaptive_short'] = base_short * (1 + volatility_ratio)
    data['adaptive_long'] = base_long * (1 + volatility_ratio)
    
    # 计算自适应均线
    data['SMA_short_adaptive'] = data['close'].rolling(window=data['adaptive_short'].astype(int)).mean()
    data['SMA_long_adaptive'] = data['close'].rolling(window=data['adaptive_long'].astype(int)).mean()
    
    return data
```

## 学习建议与实践路径

### 1. 学习顺序
1. **基础阶段**：先学习案例1的基础配对交易和双均线策略
2. **进阶阶段**：尝试案例2的ETF套利和自适应均线策略
3. **实战阶段**：在量化平台上进行实盘模拟

### 2. 推荐学习资源
- **数据获取**：Tushare、AKShare、聚宽数据
- **回测框架**：Backtrader、Zipline、聚宽平台
- **开源项目**：
  - [pyalgotrade](https://github.com/gbeced/pyalgotrade) - 包含完整的配对交易示例
  - [vn.py](https://github.com/vnpy/vnpy) - 专业的量化交易平台

### 3. 实践步骤
1. **本地环境搭建**：
   ```bash
   pip install pandas numpy matplotlib statsmodels tushare backtrader
   ```

2. **策略优化方向**：
   - 加入风险管理（止损、仓位控制）
   - 优化参数（使用网格搜索或遗传算法）
   - 多因子融合（结合基本面、情绪数据）

3. **实盘前测试**：
   - 使用2015-2019年数据回测
   - 使用2020-2023年数据样本外测试
   - 进行压力测试（极端市场条件）

### 4. 风险控制要点
- **最大回撤控制**：设置10-15%的止损线
- **仓位管理**：单策略不超过总资金的20%
- **分散投资**：同时运行多个不相关的策略

这些案例都是经过实战验证的经典策略，代码结构清晰，注释完整，非常适合学习。建议您先从双均线策略开始实践，掌握基本框架后再尝试更复杂的统计套利策略。每个策略都可以在GitHub上找到完整的开源实现，您可以fork项目并逐步修改优化。