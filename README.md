# FinRL - Dynamic Ticker Universe Trading Bot

A sophisticated reinforcement learning trading system that adapts to changing market conditions using a dynamic ticker universe and multi-timeframe technical indicators.

## Features

### Dynamic Ticker Selection
- **Adaptive Universe**: Only trades tickers with complete 120-day price history at each time period
- **Data Quality Focus**: Eliminates look-forward bias by using real-time data availability
- **31 ETF Universe**: Covers global equities, bonds, commodities, and sector-specific investments

### Multi-Timeframe Technical Indicators
- **Multiple Returns**: 20-day, 60-day, and 120-day trailing returns
- **120-day Moving Average**: Long-term trend identification
- **120-day Volatility**: Extended risk assessment
- **Portfolio Metrics**: Cash ratio, portfolio change, available ticker count

### Training Configuration
- **Training Period**: 2010-2015 (6 years of market data)
- **Testing Period**: 2016-2025 (9+ years out-of-sample)
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Optimization**: M4 Max GPU acceleration with MPS

## Ticker Universe

```python
TICKER_LIST = ["CMF", "VCIT", "TLT", "EMB", "LEMB", "HYG", "SPY", "IWM", "QQQ", "DXJ", 
               "EWJV", "IEV", "ASEA", "EMXC", "ILF", "GMF", "CQQQ", "ASHR", "INDA", "SMIN", 
               "EWW", "TUR", "EPOL", "VNM", "MCHI", "VDE", "DBA", "DBB", "VNQ", "COPX", "GDX"]
```

## Installation

```bash
conda create -n finrl python=3.9 -y
conda activate finrl
pip install finrl[full]
pip install stable-baselines3[extra]
pip install yfinance matplotlib pandas
```

## Usage

```bash
python dynamic_rl_bot.py
```

## Output Files

- **`rl_trading_performance.pdf`**: Performance visualization
- **`rl_trading_results.xlsx`**: Detailed trading results
- **`rl_portfolio_weights.xlsx`**: Daily portfolio allocations
- **`ticker_availability.xlsx`**: Dynamic ticker universe tracking

## Key Architecture

### Trading Environment
- **Daily Rebalancing**: Makes trading decisions every market day
- **Continuous Actions**: Portfolio weights from -1 (full sell) to +1 (full buy)
- **Transaction Costs**: 0.1% per trade
- **Dynamic Observation Space**: 189 dimensions (31 tickers × 6 features + 3 portfolio metrics)

### Reward Structure
- **Pure Returns**: Maximizes daily portfolio returns
- **No Risk Adjustment**: Focuses on absolute performance
- **Real-time Execution**: Actions at time t, rewards at t+1

## Performance Metrics
- Total returns vs equal-weighted benchmark
- Annualized volatility and Sharpe ratios
- Maximum drawdown analysis
- Dynamic ticker utilization statistics

## Technical Implementation
- **No Look-Forward Bias**: Strict temporal data separation
- **Missing Data Handling**: Limited forward-fill (max 5 days)
- **Memory Optimization**: Efficient data structures for 15+ years of data
- **GPU Acceleration**: MPS backend for M4 Max optimization
