#!/usr/bin/env python3
"""
INPUT FILES:
- None (downloads data from Yahoo Finance)

OUTPUT FILES:
- rl_trading_performance_single.pdf: Performance comparison chart (PPO single-env vs Dynamic Equal-Weight)
- rl_trading_results_single.xlsx: Daily portfolio values and benchmark
- rl_portfolio_weights_single.xlsx: Ticker weights for each date (variable columns)
- ticker_availability_single.xlsx: Which tickers were available each date
- rl_metrics_report_single.xlsx: Summary metrics (CAGR, ann. volatility, Sharpe, max drawdown) for PPO, Equal-Weight, Net (relative)

DESCRIPTION:
Single-environment baseline of the dynamic ticker universe RL trading bot. This replicates the
configuration that produced the strong results: single DummyVecEnv, 100k timesteps, n_steps=2048,
batch_size=128, and other PPO defaults as in our prior run. Evaluation and outputs are unchanged in
logic, but file names are suffixed with _single to avoid conflicts with other runs.

VERSION: 1.0 (Single-env PPO baseline)
DATE: 2025-09-03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
import warnings
warnings.filterwarnings('ignore')

# Configuration - User's specific ticker universe
TICKER_LIST = [
    "CMF", "VCIT", "TLT", "EMB", "LEMB", "HYG", "SPY", "IWM", "QQQ", "DXJ",
    "EWJV", "IEV", "ASEA", "EMXC", "ILF", "GMF", "CQQQ", "ASHR", "INDA", "SMIN",
    "EWW", "TUR", "EPOL", "VNM", "MCHI", "VDE", "DBA", "DBB", "VNQ", "COPX", "GDX"
]
START_DATE = "2010-01-01"
END_DATE = "2025-09-02"  # Today's date
INITIAL_CAPITAL = 100_000
TRAIN_END_DATE = "2016-01-01"  # Train on 2010-2015, test on 2016-2025
MIN_HISTORY_DAYS = 120  # Minimum days of data required for 120-day indicators

class DynamicTradingEnv(gym.Env):
    """
    Trading Environment with Dynamic Ticker Universe
    Action space adapts to available tickers at each time step
    """
    def __init__(self, close_df, high_df, low_df, ticker_availability, initial_amount=100000, transaction_cost_pct=0.001):
        super(DynamicTradingEnv, self).__init__()
        
        # Store price data
        self.close_df = close_df.reset_index(drop=True)
        self.high_df = high_df.reset_index(drop=True)
        self.low_df = low_df.reset_index(drop=True)
        self.ticker_availability = ticker_availability
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.max_tickers = len(TICKER_LIST)  # Maximum possible tickers
        
        # Fixed action space for maximum possible tickers (pad with zeros for unavailable)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.max_tickers,), dtype=np.float32)
        
        # Fixed observation space for maximum possible tickers
        # Per-ticker features (15): normalized_price, returns_20/60/120, ma120_ratio, vol120_norm,
        # ATR14/price, RSI14, Bollinger %B(20), BandWidth(20), %B(60), BandWidth(60),
        # xsec_mom_rank_120, beta_120_to_SPY, idio_vol_120
        # Global features (5): cash_ratio, portfolio_change, num_available, breadth_above_ma120, dispersion_cs_1d
        obs_dim = self.max_tickers * 15 + 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = MIN_HISTORY_DAYS
        self.portfolio_value = self.initial_amount
        self.cash = self.initial_amount
        self.holdings = np.zeros(self.max_tickers)  # Holdings for all possible tickers
        self.portfolio_values = []
        return self._get_observation(), {}

    def _get_available_tickers(self):
        """Get list of tickers with complete data at current step"""
        if self.current_step >= len(self.ticker_availability):
            return []
        return self.ticker_availability.iloc[self.current_step]['available_tickers']

    def _get_observation(self):
        if self.current_step >= len(self.close_df):
            self.current_step = len(self.close_df) - 1
        
        available_tickers = self._get_available_tickers()
        
        # Initialize arrays for all possible tickers
        normalized_prices = np.zeros(self.max_tickers)
        returns_20d = np.zeros(self.max_tickers)
        returns_60d = np.zeros(self.max_tickers)
        returns_120d = np.zeros(self.max_tickers)
        moving_averages = np.zeros(self.max_tickers)
        volatilities = np.zeros(self.max_tickers)
        atr14_over_price = np.zeros(self.max_tickers)
        rsi14 = np.zeros(self.max_tickers)
        boll_pctb_20 = np.zeros(self.max_tickers)
        boll_bw_20 = np.zeros(self.max_tickers)
        boll_pctb_60 = np.zeros(self.max_tickers)
        boll_bw_60 = np.zeros(self.max_tickers)
        xsec_rank_120 = np.zeros(self.max_tickers)
        beta_120 = np.zeros(self.max_tickers)
        idio_vol_120 = np.zeros(self.max_tickers)
        
        # First pass: compute per-ticker basic features
        for ticker in available_tickers:
            if ticker in TICKER_LIST:
                i = TICKER_LIST.index(ticker)
                current_price = self.close_df.iloc[self.current_step][ticker]
                
                # Multiple trailing returns
                if self.current_step >= 20:
                    ret_20d = (current_price / self.close_df.iloc[self.current_step-20][ticker] - 1)
                else:
                    ret_20d = 0
                returns_20d[i] = ret_20d
                
                if self.current_step >= 60:
                    ret_60d = (current_price / self.close_df.iloc[self.current_step-60][ticker] - 1)
                else:
                    ret_60d = 0
                returns_60d[i] = ret_60d
                
                if self.current_step >= 120:
                    ret_120d = (current_price / self.close_df.iloc[self.current_step-120][ticker] - 1)
                else:
                    ret_120d = 0
                returns_120d[i] = ret_120d
                
                # 120-day moving average ratio (exclude current from MA)
                if self.current_step >= 120:
                    ma = self.close_df.iloc[self.current_step-120:self.current_step][ticker].mean()
                    ma_ratio = current_price / ma - 1 if ma > 0 else 0
                else:
                    ma_ratio = 0
                moving_averages[i] = ma_ratio
                
                # 120-day volatility of price normalized by current price (exclude current)
                if self.current_step >= 120:
                    vol = self.close_df.iloc[self.current_step-120:self.current_step][ticker].std()
                    vol_norm = vol / current_price if current_price > 0 else 0
                else:
                    vol_norm = 0
                volatilities[i] = vol_norm
                
                normalized_prices[i] = current_price
        
        # Cross-sectional momentum rank (120D) across available tickers
        if len(available_tickers) > 1:
            vals = {t: returns_120d[TICKER_LIST.index(t)] for t in available_tickers}
            s = pd.Series(vals)
            ranks = s.rank(pct=True, method='average')
            for t in available_tickers:
                xsec_rank_120[TICKER_LIST.index(t)] = ranks[t]
        
        # Second pass: compute ATR14/price, RSI14, Bollinger (20/60), beta/idiosyncratic vol(120)
        for ticker in available_tickers:
            if ticker in TICKER_LIST:
                i = TICKER_LIST.index(ticker)
                current_price = self.close_df.iloc[self.current_step][ticker]
                
                # ATR(14)/price
                if self.current_step >= 1 and self.current_step >= 13:
                    tr_values = []
                    start_k = self.current_step - 13
                    for k in range(start_k, self.current_step + 1):
                        high_k = self.high_df.iloc[k][ticker]
                        low_k = self.low_df.iloc[k][ticker]
                        prev_close = self.close_df.iloc[k-1][ticker] if k - 1 >= 0 else self.close_df.iloc[k][ticker]
                        tr = max(high_k - low_k, abs(high_k - prev_close), abs(low_k - prev_close))
                        tr_values.append(tr)
                    atr14 = np.mean(tr_values) if len(tr_values) == 14 else 0.0
                    atr14_over_price[i] = (atr14 / current_price) if current_price > 0 else 0.0
                else:
                    atr14_over_price[i] = 0.0
                
                # RSI(14) scaled to 0-1
                if self.current_step >= 14:
                    window = self.close_df[ticker].iloc[self.current_step-14:self.current_step+1].to_numpy()
                    diffs = np.diff(window)
                    gains = np.where(diffs > 0, diffs, 0.0)
                    losses = np.where(diffs < 0, -diffs, 0.0)
                    avg_gain = gains.mean() if gains.size > 0 else 0.0
                    avg_loss = losses.mean() if losses.size > 0 else 0.0
                    if avg_loss == 0:
                        rsi = 1.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi = 1 - (1 / (1 + rs))  # 0-1
                    rsi14[i] = rsi
                else:
                    rsi14[i] = 0.0
                
                # Bollinger Bands %B and BandWidth (20)
                if self.current_step >= 19:
                    w20 = self.close_df[ticker].iloc[self.current_step-19:self.current_step+1]
                    ma20 = w20.mean()
                    sd20 = w20.std()
                    upper20 = ma20 + 2 * sd20
                    lower20 = ma20 - 2 * sd20
                    denom20 = upper20 - lower20
                    boll_pctb_20[i] = (current_price - lower20) / denom20 if denom20 > 0 else 0.0
                    boll_bw_20[i] = (denom20 / ma20) if ma20 > 0 else 0.0
                else:
                    boll_pctb_20[i] = 0.0
                    boll_bw_20[i] = 0.0
                
                # Bollinger Bands %B and BandWidth (60)
                if self.current_step >= 59:
                    w60 = self.close_df[ticker].iloc[self.current_step-59:self.current_step+1]
                    ma60 = w60.mean()
                    sd60 = w60.std()
                    upper60 = ma60 + 2 * sd60
                    lower60 = ma60 - 2 * sd60
                    denom60 = upper60 - lower60
                    boll_pctb_60[i] = (current_price - lower60) / denom60 if denom60 > 0 else 0.0
                    boll_bw_60[i] = (denom60 / ma60) if ma60 > 0 else 0.0
                else:
                    boll_pctb_60[i] = 0.0
                    boll_bw_60[i] = 0.0
                
                # Rolling beta to SPY (120) and idiosyncratic volatility (residual std)
                if self.current_step >= 120 and 'SPY' in TICKER_LIST:
                    # Build return windows
                    close_i = self.close_df[ticker].iloc[self.current_step-120:self.current_step+1].to_numpy()
                    close_spy = self.close_df['SPY'].iloc[self.current_step-120:self.current_step+1].to_numpy()
                    r_i = close_i[1:] / close_i[:-1] - 1
                    r_spy = close_spy[1:] / close_spy[:-1] - 1
                    if r_spy.var(ddof=1) > 0 and r_i.size == r_spy.size:
                        cov = np.cov(r_i, r_spy, ddof=1)[0, 1]
                        var_spy = np.var(r_spy, ddof=1)
                        b = cov / var_spy if var_spy > 0 else 0.0
                        beta_120[i] = b
                        resid = r_i - b * r_spy
                        idio_vol_120[i] = np.std(resid, ddof=1)
                    else:
                        beta_120[i] = 0.0
                        idio_vol_120[i] = 0.0
                else:
                    # For SPY itself, set beta ~1 when data insufficient; else 0 if SPY missing
                    beta_120[i] = 1.0 if ticker == 'SPY' else 0.0
                    idio_vol_120[i] = 0.0
        
        # Normalize prices across available tickers only
        available_prices = normalized_prices[normalized_prices > 0]
        if len(available_prices) > 0:
            max_price = available_prices.max()
            if max_price > 0:
                normalized_prices = normalized_prices / max_price
        
        # Portfolio state
        current_prices = np.array([
            self.close_df.iloc[self.current_step][ticker] if ticker in available_tickers else 0
            for ticker in TICKER_LIST
        ])
        self.portfolio_value = self.cash + np.sum(self.holdings * current_prices)
        cash_ratio = self.cash / max(self.portfolio_value, 1)
        portfolio_change = (self.portfolio_value / self.initial_amount - 1)
        num_available = len(available_tickers) / self.max_tickers  # Normalized count
        
        # Global market health features
        # Breadth: % of available tickers above MA120 (exclude current from MA)
        breadth = 0.0
        if len(available_tickers) > 0 and self.current_step >= 120:
            above = 0
            count = 0
            for t in available_tickers:
                cp = self.close_df.iloc[self.current_step][t]
                ma = self.close_df.iloc[self.current_step-120:self.current_step][t].mean()
                if ma > 0:
                    above += 1 if cp > ma else 0
                    count += 1
            breadth = (above / count) if count > 0 else 0.0
        
        # Dispersion: cross-sectional std of 1D returns across available tickers
        dispersion = 0.0
        if len(available_tickers) > 1 and self.current_step >= 1:
            rets = []
            for t in available_tickers:
                c0 = self.close_df.iloc[self.current_step-1][t]
                c1 = self.close_df.iloc[self.current_step][t]
                if c0 > 0:
                    rets.append(c1 / c0 - 1)
            if len(rets) > 1:
                dispersion = float(np.std(rets))
        
        # Clean arrays
        returns_20d = np.nan_to_num(returns_20d, nan=0.0, posinf=1.0, neginf=-1.0)
        returns_60d = np.nan_to_num(returns_60d, nan=0.0, posinf=1.0, neginf=-1.0)
        returns_120d = np.nan_to_num(returns_120d, nan=0.0, posinf=1.0, neginf=-1.0)
        moving_averages = np.nan_to_num(moving_averages, nan=0.0, posinf=1.0, neginf=-1.0)
        volatilities = np.nan_to_num(volatilities, nan=0.0, posinf=1.0, neginf=-1.0)
        normalized_prices = np.nan_to_num(normalized_prices, nan=0.0, posinf=1.0, neginf=-1.0)
        atr14_over_price = np.nan_to_num(atr14_over_price, nan=0.0, posinf=1.0, neginf=-1.0)
        rsi14 = np.nan_to_num(rsi14, nan=0.0, posinf=1.0, neginf=-1.0)
        boll_pctb_20 = np.nan_to_num(boll_pctb_20, nan=0.0, posinf=1.0, neginf=-1.0)
        boll_bw_20 = np.nan_to_num(boll_bw_20, nan=0.0, posinf=1.0, neginf=-1.0)
        boll_pctb_60 = np.nan_to_num(boll_pctb_60, nan=0.0, posinf=1.0, neginf=-1.0)
        boll_bw_60 = np.nan_to_num(boll_bw_60, nan=0.0, posinf=1.0, neginf=-1.0)
        xsec_rank_120 = np.nan_to_num(xsec_rank_120, nan=0.0, posinf=1.0, neginf=-1.0)
        beta_120 = np.nan_to_num(beta_120, nan=0.0, posinf=1.0, neginf=-1.0)
        idio_vol_120 = np.nan_to_num(idio_vol_120, nan=0.0, posinf=1.0, neginf=-1.0)
        
        obs = np.concatenate([
            normalized_prices,
            returns_20d,
            returns_60d,
            returns_120d,
            moving_averages,
            volatilities,
            atr14_over_price,
            rsi14,
            boll_pctb_20,
            boll_bw_20,
            boll_pctb_60,
            boll_bw_60,
            xsec_rank_120,
            beta_120,
            idio_vol_120,
            [cash_ratio, portfolio_change, num_available, breadth, dispersion]
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action):
        if self.current_step >= len(self.close_df) - 1:
            return self._get_observation(), 0, True, True, {}
        
        available_tickers = self._get_available_tickers()
        current_prices = np.array([
            self.close_df.iloc[self.current_step][ticker] if ticker in available_tickers else 0 
            for ticker in TICKER_LIST
        ])
        
        # Calculate current portfolio value
        old_portfolio_value = self.cash + np.sum(self.holdings * current_prices)
        
        # Execute trades only for available tickers
        for i, ticker in enumerate(TICKER_LIST):
            if ticker in available_tickers and current_prices[i] > 0:
                act = action[i]
                if act > 0:  # Buy
                    buy_amount = abs(act) * self.cash
                    shares = buy_amount / (current_prices[i] * (1 + self.transaction_cost_pct))
                    cost = shares * current_prices[i] * (1 + self.transaction_cost_pct)
                    if cost <= self.cash:
                        self.holdings[i] += shares
                        self.cash -= cost
                
                elif act < 0 and self.holdings[i] > 0:  # Sell
                    sell_shares = abs(act) * self.holdings[i]
                    proceeds = sell_shares * current_prices[i] * (1 - self.transaction_cost_pct)
                    self.holdings[i] -= sell_shares
                    self.cash += proceeds
        
        # Move to next step
        self.current_step += 1
        
        if self.current_step < len(self.close_df):
            next_available_tickers = self._get_available_tickers()
            new_prices = np.array([
                self.close_df.iloc[self.current_step][ticker] if ticker in next_available_tickers else 0 
                for ticker in TICKER_LIST
            ])
            new_portfolio_value = self.cash + np.sum(self.holdings * new_prices)
        else:
            new_portfolio_value = old_portfolio_value
        
        # Calculate reward
        portfolio_return = (new_portfolio_value - old_portfolio_value) / max(old_portfolio_value, 1)
        reward = portfolio_return
        
        self.portfolio_value = new_portfolio_value
        self.portfolio_values.append(self.portfolio_value)
        
        done = self.current_step >= len(self.close_df) - 1
        
        return self._get_observation(), reward, done, False, {}

def create_ticker_availability(price_df):
    """
    Create a DataFrame tracking which tickers have complete data at each date
    """
    print("Creating dynamic ticker availability matrix...")
    
    availability_data = []
    
    for i, date in enumerate(price_df.index):
        if i < MIN_HISTORY_DAYS:
            # Not enough history for any ticker
            available_tickers = []
        else:
            # Check which tickers have complete data for the required history window
            available_tickers = []
            for ticker in TICKER_LIST:
                if ticker in price_df.columns:
                    # Check if ticker has complete data for last MIN_HISTORY_DAYS
                    history_data = price_df.iloc[i-MIN_HISTORY_DAYS:i+1][ticker]
                    if not history_data.isna().any() and (history_data > 0).all():
                        available_tickers.append(ticker)
        
        availability_data.append({
            'date': date,
            'available_tickers': available_tickers,
            'num_available': len(available_tickers)
        })
    
    availability_df = pd.DataFrame(availability_data)
    print(f"Average tickers available per day: {availability_df['num_available'].mean():.1f}")
    print(f"Max tickers available: {availability_df['num_available'].max()}")
    print(f"Min tickers available: {availability_df['num_available'].min()}")
    
    return availability_df

def download_data():
    """Download Close/High/Low data from Yahoo Finance with improved missing data handling"""
    print("Downloading data from Yahoo Finance...")
    
    try:
        price_all = yf.download(TICKER_LIST, start=START_DATE, end=END_DATE, progress=False)
        
        if len(TICKER_LIST) > 1:
            close_df = price_all['Close']
            high_df = price_all['High']
            low_df = price_all['Low']
        else:
            # If only one ticker, ensure DataFrame shape
            close_df = price_all[['Close']].rename(columns={'Close': TICKER_LIST[0]})
            high_df = price_all[['High']].rename(columns={'High': TICKER_LIST[0]})
            low_df = price_all[['Low']].rename(columns={'Low': TICKER_LIST[0]})
        
        # Forward fill small gaps (max 5 days)
        close_df = close_df.fillna(method='ffill', limit=5)
        high_df = high_df.fillna(method='ffill', limit=5)
        low_df = low_df.fillna(method='ffill', limit=5)
        
        print(f"Downloaded Close shape: {close_df.shape}")
        print("Close sample:")
        print(close_df.head())
        
        return close_df, high_df, low_df
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None, None, None

def evaluate_model(model, test_env, test_close_df, ticker_availability):
    """Evaluate the trained model with dynamic ticker universe"""
    print("Evaluating PPO agent on test data...")
    
    obs, _ = test_env.reset()
    done = False
    portfolio_values = []
    actions_taken = []
    weights_data = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        portfolio_values.append(test_env.portfolio_value)
        actions_taken.append(action.copy())
        
        # Calculate current weights for available tickers only
        available_tickers = test_env._get_available_tickers()
        current_prices = np.array([
            test_close_df.iloc[test_env.current_step][ticker] if ticker in available_tickers else 0 
            for ticker in TICKER_LIST
        ])
        portfolio_value = test_env.cash + np.sum(test_env.holdings * current_prices)
        
        weight_row = {'Date': test_close_df.index[test_env.current_step], 'Cash': test_env.cash / portfolio_value if portfolio_value > 0 else 1.0}
        
        for i, ticker in enumerate(TICKER_LIST):
            if ticker in available_tickers and portfolio_value > 0:
                weight_row[ticker] = (test_env.holdings[i] * current_prices[i]) / portfolio_value
            else:
                weight_row[ticker] = 0.0
        
        weight_row['num_available_tickers'] = len(available_tickers)
        weights_data.append(weight_row)
        
        done = done or truncated
    
    # Create dynamic equal-weighted benchmark
    equal_weight_values = []
    initial_value = 100000
    equal_weight_values.append(initial_value)
    
    for i in range(1, len(test_close_df)):
        if i < len(ticker_availability):
            available_tickers = ticker_availability.iloc[i]['available_tickers']
            if len(available_tickers) > 0:
                # Calculate equal-weighted return using only available tickers
                returns = []
                for ticker in available_tickers:
                    if ticker in test_close_df.columns:
                        ret = test_close_df.iloc[i][ticker] / test_close_df.iloc[i-1][ticker] - 1
                        returns.append(ret)
                
                if len(returns) > 0:
                    equal_weight_return = np.mean(returns)
                    new_value = equal_weight_values[-1] * (1 + equal_weight_return)
                    equal_weight_values.append(new_value)
                else:
                    equal_weight_values.append(equal_weight_values[-1])
            else:
                equal_weight_values.append(equal_weight_values[-1])
        else:
            equal_weight_values.append(equal_weight_values[-1])
    
    return portfolio_values, actions_taken, weights_data, equal_weight_values

class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        if len(self.locals.get('rewards', [])) > 0:
            self.episode_rewards.extend(self.locals['rewards'])
        return True

def main():
    print("=== Dynamic Ticker Universe RL Trading Bot (Single-Env Baseline) ===")
    
    # Download data
    close_df, high_df, low_df = download_data()
    if close_df is None:
        return
    
    # Create ticker availability matrix
    ticker_availability = create_ticker_availability(close_df)
    
    # Split data
    train_close_df = close_df[close_df.index < TRAIN_END_DATE].copy()
    test_close_df = close_df[close_df.index >= TRAIN_END_DATE].copy()
    train_high_df = high_df[high_df.index < TRAIN_END_DATE].copy()
    test_high_df = high_df[high_df.index >= TRAIN_END_DATE].copy()
    train_low_df = low_df[low_df.index < TRAIN_END_DATE].copy()
    test_low_df = low_df[low_df.index >= TRAIN_END_DATE].copy()
    
    train_availability = ticker_availability[ticker_availability['date'] < TRAIN_END_DATE].copy()
    test_availability = ticker_availability[ticker_availability['date'] >= TRAIN_END_DATE].copy()
    
    print(f"Training period: {train_close_df.index[0].date()} to {train_close_df.index[-1].date()}")
    print(f"Testing period: {test_close_df.index[0].date()} to {test_close_df.index[-1].date()}")
    
    # Create environments (single-env baseline)
    train_env = DynamicTradingEnv(train_close_df, train_high_df, train_low_df, train_availability, INITIAL_CAPITAL)
    test_env = DynamicTradingEnv(test_close_df, test_high_df, test_low_df, test_availability, INITIAL_CAPITAL)
    
    # Wrap training environment as DummyVecEnv (single env)
    train_env = DummyVecEnv([lambda: train_env])
    
    print("Training PPO agent (single-env baseline)...")
    
    # Setup device for M4 Max optimization
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create PPO model (baseline hyperparameters)
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device=device,
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
            activation_fn=torch.nn.ReLU
        )
    )
    
    # Train with progress tracking
    callback = ProgressCallback()
    model.learn(total_timesteps=100000, callback=callback, progress_bar=True)
    
    # Test the trained model
    portfolio_values, actions_taken, weights_data, equal_weight_values = evaluate_model(model, test_env, test_close_df, test_availability)
    
    # Create results DataFrame
    test_dates = test_close_df.index[MIN_HISTORY_DAYS:MIN_HISTORY_DAYS+len(portfolio_values)]
    results_df = pd.DataFrame({
        'Date': test_dates,
        'PPO_Portfolio': portfolio_values
    })
    
    # Calculate Equal-Weighted benchmark
    results_df['Equal_Weight_Benchmark'] = equal_weight_values[:len(portfolio_values)]
    
    # Calculate performance metrics
    ppo_final = results_df['PPO_Portfolio'].iloc[-1]
    eq_weight_final = results_df['Equal_Weight_Benchmark'].iloc[-1]
    
    ppo_return = (ppo_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    eq_weight_return = (eq_weight_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    print(f"\n=== DYNAMIC TICKER UNIVERSE RESULTS (Single-Env) ===")
    print(f"PPO Agent Final Value: ${ppo_final:,.2f}")
    print(f"Dynamic Equal-Weight Final Value: ${eq_weight_final:,.2f}")
    print(f"PPO Agent Return: {ppo_return:.2f}%")
    print(f"Dynamic Equal-Weight Return: {eq_weight_return:.2f}%")
    print(f"RL Outperformance: {ppo_return - eq_weight_return:.2f}%")
    
    # Additional metrics
    ppo_volatility = results_df['PPO_Portfolio'].pct_change().std() * np.sqrt(252) * 100
    eq_weight_volatility = results_df['Equal_Weight_Benchmark'].pct_change().std() * np.sqrt(252) * 100
    
    print(f"PPO Volatility (annualized): {ppo_volatility:.2f}%")
    print(f"Equal-Weight Volatility (annualized): {eq_weight_volatility:.2f}%")
    
    if ppo_volatility > 0:
        ppo_sharpe = ppo_return / ppo_volatility
        eq_weight_sharpe = eq_weight_return / eq_weight_volatility
        print(f"PPO Sharpe Ratio: {ppo_sharpe:.3f}")
        print(f"Equal-Weight Sharpe Ratio: {eq_weight_sharpe:.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Main performance chart
    plt.subplot(2, 1, 1)
    plt.plot(results_df['Date'], results_df['PPO_Portfolio'], label='PPO RL Agent', linewidth=2, color='blue')
    plt.plot(results_df['Date'], results_df['Equal_Weight_Benchmark'], label='Dynamic Equal-Weight Benchmark', linewidth=2, color='red')
    plt.title('PPO RL Agent vs Dynamic Equal-Weight Benchmark (Single-Env)', fontsize=16)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Returns comparison
    plt.subplot(2, 1, 2)
    ppo_returns = results_df['PPO_Portfolio'].pct_change().fillna(0)
    eq_weight_returns = results_df['Equal_Weight_Benchmark'].pct_change().fillna(0)
    plt.plot(results_df['Date'], ppo_returns.cumsum(), label='PPO Cumulative Returns', linewidth=2, color='blue')
    plt.plot(results_df['Date'], eq_weight_returns.cumsum(), label='Equal-Weight Cumulative Returns', linewidth=2, color='red')
    plt.title('Cumulative Returns Comparison', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Returns', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    # Save as PDF
    plt.savefig('rl_trading_performance_single.pdf', dpi=300, bbox_inches='tight')
    print(f"RL performance chart saved as: rl_trading_performance_single.pdf")
    
    # Save detailed results to Excel
    results_df.to_excel('rl_trading_results_single.xlsx', index=False)
    print(f"RL detailed results saved as: rl_trading_results_single.xlsx")
    
    # Save weights data to Excel
    weights_df = pd.DataFrame(weights_data)
    weights_df.to_excel('rl_portfolio_weights_single.xlsx', index=False)
    print(f"Portfolio weights saved as: rl_portfolio_weights_single.xlsx")
    
    # Save ticker availability data
    ticker_availability.to_excel('ticker_availability_single.xlsx', index=False)
    print(f"Ticker availability data saved as: ticker_availability_single.xlsx")
    
    # ----- Metrics Report (CAGR, Ann. Vol, Sharpe, Max Drawdown) -----
    def compute_max_drawdown(values: pd.Series) -> float:
        cum_max = values.cummax()
        dd = values / cum_max - 1.0
        return float(dd.min()) if len(dd) > 0 else 0.0

    def compute_metrics_from_values(values: pd.Series) -> dict:
        values = values.dropna()
        if len(values) < 2:
            return {"CAGR_%": 0.0, "AnnVol_%": 0.0, "Sharpe": 0.0, "MaxDrawdown_%": 0.0}
        daily_rets = values.pct_change().dropna()
        # Annualization factors
        ann_factor = 252.0
        years = len(values) / ann_factor
        final = values.iloc[-1]
        initial = values.iloc[0]
        cagr = (final / max(initial, 1e-9)) ** (1.0 / max(years, 1e-9)) - 1.0
        ann_vol = daily_rets.std(ddof=1) * np.sqrt(ann_factor)
        sharpe = (daily_rets.mean() / daily_rets.std(ddof=1)) * np.sqrt(ann_factor) if daily_rets.std(ddof=1) > 0 else 0.0
        mdd = compute_max_drawdown(values)
        return {
            "CAGR_%": float(cagr * 100.0),
            "AnnVol_%": float(ann_vol * 100.0),
            "Sharpe": float(sharpe),
            "MaxDrawdown_%": float(mdd * 100.0),
        }

    # Align values
    ppo_vals = results_df['PPO_Portfolio']
    ew_vals = results_df['Equal_Weight_Benchmark']
    # Net portfolio: daily excess returns (PPO - EW), cumulated from 100,000
    net_initial = INITIAL_CAPITAL
    ppo_rets = ppo_vals.pct_change().fillna(0.0)
    ew_rets = ew_vals.pct_change().fillna(0.0)
    net_rets = (ppo_rets - ew_rets)
    net_vals = pd.Series(net_initial * (1.0 + net_rets).cumprod(), index=results_df['Date'])

    metrics = []
    metrics.append({"Portfolio": "PPO_Trained_SingleEnv", **compute_metrics_from_values(ppo_vals)})
    metrics.append({"Portfolio": "Equal_Weight", **compute_metrics_from_values(ew_vals)})
    metrics.append({"Portfolio": "Net(PPO-EW)", **compute_metrics_from_values(net_vals)})

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df[["Portfolio", "CAGR_%", "AnnVol_%", "Sharpe", "MaxDrawdown_%"]]
    metrics_df.to_excel('rl_metrics_report_single.xlsx', index=False)
    print("Metrics report saved as: rl_metrics_report_single.xlsx")
    
    return results_df, model

if __name__ == "__main__":
    main()
