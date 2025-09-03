#!/usr/bin/env python3
"""
INPUT FILES:
- None (downloads data from Yahoo Finance)

OUTPUT FILES:
- rl_trading_performance.pdf: Performance comparison chart
- rl_trading_results.xlsx: Daily portfolio values and benchmark
- rl_portfolio_weights.xlsx: Ticker weights for each date (variable columns)
- ticker_availability.xlsx: Which tickers were available each date

DESCRIPTION:
Dynamic ticker universe RL trading bot that uses only tickers with complete data
at each time period. The number of available tickers changes over time.

VERSION: 1.0
DATE: 2025-01-02
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
TICKER_LIST = ["CMF", "VCIT", "TLT", "EMB", "LEMB", "HYG", "SPY", "IWM", "QQQ", "DXJ", 
               "EWJV", "IEV", "ASEA", "EMXC", "ILF", "GMF", "CQQQ", "ASHR", "INDA", "SMIN", 
               "EWW", "TUR", "EPOL", "VNM", "MCHI", "VDE", "DBA", "DBB", "VNQ", "COPX", "GDX"]
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
    def __init__(self, df, ticker_availability, initial_amount=100000, transaction_cost_pct=0.001):
        super(DynamicTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.ticker_availability = ticker_availability
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.max_tickers = len(TICKER_LIST)  # Maximum possible tickers
        
        # Fixed action space for maximum possible tickers (pad with zeros for unavailable)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.max_tickers,), dtype=np.float32)
        
        # Fixed observation space for maximum possible tickers
        obs_dim = self.max_tickers * 6 + 3  # prices, 3 returns (20/60/120), ma, vol + cash_ratio + portfolio_change + num_available
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
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        
        available_tickers = self._get_available_tickers()
        
        # Initialize arrays for all possible tickers
        normalized_prices = np.zeros(self.max_tickers)
        returns_20d = np.zeros(self.max_tickers)
        returns_60d = np.zeros(self.max_tickers)
        returns_120d = np.zeros(self.max_tickers)
        moving_averages = np.zeros(self.max_tickers)
        volatilities = np.zeros(self.max_tickers)
        
        # Fill data only for available tickers
        for ticker in available_tickers:
            if ticker in TICKER_LIST:
                i = TICKER_LIST.index(ticker)
                current_price = self.df.iloc[self.current_step][ticker]
                
                # Multiple trailing returns
                if self.current_step >= 20:
                    ret_20d = (current_price / self.df.iloc[self.current_step-20][ticker] - 1)
                else:
                    ret_20d = 0
                returns_20d[i] = ret_20d
                
                if self.current_step >= 60:
                    ret_60d = (current_price / self.df.iloc[self.current_step-60][ticker] - 1)
                else:
                    ret_60d = 0
                returns_60d[i] = ret_60d
                
                if self.current_step >= 120:
                    ret_120d = (current_price / self.df.iloc[self.current_step-120][ticker] - 1)
                else:
                    ret_120d = 0
                returns_120d[i] = ret_120d
                
                # 120-day moving average ratio
                if self.current_step >= 120:
                    ma = self.df.iloc[self.current_step-120:self.current_step][ticker].mean()
                    ma_ratio = current_price / ma - 1 if ma > 0 else 0
                else:
                    ma_ratio = 0
                moving_averages[i] = ma_ratio
                
                # 120-day volatility
                if self.current_step >= 120:
                    vol = self.df.iloc[self.current_step-120:self.current_step][ticker].std()
                    vol_norm = vol / current_price if current_price > 0 else 0
                else:
                    vol_norm = 0
                volatilities[i] = vol_norm
                
                normalized_prices[i] = current_price
        
        # Normalize prices across available tickers only
        available_prices = normalized_prices[normalized_prices > 0]
        if len(available_prices) > 0:
            max_price = available_prices.max()
            if max_price > 0:
                normalized_prices = normalized_prices / max_price
        
        # Portfolio state
        current_prices = np.array([self.df.iloc[self.current_step][ticker] if ticker in available_tickers else 0 
                                  for ticker in TICKER_LIST])
        self.portfolio_value = self.cash + np.sum(self.holdings * current_prices)
        cash_ratio = self.cash / max(self.portfolio_value, 1)
        portfolio_change = (self.portfolio_value / self.initial_amount - 1)
        num_available = len(available_tickers) / self.max_tickers  # Normalized count
        
        # Clean arrays
        returns_20d = np.nan_to_num(returns_20d, nan=0.0, posinf=1.0, neginf=-1.0)
        returns_60d = np.nan_to_num(returns_60d, nan=0.0, posinf=1.0, neginf=-1.0)
        returns_120d = np.nan_to_num(returns_120d, nan=0.0, posinf=1.0, neginf=-1.0)
        moving_averages = np.nan_to_num(moving_averages, nan=0.0, posinf=1.0, neginf=-1.0)
        volatilities = np.nan_to_num(volatilities, nan=0.0, posinf=1.0, neginf=-1.0)
        normalized_prices = np.nan_to_num(normalized_prices, nan=0.0, posinf=1.0, neginf=-1.0)
        
        obs = np.concatenate([
            normalized_prices,
            returns_20d,
            returns_60d,
            returns_120d,
            moving_averages,
            volatilities,
            [cash_ratio, portfolio_change, num_available]
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, True, {}
        
        available_tickers = self._get_available_tickers()
        current_prices = np.array([self.df.iloc[self.current_step][ticker] if ticker in available_tickers else 0 
                                  for ticker in TICKER_LIST])
        
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
        
        if self.current_step < len(self.df):
            next_available_tickers = self._get_available_tickers()
            new_prices = np.array([self.df.iloc[self.current_step][ticker] if ticker in next_available_tickers else 0 
                                  for ticker in TICKER_LIST])
            new_portfolio_value = self.cash + np.sum(self.holdings * new_prices)
        else:
            new_portfolio_value = old_portfolio_value
        
        # Calculate reward
        portfolio_return = (new_portfolio_value - old_portfolio_value) / max(old_portfolio_value, 1)
        reward = portfolio_return
        
        self.portfolio_value = new_portfolio_value
        self.portfolio_values.append(self.portfolio_value)
        
        done = self.current_step >= len(self.df) - 1
        
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
    """Download stock data from Yahoo Finance with improved missing data handling"""
    print("Downloading data from Yahoo Finance...")
    
    try:
        price_df = yf.download(TICKER_LIST, start=START_DATE, end=END_DATE, progress=False)
        
        if len(TICKER_LIST) > 1:
            price_df = price_df['Close']
        
        # Keep all tickers, don't drop any columns
        # Only forward fill small gaps (max 5 days)
        price_df = price_df.fillna(method='ffill', limit=5)
        
        print(f"Downloaded data shape: {price_df.shape}")
        print("Data sample:")
        print(price_df.head())
        
        return price_df
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def evaluate_model(model, test_env, test_df, ticker_availability):
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
        current_prices = np.array([test_df.iloc[test_env.current_step][ticker] if ticker in available_tickers else 0 
                                  for ticker in TICKER_LIST])
        portfolio_value = test_env.cash + np.sum(test_env.holdings * current_prices)
        
        weight_row = {'Date': test_df.index[test_env.current_step], 'Cash': test_env.cash / portfolio_value if portfolio_value > 0 else 1.0}
        
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
    
    for i in range(1, len(test_df)):
        if i < len(ticker_availability):
            available_tickers = ticker_availability.iloc[i]['available_tickers']
            if len(available_tickers) > 0:
                # Calculate equal-weighted return using only available tickers
                returns = []
                for ticker in available_tickers:
                    if ticker in test_df.columns:
                        ret = test_df.iloc[i][ticker] / test_df.iloc[i-1][ticker] - 1
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
    print("=== Dynamic Ticker Universe RL Trading Bot ===")
    
    # Download data
    price_df = download_data()
    if price_df is None:
        return
    
    # Create ticker availability matrix
    ticker_availability = create_ticker_availability(price_df)
    
    # Split data
    train_df = price_df[price_df.index < TRAIN_END_DATE].copy()
    test_df = price_df[price_df.index >= TRAIN_END_DATE].copy()
    
    train_availability = ticker_availability[ticker_availability['date'] < TRAIN_END_DATE].copy()
    test_availability = ticker_availability[ticker_availability['date'] >= TRAIN_END_DATE].copy()
    
    print(f"Training period: {train_df.index[0].date()} to {train_df.index[-1].date()}")
    print(f"Testing period: {test_df.index[0].date()} to {test_df.index[-1].date()}")
    
    # Create environments
    train_env = DynamicTradingEnv(train_df, train_availability, INITIAL_CAPITAL)
    test_env = DynamicTradingEnv(test_df, test_availability, INITIAL_CAPITAL)
    
    # Wrap training environment
    train_env = DummyVecEnv([lambda: train_env])
    
    print("Training PPO agent...")
    
    # Setup device for M4 Max optimization
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create PPO model
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
    portfolio_values, actions_taken, weights_data, equal_weight_values = evaluate_model(model, test_env, test_df, test_availability)
    
    # Create results DataFrame
    test_dates = test_df.index[MIN_HISTORY_DAYS:MIN_HISTORY_DAYS+len(portfolio_values)]
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
    
    print(f"\n=== DYNAMIC TICKER UNIVERSE RESULTS ===")
    print(f"PPO Agent Final Value: ${ppo_final:,.2f}")
    print(f"Dynamic Equal-Weight Benchmark Final Value: ${eq_weight_final:,.2f}")
    print(f"PPO Agent Return: {ppo_return:.2f}%")
    print(f"Dynamic Equal-Weight Benchmark Return: {eq_weight_return:.2f}%")
    print(f"RL Outperformance: {ppo_return - eq_weight_return:.2f}%")
    
    # Calculate additional metrics
    ppo_volatility = results_df['PPO_Portfolio'].pct_change().std() * np.sqrt(252) * 100
    eq_weight_volatility = results_df['Equal_Weight_Benchmark'].pct_change().std() * np.sqrt(252) * 100
    
    print(f"PPO Volatility (annualized): {ppo_volatility:.2f}%")
    print(f"Dynamic Equal-Weight Volatility (annualized): {eq_weight_volatility:.2f}%")
    
    if ppo_volatility > 0:
        ppo_sharpe = ppo_return / ppo_volatility
        eq_weight_sharpe = eq_weight_return / eq_weight_volatility
        print(f"PPO Sharpe Ratio: {ppo_sharpe:.3f}")
        print(f"Dynamic Equal-Weight Sharpe Ratio: {eq_weight_sharpe:.3f}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Main performance chart
    plt.subplot(2, 1, 1)
    plt.plot(results_df['Date'], results_df['PPO_Portfolio'], 
             label='PPO RL Agent', linewidth=2, color='blue')
    plt.plot(results_df['Date'], results_df['Equal_Weight_Benchmark'], 
             label='Dynamic Equal-Weight Benchmark', linewidth=2, color='red')
    plt.title('PPO RL Agent vs Dynamic Equal-Weight Benchmark', fontsize=16)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Returns comparison
    plt.subplot(2, 1, 2)
    ppo_returns = results_df['PPO_Portfolio'].pct_change().fillna(0)
    eq_weight_returns = results_df['Equal_Weight_Benchmark'].pct_change().fillna(0)
    plt.plot(results_df['Date'], ppo_returns.cumsum(), 
             label='PPO Cumulative Returns', linewidth=2, color='blue')
    plt.plot(results_df['Date'], eq_weight_returns.cumsum(), 
             label='Dynamic Equal-Weight Cumulative Returns', linewidth=2, color='red')
    plt.title('Cumulative Returns Comparison', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Returns', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    # Save as PDF
    plt.savefig('rl_trading_performance.pdf', dpi=300, bbox_inches='tight')
    print(f"RL performance chart saved as: rl_trading_performance.pdf")
    plt.show()
    
    # Save detailed results to Excel
    results_df.to_excel('rl_trading_results.xlsx', index=False)
    print(f"RL detailed results saved as: rl_trading_results.xlsx")
    
    # Save weights data to Excel
    weights_df = pd.DataFrame(weights_data)
    weights_df.to_excel('rl_portfolio_weights.xlsx', index=False)
    print(f"Portfolio weights saved as: rl_portfolio_weights.xlsx")
    
    # Save ticker availability data
    ticker_availability.to_excel('ticker_availability.xlsx', index=False)
    print(f"Ticker availability data saved as: ticker_availability.xlsx")
    
    return results_df, model

if __name__ == "__main__":
    main()
