#!/usr/bin/env python3
"""
INPUT FILES (name):
- (yfinance) dynamic download of close/high/low for TICKER_LIST

OUTPUT FILES (name):
- rl_trading_performance_vec42.pdf: Performance chart for PPO (vec, seed=42) vs Dynamic Equal-Weight
- rl_trading_results_vec42.xlsx: Daily portfolio values for PPO and benchmark
- rl_portfolio_weights_vec42.xlsx: Daily portfolio weights (columns vary with availability)
- ticker_availability_vec42.xlsx: Daily available tickers summary
- rl_metrics_report_vec42.xlsx: Summary metrics (CAGR, AnnVol, Sharpe, MaxDD)

DESCRIPTION:
Vectorized (8-env SubprocVecEnv) PPO training with total_timesteps=800,000 and fixed seed=42 to
match the update count of the original single-env baseline while using parallel environment stepping
for faster wall-clock training. Evaluation remains single-env for consistent reporting.

VERSION HISTORY:
- v1.0 (2025-09-03): Initial release (vec 8 envs, seed=42, 800k steps). Output filenames suffixed with _vec42.

Notes on Missing Data Handling:
- We forward-fill up to 5 business days per ticker (small gaps); dynamic availability excludes tickers
  without full MIN_HISTORY_DAYS. All replacements are implicit in the availability matrix and logs
  (average, min, max available per day). For stricter reproducibility, consider caching prices to xlsx.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Reuse environment and helpers from the single-env baseline to ensure identical logic
from dynamic_rl_bot_single import (
    DynamicTradingEnv,
    create_ticker_availability,
    download_data,
    evaluate_model,
    ProgressCallback,
    TICKER_LIST,
    MIN_HISTORY_DAYS,
    TRAIN_END_DATE,
    INITIAL_CAPITAL,
)

SEED = 42
NUM_ENVS = 8
TOTAL_TIMESTEPS = 800_000  # 8x 100k to preserve update count with n_steps=2048


def main():
    print("=== Dynamic Ticker Universe RL Trading Bot (Vectorized 8-env, Seed=42) ===")
    print(f"Seed: {SEED}, Num Envs: {NUM_ENVS}, Total Timesteps: {TOTAL_TIMESTEPS}")

    # Global seeding
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    set_random_seed(SEED)

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

    # Factory for vectorized training envs, each with unique seed for determinism across processes
    def make_env(rank: int):
        def _init():
            env = DynamicTradingEnv(train_close_df, train_high_df, train_low_df, train_availability, INITIAL_CAPITAL)
            env.reset(seed=SEED + rank)
            return env
        return _init

    # Create vectorized training env (SubprocVecEnv with fallback)
    try:
        train_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
        print(f"Training with SubprocVecEnv, num_envs={NUM_ENVS}")
    except Exception as e:
        print(f"SubprocVecEnv failed ({e}). Falling back to DummyVecEnv with {NUM_ENVS} envs.")
        train_env = DummyVecEnv([make_env(i) for i in range(NUM_ENVS)])

    # Seed vec env
    try:
        train_env.seed(SEED)
    except Exception:
        pass

    # Single-env test environment
    test_env = DynamicTradingEnv(test_close_df, test_high_df, test_low_df, test_availability, INITIAL_CAPITAL)
    test_env.reset(seed=SEED)

    # Setup device for M4 Max optimization
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # PPO model (same hyperparameters as baseline)
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=SEED,
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

    # Train
    callback = ProgressCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)

    # Evaluate on test set (single env)
    portfolio_values, actions_taken, weights_data, equal_weight_values = evaluate_model(
        model, test_env, test_close_df, test_availability
    )

    # Results DataFrame
    test_dates = test_close_df.index[MIN_HISTORY_DAYS:MIN_HISTORY_DAYS+len(portfolio_values)]
    results_df = pd.DataFrame({
        'Date': test_dates,
        'PPO_Portfolio': portfolio_values
    })
    results_df['Equal_Weight_Benchmark'] = equal_weight_values[:len(portfolio_values)]

    # Metrics
    ppo_final = results_df['PPO_Portfolio'].iloc[-1]
    eq_weight_final = results_df['Equal_Weight_Benchmark'].iloc[-1]
    ppo_return = (ppo_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    eq_weight_return = (eq_weight_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    print("\n=== DYNAMIC TICKER UNIVERSE RESULTS (Vec 8, Seed=42) ===")
    print(f"PPO Agent Final Value: ${ppo_final:,.2f}")
    print(f"Dynamic Equal-Weight Final Value: ${eq_weight_final:,.2f}")
    print(f"PPO Agent Return: {ppo_return:.2f}%")
    print(f"Dynamic Equal-Weight Return: {eq_weight_return:.2f}%")
    print(f"RL Outperformance: {ppo_return - eq_weight_return:.2f}%")

    # Additional metrics
    def compute_max_drawdown(values: pd.Series) -> float:
        cum_max = values.cummax()
        dd = values / cum_max - 1.0
        return float(dd.min()) if len(dd) > 0 else 0.0

    def compute_metrics_from_values(values: pd.Series) -> dict:
        values = values.dropna()
        if len(values) < 2:
            return {"CAGR_%": 0.0, "AnnVol_%": 0.0, "Sharpe": 0.0, "MaxDrawdown_%": 0.0}
        daily_rets = values.pct_change().dropna()
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

    ppo_vals = results_df['PPO_Portfolio']
    ew_vals = results_df['Equal_Weight_Benchmark']
    net_initial = INITIAL_CAPITAL
    ppo_rets = ppo_vals.pct_change().fillna(0.0)
    ew_rets = ew_vals.pct_change().fillna(0.0)
    net_rets = (ppo_rets - ew_rets)
    net_vals = pd.Series(net_initial * (1.0 + net_rets).cumprod(), index=results_df['Date'])

    metrics = []
    metrics.append({"Portfolio": "PPO_Trained_Vec8_Seed42", **compute_metrics_from_values(ppo_vals)})
    metrics.append({"Portfolio": "Equal_Weight", **compute_metrics_from_values(ew_vals)})
    metrics.append({"Portfolio": "Net(PPO-EW)", **compute_metrics_from_values(net_vals)})

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df[["Portfolio", "CAGR_%", "AnnVol_%", "Sharpe", "MaxDrawdown_%"]]

    # Visualization
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(results_df['Date'], results_df['PPO_Portfolio'], label='PPO RL Agent (Vec8 Seed42)', linewidth=2, color='blue')
    plt.plot(results_df['Date'], results_df['Equal_Weight_Benchmark'], label='Dynamic Equal-Weight Benchmark', linewidth=2, color='red')
    plt.title('PPO RL Agent vs Dynamic Equal-Weight (Vec8, Seed=42)', fontsize=16)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

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

    # Save outputs
    plt.savefig('rl_trading_performance_vec42.pdf', dpi=300, bbox_inches='tight')
    print("RL performance chart saved as: rl_trading_performance_vec42.pdf")

    results_df.to_excel('rl_trading_results_vec42.xlsx', index=False)
    print("RL detailed results saved as: rl_trading_results_vec42.xlsx")

    weights_df = pd.DataFrame(weights_data)
    weights_df.to_excel('rl_portfolio_weights_vec42.xlsx', index=False)
    print("Portfolio weights saved as: rl_portfolio_weights_vec42.xlsx")

    test_availability.to_excel('ticker_availability_vec42.xlsx', index=False)
    print("Ticker availability data saved as: ticker_availability_vec42.xlsx")

    metrics_df.to_excel('rl_metrics_report_vec42.xlsx', index=False)
    print("Metrics report saved as: rl_metrics_report_vec42.xlsx")

    return results_df, model


if __name__ == "__main__":
    main()
