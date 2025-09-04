#!/usr/bin/env python3
"""
INPUT FILES (prominent):
- None required. Prices are downloaded from Yahoo Finance at runtime.

OUTPUT FILES (prominent):
- rl_trading_performance_single_seed{SEED}.pdf: Performance plot for this seed
- rl_trading_results_single_seed{SEED}.xlsx: Daily values for PPO portfolio and equal-weight benchmark
- rl_portfolio_weights_single_seed{SEED}.xlsx: Daily portfolio weights for PPO

Version history:
- v1.0 (2025-09-03): Initial single-seed runner for single-env PPO.

Notes:
- Uses single environment (DummyVecEnv) and the same PPO hyperparameters as `dynamic_rl_bot_single.py`.
- Sets Python, NumPy, and Torch seeds; passes seed into PPO and env reset for reproducibility.
- For exact reproducibility of data, consider caching prices to XLSX before running.
"""

print("DEBUG: Script started", flush=True)
import argparse
print("DEBUG: Imported argparse", flush=True)
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

print("DEBUG: Importing from dynamic_rl_bot_single...", flush=True)
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
print("DEBUG: Imports from dynamic_rl_bot_single complete", flush=True)


def run_one_seed(seed: int):
    print(f"=== Single-Env PPO run (seed={seed}) ===", flush=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Global RNG seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Data
    print("DEBUG: Calling download_data()", flush=True)
    close_df, high_df, low_df = download_data()
    print("DEBUG: download_data() complete", flush=True)
    if close_df is None:
        raise SystemExit(1)

    ticker_availability = create_ticker_availability(close_df)

    # Split
    train_close_df = close_df[close_df.index < TRAIN_END_DATE].copy()
    test_close_df = close_df[close_df.index >= TRAIN_END_DATE].copy()
    train_high_df = high_df[high_df.index < TRAIN_END_DATE].copy()
    test_high_df = high_df[high_df.index >= TRAIN_END_DATE].copy()
    train_low_df = low_df[low_df.index < TRAIN_END_DATE].copy()
    test_low_df = low_df[low_df.index >= TRAIN_END_DATE].copy()

    train_availability = ticker_availability[ticker_availability['date'] < TRAIN_END_DATE].copy()
    test_availability = ticker_availability[ticker_availability['date'] >= TRAIN_END_DATE].copy()

    # Envs
    train_env_single = DynamicTradingEnv(train_close_df, train_high_df, train_low_df, train_availability, INITIAL_CAPITAL)
    test_env_single = DynamicTradingEnv(test_close_df, test_high_df, test_low_df, test_availability, INITIAL_CAPITAL)

    train_env = DummyVecEnv([lambda: train_env_single])
    try:
        train_env.reset(seed=seed)
    except Exception:
        pass

    # PPO
    print("DEBUG: Instantiating PPO model...", flush=True)
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=seed,
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

    print("DEBUG: PPO model instantiated", flush=True)
    # Train
    callback = ProgressCallback()
    model.learn(total_timesteps=100_000, callback=callback, progress_bar=True)

    # Evaluate
    portfolio_values, actions_taken, weights_data, equal_weight_values = evaluate_model(
        model, test_env_single, test_close_df, test_availability
    )

    # Results
    test_dates = test_close_df.index[MIN_HISTORY_DAYS:MIN_HISTORY_DAYS+len(portfolio_values)]
    results_df = pd.DataFrame({
        'Date': test_dates,
        'PPO_Portfolio': portfolio_values
    })
    results_df['Equal_Weight_Benchmark'] = equal_weight_values[:len(portfolio_values)]

    # Save
    pdf_file = f"rl_trading_performance_single_seed{seed}.pdf"
    xlsx_results = f"rl_trading_results_single_seed{seed}.xlsx"
    xlsx_weights = f"rl_portfolio_weights_single_seed{seed}.xlsx"

    plt.figure(figsize=(12, 7))
    plt.plot(results_df['Date'], results_df['PPO_Portfolio'], label=f'PPO RL Agent (seed {seed})', linewidth=2)
    plt.plot(results_df['Date'], results_df['Equal_Weight_Benchmark'], label='Dynamic Equal-Weight', linewidth=2)
    plt.title(f'PPO vs Dynamic Equal-Weight (Single-Env, seed={seed})')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {pdf_file}")

    results_df.to_excel(xlsx_results, index=False)
    print(f"Saved: {xlsx_results}")
    weights_df = pd.DataFrame(weights_data)
    weights_df.to_excel(xlsx_weights, index=False)
    print(f"Saved: {xlsx_weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run single-env PPO for one seed and save outputs.')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for this run')
    args = parser.parse_args()
    run_one_seed(args.seed)
