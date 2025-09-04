#!/usr/bin/env python3
"""
INPUT FILES (prominent):
- Prices (downloaded via yfinance at runtime). No local inputs are required.

OUTPUT FILES (prominent):
- rl_trading_performance_single_seed{SEED}.pdf: Per-seed performance chart (PPO vs Dynamic Equal-Weight)
- rl_trading_results_single_seed{SEED}.xlsx: Per-seed daily series (Date, PPO_Portfolio, Equal_Weight_Benchmark)
- rl_portfolio_weights_single_seed{SEED}.xlsx: Per-seed daily portfolio weights
- seed_sweep_single_summary.xlsx: Aggregate metrics across seeds with reproducibility metadata and data completeness

Version history:
- v1.0 (2025-09-03): Initial seed sweep for single-env PPO across 5 seeds.

Documentation notes:
- Missing data handling: We forward-fill up to 5 business days per ticker (small gaps).
  A dynamic availability matrix excludes tickers lacking a full history window (MIN_HISTORY_DAYS).
  The summary workbook includes a DataCompleteness sheet (% availability by ticker in test period).
- Reproducibility: We set Python random, NumPy, Torch, Gym env reset seeds, and PPO(seed).
  External data may still change day-to-day; for exact reruns, cache prices to XLSX outside this script.
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Reuse core logic and environment from the single-env baseline
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

SEEDS = [0, 42, 123, 2024, 999]
TOTAL_TIMESTEPS = 100_000


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


def main():
    print("=== PPO Single-Env Seed Sweep (5 seeds) ===")
    print(f"Seeds: {SEEDS}")

    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Download once and create availability once for consistent inputs
    close_df, high_df, low_df = download_data()
    if close_df is None:
        return

    ticker_availability = create_ticker_availability(close_df)

    # Split into train/test once
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

    # Data completeness by ticker in test period (% of days available)
    test_len = len(test_availability)
    comp = []
    for t in TICKER_LIST:
        count = 0
        for i in range(test_len):
            if t in test_availability.iloc[i]['available_tickers']:
                count += 1
        pct = (count / max(test_len, 1)) * 100.0
        comp.append({"Ticker": t, "%Available_Test_%": pct})
    completeness_df = pd.DataFrame(comp)

    # Aggregate results across seeds
    summary_rows = []

    # For overlay plot: store (seed, dates, portfolio_values)
    series_by_seed = []
    eq_benchmark = None
    dates_common = None

    for seed in SEEDS:
        print(f"\n--- Running seed {seed} ---")
        # Re-seed all RNGs
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create fresh envs
        train_env_single = DynamicTradingEnv(train_close_df, train_high_df, train_low_df, train_availability, INITIAL_CAPITAL)
        test_env_single = DynamicTradingEnv(test_close_df, test_high_df, test_low_df, test_availability, INITIAL_CAPITAL)

        # Wrap training env
        train_env = DummyVecEnv([lambda: train_env_single])
        # Reset with seed
        try:
            train_env.reset(seed=seed)
        except Exception:
            pass

        # PPO Model
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

        # Train
        callback = ProgressCallback()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)

        # Evaluate
        portfolio_values, actions_taken, weights_data, equal_weight_values = evaluate_model(
            model, test_env_single, test_close_df, test_availability
        )

        # Build results df
        test_dates = test_close_df.index[MIN_HISTORY_DAYS:MIN_HISTORY_DAYS+len(portfolio_values)]
        results_df = pd.DataFrame({
            'Date': test_dates,
            'PPO_Portfolio': portfolio_values
        })
        results_df['Equal_Weight_Benchmark'] = equal_weight_values[:len(portfolio_values)]

        # Save per-seed outputs
        pdf_file = f"rl_trading_performance_single_seed{seed}.pdf"
        xlsx_results = f"rl_trading_results_single_seed{seed}.xlsx"
        xlsx_weights = f"rl_portfolio_weights_single_seed{seed}.xlsx"

        # Plot
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

        # Results and weights
        results_df.to_excel(xlsx_results, index=False)
        print(f"Saved: {xlsx_results}")
        weights_df = pd.DataFrame(weights_data)
        weights_df.to_excel(xlsx_weights, index=False)
        print(f"Saved: {xlsx_weights}")

        # Metrics and collect series
        ppo_vals = results_df['PPO_Portfolio']
        metrics = compute_metrics_from_values(ppo_vals)
        final_val = float(ppo_vals.iloc[-1]) if len(ppo_vals) else float('nan')
        summary_rows.append({
            'Seed': seed,
            'FinalValue': final_val,
            **metrics
        })

        if eq_benchmark is None:
            eq_benchmark = results_df[['Date', 'Equal_Weight_Benchmark']].copy()
        if dates_common is None:
            dates_common = results_df['Date']
        series_by_seed.append((seed, results_df['Date'], results_df['PPO_Portfolio']))

    # Summary table
    summary_df = pd.DataFrame(summary_rows)

    # Descriptive stats across seeds
    desc = summary_df[['FinalValue', 'CAGR_%', 'AnnVol_%', 'Sharpe', 'MaxDrawdown_%']].agg(['mean', 'std', 'min', 'max']).reset_index()
    desc.rename(columns={'index': 'Statistic'}, inplace=True)

    # Save summary workbook with multiple sheets
    with pd.ExcelWriter('seed_sweep_single_summary.xlsx', engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name='PerSeedMetrics', index=False)
        desc.to_excel(writer, sheet_name='AcrossSeedStats', index=False)
        completeness_df.to_excel(writer, sheet_name='DataCompleteness', index=False)
        # Metadata sheet
        meta = pd.DataFrame([
            {'Key': 'Device', 'Value': device},
            {'Key': 'Seeds', 'Value': ','.join(str(s) for s in SEEDS)},
            {'Key': 'TotalTimesteps', 'Value': TOTAL_TIMESTEPS},
            {'Key': 'n_steps', 'Value': 2048},
            {'Key': 'batch_size', 'Value': 128},
            {'Key': 'n_epochs', 'Value': 10},
            {'Key': 'gamma', 'Value': 0.99},
            {'Key': 'gae_lambda', 'Value': 0.95},
            {'Key': 'clip_range', 'Value': 0.2},
            {'Key': 'ent_coef', 'Value': 0.01},
            {'Key': 'PolicyNet', 'Value': '[256,256,128], ReLU'},
            {'Key': 'MissingData', 'Value': 'Forward-fill up to 5 days; dynamic availability excludes incomplete histories'},
        ])
        meta.to_excel(writer, sheet_name='RunMetadata', index=False)

    print("Saved: seed_sweep_single_summary.xlsx")

    # Comparison plot across seeds
    plt.figure(figsize=(15, 10))
    # Plot each seed
    for seed, dts, vals in series_by_seed:
        plt.plot(dts, vals, label=f'PPO (seed {seed})', linewidth=1.8)
    # Plot benchmark (from first)
    if eq_benchmark is not None:
        plt.plot(eq_benchmark['Date'], eq_benchmark['Equal_Weight_Benchmark'], label='Dynamic Equal-Weight', linewidth=2.2, color='black')
    plt.title('PPO Single-Env: Portfolio Value Across Seeds')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig('seed_sweep_single_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: seed_sweep_single_comparison.pdf")

    print("\nSeed sweep completed.")


if __name__ == "__main__":
    main()
