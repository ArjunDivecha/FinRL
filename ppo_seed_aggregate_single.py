#!/usr/bin/env python3
"""
INPUT FILES (prominent):
- rl_trading_results_single_seed{SEED}.xlsx for each seed (produced by ppo_seed_single_run.py)
- Optionally: live download via yfinance to compute DataCompleteness sheet (no local input required)

OUTPUT FILES (prominent):
- seed_sweep_single_summary.xlsx: Per-seed metrics + across-seed stats + data completeness + metadata
- seed_sweep_single_comparison.pdf: Overlay of portfolio values across seeds and the dynamic equal-weight benchmark

Version history:
- v1.0 (2025-09-03): Initial aggregator that waits for per-seed outputs and builds summary.

Notes:
- This script will block until all expected per-seed XLSX result files are present.
- Missing data handling for the DataCompleteness sheet mirrors dynamic_rl_bot_single.create_ticker_availability.
"""

import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from dynamic_rl_bot_single import (
    download_data,
    create_ticker_availability,
    TICKER_LIST,
    MIN_HISTORY_DAYS,
    TRAIN_END_DATE,
)

SEEDS = [0, 42, 123, 2024, 999]
RESULT_TEMPLATE = 'rl_trading_results_single_seed{seed}.xlsx'


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


def wait_for_results(seeds, poll_seconds=15):
    print(f"Waiting for per-seed result files: {seeds}")
    while True:
        missing = []
        for s in seeds:
            path = RESULT_TEMPLATE.format(seed=s)
            if not os.path.exists(path):
                missing.append(path)
        if not missing:
            print("All per-seed results found.")
            return
        print(f"Still missing: {missing}. Checking again in {poll_seconds}s...")
        time.sleep(poll_seconds)


def aggregate(seeds):
    print("Aggregating per-seed results...")

    # Read all seed results
    per_seed_frames = []
    summary_rows = []

    for s in seeds:
        path = RESULT_TEMPLATE.format(seed=s)
        df = pd.read_excel(path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        per_seed_frames.append((s, df))
        metrics = compute_metrics_from_values(df['PPO_Portfolio'])
        final_val = float(df['PPO_Portfolio'].iloc[-1]) if len(df) else float('nan')
        summary_rows.append({'Seed': s, 'FinalValue': final_val, **metrics})

    summary_df = pd.DataFrame(summary_rows)
    desc = summary_df[['FinalValue', 'CAGR_%', 'AnnVol_%', 'Sharpe', 'MaxDrawdown_%']].agg(['mean', 'std', 'min', 'max']).reset_index()
    desc.rename(columns={'index': 'Statistic'}, inplace=True)

    # DataCompleteness sheet (recomputed)
    close_df, high_df, low_df = download_data()
    ticker_availability = create_ticker_availability(close_df)
    test_availability = ticker_availability[ticker_availability['date'] >= TRAIN_END_DATE].copy()
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

    # Metadata
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    meta = pd.DataFrame([
        {'Key': 'Device', 'Value': device},
        {'Key': 'Seeds', 'Value': ','.join(str(s) for s in seeds)},
        {'Key': 'TotalTimesteps', 'Value': 100_000},
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

    with pd.ExcelWriter('seed_sweep_single_summary.xlsx', engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name='PerSeedMetrics', index=False)
        desc.to_excel(writer, sheet_name='AcrossSeedStats', index=False)
        completeness_df.to_excel(writer, sheet_name='DataCompleteness', index=False)
        meta.to_excel(writer, sheet_name='RunMetadata', index=False)

    print("Saved: seed_sweep_single_summary.xlsx")

    # Comparison PDF
    plt.figure(figsize=(15, 10))
    for s, df in per_seed_frames:
        plt.plot(df['Date'], df['PPO_Portfolio'], label=f'PPO (seed {s})', linewidth=1.8)
    # Use any equal-weight from the first df if available
    if len(per_seed_frames) > 0 and 'Equal_Weight_Benchmark' in per_seed_frames[0][1].columns:
        df0 = per_seed_frames[0][1]
        plt.plot(df0['Date'], df0['Equal_Weight_Benchmark'], label='Dynamic Equal-Weight', linewidth=2.2, color='black')
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


def main():
    print("=== Aggregator starting ===")
    wait_for_results(SEEDS)
    aggregate(SEEDS)
    print("Aggregation complete.")


if __name__ == '__main__':
    main()
