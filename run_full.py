"""Run the ESS-LLM experiment grid from the command line.

The experiment grid lives in Python (ess_sim.config.generate_ess_conditions), so the
full set of conditions is transparent and reproducible; this script is a thin CLI +
resumable runner around it.

Examples
--------
    python run_full.py                                   # full grid: 3 seeds x 3 topo x 3 interv = 27
    python run_full.py --mode medium                     # 1 seed  x 3 topo x 3 interv = 9
    python run_full.py --seeds 42,123                     # custom seeds (full-style grid)
    python run_full.py --mode single --topology centralized --intervention denial --seed 42
"""
import os
import time
import random
import argparse
import traceback

import numpy as np
import pandas as pd

from ess_sim.config import ExperimentConfig, generate_ess_conditions
from ess_sim.model import World
from ess_sim.llm_client import LLM_STATS

# gpt-4o-mini pricing (USD per 1M tokens); matches runs/_run_report.json assumption.
_PRICE_IN, _PRICE_OUT = 0.15, 0.60


def _is_complete(cfg) -> bool:
    """A condition is done if its model_summary.csv already has step_count+1 rows."""
    ms = os.path.join(cfg.exp_dir, cfg.exp_id, "model_summary.csv")
    return os.path.exists(ms) and len(pd.read_csv(ms)) >= cfg.step_count + 1


def _build_configs(args) -> list:
    if args.mode == "single":
        return [ExperimentConfig(
            network_type=args.topology, num_agents=args.num_agents, step_count=args.step_count,
            seed=args.seed, network_seed=args.seed, max_interactions=1,
            fca_enabled=(args.intervention == "fca"),
            denial_enabled=(args.intervention == "denial"),
            exp_dir=args.exp_dir,
        )]
    seeds = [42] if args.mode == "medium" else args.seeds
    return generate_ess_conditions(num_agents=args.num_agents, step_count=args.step_count,
                                   seeds=seeds)


def run(configs) -> None:
    """Run each condition; skip any that are already complete (resumable)."""
    LLM_STATS.update(calls=0, prompt_tokens=0, completion_tokens=0, failures=0)
    t0 = time.time()
    failed = []
    for i, cfg in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] {cfg.exp_id}")
        if _is_complete(cfg):
            print("  -> SKIP (already complete)")
            continue
        try:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            w = World(cfg, load_network=True)
            w.run_model(step_count=cfg.step_count)
            print(f"  Mean: {w.compute_mean_concern():.3f} | Disp: {w.compute_dispersion():.3f}")
        except Exception:
            # Keep the batch going, but keep the traceback and make the summary say so --
            # a dead condition leaves no model_summary.csv and would silently shrink any aggregate.
            failed.append(cfg.exp_id)
            traceback.print_exc()
    cost = LLM_STATS["prompt_tokens"] / 1e6 * _PRICE_IN + LLM_STATS["completion_tokens"] / 1e6 * _PRICE_OUT
    status = "DONE" if not failed else f"DONE WITH {len(failed)} FAILED"
    print(f"\n[{status}] {time.time() - t0:.0f}s | LLM calls: {LLM_STATS['calls']} | "
          f"tokens in/out: {LLM_STATS['prompt_tokens']}/{LLM_STATS['completion_tokens']} | "
          f"est. cost: ${cost:.4f}")
    if LLM_STATS["failures"]:
        print(f"[WARN] {LLM_STATS['failures']} LLM call(s) exhausted every retry and were "
              f"recorded as 'no change' -- this biases the metrics toward the null.")
    if failed:
        print(f"[WARN] conditions with NO outputs written: {', '.join(failed)}")


def _parse_args():
    p = argparse.ArgumentParser(description="Run the ESS-LLM experiment grid (resumable).")
    p.add_argument("--mode", choices=["full", "medium", "single"], default="full",
                   help="full: seeds x 3 topo x 3 interv; medium: 1 seed; single: one condition")
    p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[42, 123, 777],
                   help="comma-separated seeds for full mode, e.g. 42,123,777")
    p.add_argument("--num-agents", type=int, default=30)
    p.add_argument("--step-count", type=int, default=20)
    p.add_argument("--exp-dir", default="./runs")
    # single-mode only
    p.add_argument("--topology", choices=["centralized", "small_world", "random"], default="centralized")
    p.add_argument("--intervention", choices=["none", "fca", "denial"], default="none")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    configs = _build_configs(args)
    print(f"Mode: {args.mode} | {len(configs)} condition(s) | N={args.num_agents} | steps={args.step_count}")
    run(configs)
