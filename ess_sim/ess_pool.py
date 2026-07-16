"""The ESS respondent pool: build it from ESS10 microdata, load it, draw citizens from it.

Build once (read -> filter -> save -> plot):   python -m ess_sim.ess_pool
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

_DATA_DIR      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_ESS_POOL_PATH = os.path.join(_DATA_DIR, "ess_respondent_pool.csv")
RAW_PATH       = os.path.join(_DATA_DIR, "ESS10e03_3.csv")
_ESS_POOL: list = []


#   wrclmch  6=not applicable, 7=refusal, 8=don't know, 9=no answer
#   agea     999=not available (and we scope the study to adults)
#   gndr     9=no answer
#   eisced   55=other, 77=refusal, 88=don't know, 99=no answer
VALID = {"wrclmch": (1, 5), "agea": (18, 100), "gndr": (1, 2), "eisced": (1, 7)}

_POOL_RANGES = {"climate_concern": (1, 5), "agea": (18, 100), "gndr": (1, 2), "eisced": (1, 7)}
_POOL_COLUMNS = tuple(_POOL_RANGES) + ("cntry",)

_REGEN = ("Build it with `python -m ess_sim.ess_pool` from the ESS Round 10 microdata "
          "(free from the ESS Data Portal after registration).")


def build_pool(raw_path=RAW_PATH, out_path=None, plot=True) -> pd.DataFrame:
    """Read the ESS10 microdata, drop missing answers, keep our variables, save + plot."""
    global _ESS_POOL
    out_path = out_path or _ESS_POOL_PATH
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"ESS10 microdata not found at {raw_path}. Download the integrated file from "
            f"the ESS Data Portal (https://ess.sikt.no) and place it there."
        )
    df = pd.read_csv(raw_path, usecols=lambda c: c in list(VALID) + ["cntry"], low_memory=False)
    n_raw = len(df)

    print(f"ESS10 raw N = {n_raw:,}   ({os.path.basename(raw_path)})\n")
    print(f"{'filter':<24}{'dropped':>10}{'remaining':>12}")
    print("-" * 46)
    keep = pd.Series(True, index=df.index)
    for col, (lo, hi) in VALID.items():
        ok = df[col].between(lo, hi)
        dropped = int((keep & ~ok).sum())          # newly dropped, given what is already gone
        keep &= ok
        print(f"{col + f' in [{lo}, {hi}]':<24}{dropped:>10,}{int(keep.sum()):>12,}")

    pool = df[keep].rename(columns={"wrclmch": "climate_concern"})
    pool = pool[["climate_concern", "agea", "gndr", "eisced", "cntry"]].astype(
        {"climate_concern": int, "agea": int, "gndr": int, "eisced": int})
    print("-" * 46)
    print(f"{'POOL':<24}{n_raw - len(pool):>10,}{len(pool):>12,}"
          f"   ({len(pool) / n_raw * 100:.1f}% of ESS10)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pool.to_csv(out_path, index=False)
    _ESS_POOL = []          # the file just changed underneath the cache; force a reload
    print(f"\n[saved] {out_path}")
    _plot_pool(pool, out_path, plot)
    return pool


def _plot_pool(pool, out_path, plot=True):
    x = pool["climate_concern"]
    pct = x.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0) / len(x) * 100
    print(f"\n  climate_concern   mean = {x.mean():.4f}   SD = {x.std(ddof=0):.4f}")
    for lv in [1, 2, 3, 4, 5]:
        print(f"    {lv}  {pct[lv]:5.1f}%  {'#' * int(round(pct[lv] * 0.8))}")
    if not plot:
        return

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([1, 2, 3, 4, 5], pct.values, 0.62, color="#2166ac", edgecolor="white")
    for lv in [1, 2, 3, 4, 5]:
        ax.text(lv, pct[lv] + 0.7, f"{pct[lv]:.0f}%", ha="center", fontsize=9)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Climate concern (wrclmch)")
    ax.set_ylabel("Share of respondents (%)")
    ax.set_title(f"ESS10 respondent pool (N = {len(x):,})\n"
                 f"mean {x.mean():.2f}, SD {x.std(ddof=0):.2f}", fontsize=11)
    ax.text(-0.02, -0.15, "Not worried", transform=ax.transAxes, fontsize=8, color="#555555")
    ax.text(0.9, -0.15, "Extremely worried", transform=ax.transAxes, fontsize=8, color="#555555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    plt.tight_layout()

    png = os.path.join(os.path.dirname(out_path), "pool_concern_distribution.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    if matplotlib.get_backend().lower() in ("agg", "pdf", "ps", "svg", "template"):
        plt.close(fig)
    else:
        plt.show()
    print(f"[saved] {png}")


def _load_ess_pool() -> list:
    global _ESS_POOL
    if _ESS_POOL:
        return _ESS_POOL
    if not os.path.exists(_ESS_POOL_PATH):
        raise FileNotFoundError(f"ESS respondent pool not found at {_ESS_POOL_PATH}. {_REGEN}")
    df = pd.read_csv(_ESS_POOL_PATH)

    missing = [c for c in _POOL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"ESS pool at {_ESS_POOL_PATH} is missing column(s) {missing}; "
                         f"expected {list(_POOL_COLUMNS)}. {_REGEN}")
    for col, (lo, hi) in _POOL_RANGES.items():
        bad = int((~df[col].between(lo, hi)).sum())   # NaN fails between() too
        if bad:
            raise ValueError(f"ESS pool at {_ESS_POOL_PATH}: {bad} row(s) have '{col}' outside "
                             f"[{lo}, {hi}] or missing. {_REGEN}")

    _ESS_POOL = df.to_dict("records")
    print(f"[INFO] Loaded ESS respondent pool: {len(_ESS_POOL)} records from {_ESS_POOL_PATH}")
    return _ESS_POOL


def sample_citizen_rows(seed: int, citizen_nodes, pool=None) -> dict:

    pool = pool or _load_ess_pool()
    return {node: random.Random(seed + node * 31337).choice(pool) for node in citizen_nodes}


def run_pool_diagnostics():

    _DIAG_DATA_DIR = _DATA_DIR
    HIGH_MIN, LOW_MAX = 4, 2   # wrclmch >=4 -> high_concern ; <=2 -> low_concern ; 3 excluded

    _pool_df = pd.DataFrame(_load_ess_pool())
    _N = len(_pool_df)
    print("=" * 70)
    print(f"ESS10 RESPONDENT POOL DIAGNOSTICS   (N = {_N})")
    print("=" * 70)

    # ---------- TABLE A.1 : concern distribution + structure ----------
    print("\n--- TABLE A.1  climate_concern (wrclmch) distribution ---")
    _vc = _pool_df["climate_concern"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    _distA = pd.DataFrame({"count": _vc, "pct": (_vc / _N * 100).round(2)})
    _distA.index.name = "wrclmch"
    _maxc = int(_vc.max())
    for _lvl in [1, 2, 3, 4, 5]:
        _bar = "#" * int(round(_vc[_lvl] / _maxc * 40))
        print(f"  {_lvl}  {_vc[_lvl]:6d}  {_vc[_lvl] / _N * 100:5.1f}%  {_bar}")

    _x = _pool_df["climate_concern"].to_numpy(float)
    _g = skew(_x, bias=False)
    _k = kurtosis(_x, fisher=True, bias=False)
    _bc = (_g ** 2 + 1) / (_k + 3 * (_N - 1) ** 2 / ((_N - 2) * (_N - 3)))   # Sarle coeff (cf. metric_bimodality)
    print(f"\n  Sarle bimodality coefficient = {_bc:.4f}   (> 0.555 suggests bimodal/flat; <= 0.555 unimodal)")
    print(f"  median = {np.median(_x):.1f}   mode = {int(_pool_df['climate_concern'].mode().iloc[0])}")
    print(f"  middle category wrclmch==3 : {_vc[3]} = {_vc[3] / _N * 100:.1f}%  (share discarded by the pole split)")
    print(f"  wrclmch==1 total = {_vc[1]}   wrclmch==5 total = {_vc[5]}")

    _high = _pool_df[_pool_df["climate_concern"] >= HIGH_MIN]
    _low = _pool_df[_pool_df["climate_concern"] <= LOW_MAX]

    # ---------- TABLE B : sample characteristics ----------
    def _col_stats(sub):
        return {
            "n": len(sub),
            "climate_concern (wrclmch), mean (SD)": f"{sub['climate_concern'].mean():.2f} ({sub['climate_concern'].std():.2f})",
            "age (agea), mean (SD)":                f"{sub['agea'].mean():.1f} ({sub['agea'].std():.1f})",
            "gender, % female":                     f"{(sub['gndr'] == 2).mean() * 100:.1f}%",
            "education, % tertiary (EISCED>=6)":     f"{(sub['eisced'] >= 6).mean() * 100:.1f}%",
            "country, top-3 (%)":                   "; ".join(f"{c} {v / len(sub) * 100:.0f}%"
                                                              for c, v in sub['cntry'].value_counts().head(3).items()),
        }

    _tableB = pd.DataFrame({
        f"High-concern (wrclmch>={HIGH_MIN})": _col_stats(_high),
        f"Low-concern (wrclmch<={LOW_MAX})":   _col_stats(_low),
        "Full ESS10 pool":                     _col_stats(_pool_df),
    })
    _tableB = _tableB.reindex(["n"] + [r for r in _tableB.index if r != "n"])
    print("\n--- TABLE B  Sample characteristics  (category 3 excluded from the two pole pools) ---")
    print(_tableB.to_string())

    # ---------- save CSVs ----------
    _distA.to_csv(os.path.join(_DIAG_DATA_DIR, "pool_diag_A1_concern_distribution.csv"))
    _tableB.to_csv(os.path.join(_DIAG_DATA_DIR, "pool_diag_B_sample_characteristics.csv"))
    print("\n[saved] data/pool_diag_A1_concern_distribution.csv, "
          "pool_diag_B_sample_characteristics.csv")


if __name__ == "__main__":
    build_pool()
