
"""Plotting/reporting helpers: network topology comparison and interaction diary.
"""
import os

import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

from .ess_pool import _load_ess_pool, sample_citizen_rows
from .network import generate_network, save_network_structure, load_network_structure

def _show(fig):
    # plt.show() on a non-interactive backend (headless CI, a plain script) only emits a
    # UserWarning and draws nothing; skip it there and let the caller rely on save=.
    if matplotlib.get_backend().lower() in ("agg", "pdf", "ps", "svg", "cairo", "template"):
        plt.close(fig)
        return
    plt.show()


TOPO_COLORS = {"centralized": "#4dac26", "small_world": "#2166ac", "random": "#d6604d"}
TOPO_TITLES = {"centralized": "Centralized (multi-hub)", "small_world": "Small-World",
               "random": "Random (Erdos-Renyi)"}


def _load_or_gen_network(nt, num_agents, seed):
    p = f"./data/{nt}_network_num_agents_{num_agents}_seed_{seed}.json"
    if os.path.exists(p):
        return load_network_structure(p)
    G = generate_network(nt, num_agents, seed=seed)
    save_network_structure(G, p)
    return G


def _topo_layout(G, nt):
    # One layout rule per topology; centralized -> shell with the 3 highest-degree hubs inner ring.
    if nt == "centralized":
        hubs = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:3]
        return nx.shell_layout(G, [hubs, [n for n in G.nodes() if n not in hubs]]), hubs[0]
    bc = nx.betweenness_centrality(G, normalized=True)
    pos = nx.circular_layout(G) if nt == "small_world" else nx.spring_layout(G, seed=42)
    return pos, max(bc, key=lambda n: bc[n])


def plot_topology_comparison(num_agents, network_seed, highlight_fca=True, show_degree_dist=True,
                             save=None, network_types=("centralized", "small_world", "random")):
    # num_agents / network_seed were `cfg.num_agents` / `cfg.network_seed` in the notebook.
    fig, axes = plt.subplots(2 if show_degree_dist else 1, 3,
                             figsize=(15, 9) if show_degree_dist else (18, 6), squeeze=False)
    nets = {}
    for col, nt in enumerate(network_types):
        G = _load_or_gen_network(nt, num_agents, network_seed)
        nets[nt] = G
        pos, hub = _topo_layout(G, nt)
        node_colors = (["#c0392b" if n == hub else TOPO_COLORS[nt] for n in G.nodes()]
                       if highlight_fca else TOPO_COLORS[nt])
        node_sizes = [90 + 45 * G.degree(n) for n in G.nodes()]
        nx.draw(G, pos, ax=axes[0][col], node_size=node_sizes, node_color=node_colors,
                edge_color="#cccccc", width=0.8, with_labels=False)
        if highlight_fca:
            nx.draw_networkx_labels(G, pos, labels={hub: "FCA"}, ax=axes[0][col],
                                    font_size=6, font_color="white", font_weight="bold")
        axes[0][col].set_title(f"{TOPO_TITLES[nt]} (N={num_agents})\n"
                               f"E={G.number_of_edges()}, density={nx.density(G):.3f}", fontsize=11)
        if show_degree_dist:
            degs = [d for _, d in G.degree()]
            axes[1][col].bar(*np.unique(degs, return_counts=True),
                             color=TOPO_COLORS[nt], edgecolor="white")
            axes[1][col].set_xlabel("Degree"); axes[1][col].set_ylabel("Count")
            axes[1][col].set_title(f"Degree dist | mean={np.mean(degs):.1f} "
                                   f"std={np.std(degs):.2f} max={max(degs)}")
    plt.suptitle(f"Network Type Comparison (N={num_agents}, density-matched)", fontsize=13)
    plt.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    _show(fig)
    return nets


def plot_sampling_distribution(num_agents=30, seeds=(42, 123, 777), save=None):
    """ESS pool vs the citizen concern draw the model actually uses.

    The draw depends only on (seed, node) -- not on topology or intervention -- so the
    len(seeds) x num_agents rows below are exactly the distinct citizens the experiment
    sampled, reproduced without running the model. Under fca/denial 1-2 of these nodes
    are replaced by a stubborn actor. Error bars span the per-seed min-max, the same
    convention as the drift-curve bands.
    """
    levels = [1, 2, 3, 4, 5]
    pool_c = np.array([int(r["climate_concern"]) for r in _load_ess_pool()], dtype=float)
    pool_pct = np.array([(pool_c == lv).mean() * 100 for lv in levels])

    per_seed, drawn = [], []
    for s in seeds:
        rows = sample_citizen_rows(s, range(num_agents))
        c = np.array([int(r["climate_concern"]) for r in rows.values()], dtype=float)
        drawn.append(c)
        per_seed.append([(c == lv).mean() * 100 for lv in levels])
    per_seed = np.asarray(per_seed)
    samp_pct = per_seed.mean(axis=0)          # equal n per seed -> pooled pct == mean of per-seed pct
    err = [samp_pct - per_seed.min(axis=0), per_seed.max(axis=0) - samp_pct]
    all_drawn = np.concatenate(drawn)

    x, w = np.arange(len(levels)), 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, pool_pct, w, color="#2166ac", edgecolor="white",
           label=f"ESS population (N={len(pool_c):,}, mean {pool_c.mean():.2f})")
    ax.bar(x + w / 2, samp_pct, w, color="#e08214", edgecolor="white", yerr=err,
           capsize=4, ecolor="#4d4d4d",
           label=f"Sampled agents (n={all_drawn.size}, mean {all_drawn.mean():.2f})")
    for xi, v in zip(x - w / 2, pool_pct):
        ax.text(xi, v + 0.7, f"{v:.0f}%", ha="center", fontsize=8)
    for xi, v, e in zip(x + w / 2, samp_pct, err[1]):
        ax.text(xi, v + e + 0.7, f"{v:.0f}%", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(lv) for lv in levels])
    ax.set_xlabel("Climate concern (wrclmch)")
    ax.set_ylabel("Share of agents (%)")
    ax.set_title("Agents are drawn uniformly at random from the ESS pool\n"
                 f"seeds {', '.join(map(str, seeds))} x {num_agents} citizens; "
                 "error bars = per-seed min-max", fontsize=11)
    ax.text(-0.02, -0.13, "Not worried", transform=ax.transAxes, fontsize=8, color="#555555")
    ax.text(0.94, -0.13, "Very worried", transform=ax.transAxes, fontsize=8, color="#555555")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    plt.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    _show(fig)
    return {"pool_mean": float(pool_c.mean()), "sampled_mean": float(all_drawn.mean()),
            "n_sampled": int(all_drawn.size)}


def interaction_diary(agents_data, max_steps=None, text_width=120):
    """
    Human-readable step-by-step interaction log.
    For each step and each agent shows:
      contacts  — who they reached out to (network neighbours)
      heard     — what each contact said (their opinion at start of step)
      said      — the agent's own opinion at start of step
      now says  — updated opinion after LLM processing
      reason    — LLM rationale for the concern change
      concern   — before → after delta
    """
    def trunc(s, n):
        s = str(s or "")
        return s[:n] + "…" if len(s) > n else s

    stub_ids    = {k for k, v in agents_data.items() if v.get("camp") in ("fca", "denial")}
    role_label  = {k: v["camp"].upper() + "_Agent" for k, v in agents_data.items() if v.get("camp") in ("fca", "denial")}
    citizen_ids = sorted([k for k in agents_data if k not in stub_ids], key=int)

    n_steps = min(len(v["contact_ids"]) for v in agents_data.values())
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)

    # ── Contact-rate statistics ───────────────────────────────────────────────
    print("Contact-rate summary (how often each agent reached someone):")
    print(f"  {'Agent':<15}  total_contacts  empty_steps")
    print(f"  {'-'*56}")
    for aid in citizen_ids + list(stub_ids):
        info  = agents_data[aid]
        cids  = info["contact_ids"]
        total = sum(len(c) for c in cids)
        empty = sum(1 for c in cids if not c)
        tag   = role_label.get(aid, f"Agent_{aid}")
        print(f"  {tag:<15}  {total:5d}           {empty}/{len(cids)}")

    # ── Step-by-step diary ────────────────────────────────────────────────────
    for step in range(n_steps):
        print(f"\n{'═'*72}")
        print(f"  STEP {step + 1}")
        print(f"{'═'*72}")

        for aid in citizen_ids + list(stub_ids):
            info     = agents_data[aid]
            is_stub  = aid in stub_ids
            concerns  = info["concerns"]
            opinions = info["opinions"]
            contacts = (info["contact_ids"][step]
                        if step < len(info["contact_ids"]) else [])

            # Concern delta line
            b0 = concerns[step]     if step     < len(concerns) else None
            b1 = concerns[step + 1] if step + 1 < len(concerns) else None
            if is_stub:
                b_line = "concern: FROZEN (stubborn actor, never updates)"
            elif b0 is not None and b1 is not None:
                d   = (b1 or 0) - (b0 or 0)
                sym = "▲" if d > 0.05 else ("▼" if d < -0.05 else "—")
                b_line = f"concern: {b0:.2f} → {b1:.2f}  {sym} {d:+.2f}"
            else:
                b_line = "concern: N/A"

            tag = role_label.get(aid, f"Agent_{aid}")
            print(f"\n  {tag:<15}  {b_line}")

            if not contacts:
                # No contacts this round (agent had no network neighbours,
                # no neighbour selected this step by random draw.
                # or max_interactions limit applied)
                print("    contacts : (none)")

                continue

            # Contact labels
            clabels = [role_label.get(str(c), f"Agent_{c}")
                       for c in contacts]
            print(f"    contacts : {', '.join(clabels)}")

            # What this agent said = opinion[step] (before this step's update)
            said = opinions[step] if step < len(opinions) else ""
            print(f"    said     : \"{trunc(said, text_width)}\"")

            # What agent HEARD from each contact = contact's opinion[step]
            print("    heard    :")
            for c in contacts:
                cinfo  = agents_data.get(str(c), {})
                cop    = cinfo.get("opinions", [])
                ctxt   = cop[step] if step < len(cop) else "(no opinion)"
                clabel = role_label.get(str(c), f"Agent_{c}")
                print(f"      {clabel}: \"{trunc(ctxt, 90)}\"")

            if not is_stub:
                op_after  = opinions[step + 1] if step + 1 < len(opinions) else ""
                rationale = (info.get("rationales", info.get("reasonings", []))[step + 1]
                             if step + 1 < len(info.get("rationales", info.get("reasonings", []))) else "")
                print(f"    now says : \"{trunc(op_after, text_width)}\"")
                if rationale:
                    print(f"    reason   : \"{trunc(rationale, text_width)}\"")
