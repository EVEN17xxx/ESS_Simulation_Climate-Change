"""Network generation (centralized/small-world/random), stats, homophily rewiring."""
import os
import json
import random
import numpy as np
import networkx as nx

def generate_centralized_network(num_agents: int, target_edges: int, seed: int = None,
                                 n_hubs: int = 3) -> nx.Graph:
    rng = random.Random(seed)
    n_hubs = max(2, min(n_hubs, num_agents - 1))
    G = nx.Graph()
    G.add_nodes_from(range(num_agents))
    hubs   = list(range(n_hubs))
    periph = list(range(n_hubs, num_agents))

    # 1) hubs form an interconnected core (clique)
    for i in range(n_hubs):
        for j in range(i + 1, n_hubs):
            G.add_edge(hubs[i], hubs[j])

    # 2) every peripheral attaches to at least one hub (round-robin -> connected & balanced)
    for idx, p in enumerate(periph):
        G.add_edge(p, hubs[idx % n_hubs])

    # 3) grow hub degree: extra peripheral->hub links (spread over the other hubs) toward target
    extra = [(p, hubs[(i + h) % n_hubs]) for i, p in enumerate(periph) for h in range(1, n_hubs)]
    rng.shuffle(extra)
    for p, hb in extra:
        if G.number_of_edges() >= target_edges:
            break
        if not G.has_edge(p, hb):
            G.add_edge(p, hb)

    # 4) if still short, add sparse peripheral-peripheral edges to reach target exactly
    if G.number_of_edges() < target_edges:
        pp = [(a, b) for i, a in enumerate(periph) for b in periph[i + 1:]]
        rng.shuffle(pp)
        for a, b in pp:
            if G.number_of_edges() >= target_edges:
                break
            if not G.has_edge(a, b):
                G.add_edge(a, b)

    # 5) if overshot (very small target), trim non-core edges without disconnecting
    while G.number_of_edges() > target_edges:
        trimmed = False
        for a, b in list(G.edges()):
            if a in hubs and b in hubs:
                continue
            G.remove_edge(a, b)
            if not nx.is_connected(G):
                G.add_edge(a, b)
            else:
                trimmed = True
                if G.number_of_edges() <= target_edges:
                    break
        if not trimmed:
            break

    return G


def generate_small_world_network(num_agents: int, k: int = 4, p: float = 0.1,
                                 seed: int = None) -> nx.Graph:
    # Watts-Strogatz (1998 Nature): k nearest-neighbour ring rewired with prob p.
    k = min(k, num_agents - 1)
    if k % 2 != 0:
        k -= 1
    k = max(k, 2)
    return nx.watts_strogatz_graph(num_agents, k, p, seed=seed)


def generate_random_network(num_agents: int, target_edges: int,
                            seed: int = None) -> nx.Graph:
    # Erdos-Renyi G(N, p), density-matched to small_world.
    max_possible = num_agents * (num_agents - 1) // 2
    p = target_edges / max_possible

    for attempt in range(5):
        G = nx.erdos_renyi_graph(num_agents, min(p, 1.0), seed=seed)
        if nx.is_connected(G):
            break
        p *= 1.2
        seed = None
    else:
        G = nx.erdos_renyi_graph(num_agents, min(p, 1.0), seed=seed)
        for u, v in nx.minimum_spanning_tree(nx.complete_graph(num_agents)).edges():
            G.add_edge(u, v)

    # Trim excess edges
    rng = np.random.default_rng(seed)
    edges = list(G.edges())
    rng.shuffle(edges)
    for e in edges:
        if G.number_of_edges() <= target_edges:
            break
        G.remove_edge(*e)
        if not nx.is_connected(G):
            G.add_edge(*e)

    # Add missing edges
    if G.number_of_edges() < target_edges:
        non_edges = list(nx.non_edges(G))
        rng.shuffle(non_edges)
        for e in non_edges:
            if G.number_of_edges() >= target_edges:
                break
            G.add_edge(*e)

    return G


def generate_network(network_type: str, num_agents: int, **kwargs) -> nx.Graph:
    seed = kwargs.get("seed")
    k = kwargs.get("k", 4)
    target_edges = num_agents * k // 2

    if network_type == "centralized":
        return generate_centralized_network(num_agents, target_edges, seed)
    elif network_type == "small_world":
        return generate_small_world_network(num_agents, k, kwargs.get("p", 0.1), seed)
    elif network_type == "random":
        return generate_random_network(num_agents, target_edges, seed)
    else:
        raise ValueError(f"Unknown network_type '{network_type}'. "
                         f"Choose from: centralized, small_world, random")


def get_network_statistics(G) -> dict:
    """Topology-only stats. Assortativity/homophily live in World._camp_network_stats(),
    which is the single place that decides how a degenerate (NaN) coefficient is reported."""
    degrees = [d for _, d in G.degree()]
    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "clustering_coefficient": nx.average_clustering(G),
        "is_connected": nx.is_connected(G),
        "degree_mean": float(np.mean(degrees)),
        "degree_std":  float(np.std(degrees)),
        "degree_max":  int(max(degrees)),
        "degree_min":  int(min(degrees)),
    }
    if stats["is_connected"]:
        try:
            stats["avg_path_length"] = nx.average_shortest_path_length(G)
            stats["diameter"] = nx.diameter(G)
        except Exception:
            pass
    return stats


def save_network_structure(G, file_path: str):
    data = {"nodes": list(G.nodes()), "edges": [list(e) for e in G.edges()]}
    dir_part = os.path.dirname(file_path)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_network_structure(file_path: str) -> nx.Graph:
    with open(file_path, "r") as f:
        data = json.load(f)
    G = nx.Graph()
    G.add_nodes_from(data["nodes"])
    G.add_edges_from([tuple(e) for e in data["edges"]])
    return G



def homophilous_rewire(G, concern_of, target_r, network_seed, tol=0.02, max_attempts_factor=3000):
    """Degree-preserving double-edge swaps (Maslov & Sneppen 2002) that raise the NUMERIC
    assortativity of node attribute 'concern' toward target_r, rewiring ONLY citizen-citizen
    edges (both endpoints present in concern_of). Preserves every node's degree, the total
    edge count, and connectivity (a swap that disconnects G is rolled back). target_r <= 0
    -> no-op. `concern_of` maps each citizen node -> its numeric concern (1-5); stubborn nodes
    are absent from it and never rewired.

    Acceptance (concern-distance driven, NOT categorical camp): a swap
    (a-b, c-d) -> (a-d, c-b) is accepted only if it REDUCES the total concern distance of the
    rewired endpoints, i.e. |c_a-c_d| + |c_c-c_b| < |c_a-c_b| + |c_c-c_d| (pull similar-concern
    nodes together). Convergence uses nx.numeric_assortativity_coefficient(G, "concern")."""
    if target_r is None or target_r <= 0:
        return G
    nx.set_node_attributes(G, concern_of, "concern")   # ensure numeric attr is present
    rng = random.Random(network_seed + 999)
    citizens = [n for n in G.nodes() if n in concern_of]

    def current_r():
        try:
            r = nx.numeric_assortativity_coefficient(G.subgraph(citizens), "concern")
            return 0.0 if r != r else r                # NaN guard (degenerate: all-equal concern)
        except Exception:
            return 0.0

    def dist(x, y):
        return abs(concern_of[x] - concern_of[y])

    edges = [(a, b) for a, b in G.edges() if a in concern_of and b in concern_of]
    r_now = current_r()
    for _ in range(max_attempts_factor * max(1, len(edges))):
        if r_now >= target_r - tol or len(edges) < 2:
            break
        (a, b), (c, d) = rng.sample(edges, 2)
        if len({a, b, c, d}) < 4:
            continue
        if G.has_edge(a, d) or G.has_edge(c, b):
            continue
        # accept only swaps that reduce total concern distance of the rewired endpoints
        if dist(a, d) + dist(c, b) >= dist(a, b) + dist(c, d):
            continue
        G.remove_edge(a, b); G.remove_edge(c, d)
        G.add_edge(a, d);    G.add_edge(c, b)
        if not nx.is_connected(G):                     # keep the interaction graph connected
            G.remove_edge(a, d); G.remove_edge(c, b)
            G.add_edge(a, b);    G.add_edge(c, d)
            continue
        edges = [(x, y) for x, y in G.edges() if x in concern_of and y in concern_of]
        r_now = current_r()
    return G
