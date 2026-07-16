"""World: mesa model orchestrating ESS-sampled LLM agents over a social network."""
import os
import json
import random
import asyncio
import hashlib
from datetime import datetime, timedelta

import pandas as pd
import networkx as nx
import mesa
from mesa.datacollection import DataCollector
from tqdm.auto import tqdm

from .agent import ClimateAgent
from .config import ExperimentConfig
from .llm_client import (
    LLM_STATS,
    UpdateOpinionResponse,
    clamp_concern,
    reset_llm_semaphore,
    set_max_concurrent_llm_calls,
    async_get_completion_from_messages_structured,
)
from .prompt import (
    build_fca_system_prompt,
    build_denial_system_prompt,
    build_ess_persona_prompt,
    init_interpretation_prompt,
    update_prompt,
    fca_response_prompt,
    denial_response_prompt,
)
from .article import fetch_and_cache
from .ess_pool import _load_ess_pool, sample_citizen_rows
from .network import (
    generate_network,
    get_network_statistics,
    save_network_structure,
    load_network_structure,
    homophilous_rewire,
)
from .metrics import (
    metric_dispersion,
    metric_mean_concern,
    metric_bimodality,
)

CITIZEN_CAMPS = ("high_concern", "low_concern", "mid")   # 'mid' is a camp too; stubborn actors are not


class _SimpleScheduler:
    def __init__(self):
        self._agents: list = []
        self.steps: int = 0
        self.time:  int = 0
    def add(self, agent) -> None:
        self._agents.append(agent)
    @property
    def agents(self) -> list:
        return self._agents


class World(mesa.Model):

    def __init__(self, config: ExperimentConfig, load_network: bool = True):
        super().__init__()
        self.config           = config
        self.event_log        = []
        self._init_fallbacks  = 0   # agents whose initial interpretation the LLM failed to produce
        self.num_agents       = config.num_agents
        self.network_type     = config.network_type
        self.gpt_model        = config.gpt_model
        self.temp             = config.temperature
        self.max_interactions = config.max_interactions
        self._contact_rng = random.Random(config.network_seed)   # reproducible contact selection
        set_max_concurrent_llm_calls(config.max_concurrent_llm_calls)

        self.run_dir = os.path.join(config.exp_dir, config.exp_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # Topic is fixed for the concern-based redesign (config.topic removed).
        self.topic_key = "climate_europe"
        print(f"[INFO] Fetching article for topic: {self.topic_key}")
        topic_data = fetch_and_cache(self.topic_key, cache_dir="./data")
        self.topic_question = topic_data["question"]
        self.article_text   = topic_data["article"]

        network_file = (
            f"./data/{config.network_type}_network_"
            f"num_agents_{config.num_agents}_seed_{config.network_seed}.json"
        )
        if load_network and os.path.exists(network_file):
            print(f"[INFO] Loading network from {network_file}")
            self.G = load_network_structure(network_file)
        else:
            print(f"[INFO] Generating {config.network_type} network with {config.num_agents} nodes")
            network_kwargs = {"seed": config.network_seed}
            self.G = generate_network(config.network_type, config.num_agents, **network_kwargs)
            save_network_structure(self.G, network_file)

        self.schedule = _SimpleScheduler()
        self.backgrounds = {}

        # Places stubborn actors, assigns citizen camps, then homophily-rewires
        # citizen-citizen edges toward target_r. Mutates self.G in place.
        self._init_agent_params()

        # Grid + network stats on the FINAL (rewired) graph so centrality, camp
        # assortativity r and E-I reflect the injected homophily.
        self.grid = mesa.space.NetworkGrid(self.G)
        net_stats = get_network_statistics(self.G)
        net_stats.update(self._camp_network_stats())
        with open(os.path.join(self.run_dir, "network_statistics.json"), "w", encoding="utf-8") as f:
            json.dump(net_stats, f, indent=2, default=str)

        bg_path = os.path.join(self.run_dir, "agents_backgrounds.json")
        fp_path = os.path.join(self.run_dir, "agents_backgrounds.fingerprint.txt")
        fingerprint = self._bg_fingerprint()
        stale = self._bg_cache_stale(bg_path, fp_path, fingerprint)
        if stale is None:
            print(f"[CACHE] Loading agent backgrounds from {bg_path}")
            with open(bg_path, encoding="utf-8") as _f:
                _cached = json.load(_f)
            interpretations = {
                int(k): {
                    "opinion":    v["initial_opinion"],
                    "rationale":  v.get("initial_rationale", v.get("initial_reasoning", "")),
                    "llm_concern": v["initial_concern"],
                }
                for k, v in _cached.items()
            }
        else:
            print(f"[INFO] {stale} -> creating {config.num_agents} agents (parallel LLM calls)...")
            reset_llm_semaphore()
            self._init_fallbacks = 0
            interpretations = asyncio.run(self._async_generate_all_interpretations())

        for node in sorted(self.G.nodes()):
            params = self._agent_init_params[node]
            interp = interpretations[node]
            agent = ClimateAgent(
                model=self, unique_id=node, name=params["name"],
                camp=params["camp"], initial_concern=interp["llm_concern"],
                topic=self.topic_question,
                gpt_model=self.gpt_model, temp=self.temp,
                initial_opinion=interp["opinion"], initial_rationale=interp["rationale"],
                system_prompt=params["system_prompt"],
                memory_window=config.memory_window,
                ess_row=params["ess_row"],
                is_stubborn=params.get("is_stubborn", False),
            )
            self.schedule.add(agent)
            self.grid.place_agent(agent, node)
            self.backgrounds[str(node)] = {
                "name": params["name"],
                "initial_concern": interp["llm_concern"],
                "concern_hint": params["initial_concern"],
                "node_degree": self.G.degree(node),
                "initial_opinion": interp["opinion"],
                "initial_rationale": interp["rationale"],
                "demographic_profile": params["demographic_profile"],
            }

        with open(bg_path, "w", encoding="utf-8") as f:
            json.dump(self.backgrounds, f, indent=2, ensure_ascii=False)
        if self._init_fallbacks:
            # Some agents start with no LLM opinion at all. Withhold the fingerprint so this
            # degraded initialization is never silently reused as canonical.
            print(f"[WARN] {self._init_fallbacks}/{self.num_agents} initial interpretations fell "
                  f"back to the sampled ESS value (LLM returned nothing). Fingerprint NOT written; "
                  f"these backgrounds will be regenerated on the next run.")
        else:
            with open(fp_path, "w", encoding="utf-8") as f:
                f.write(fingerprint)

        self.datacollector = DataCollector(
            model_reporters={
                "MeanConcern":      self.compute_mean_concern,
                "Dispersion":       self.compute_dispersion,
                "Bimodality":       self.compute_bimodality,
                "CrossCampRate":    self.compute_cross_camp_rate,
            },
            agent_reporters={"Concern": "concern", "Camp": "camp"},
        )
        self.current_date = datetime(2024, 1, 1)
        self.datacollector.collect(self)

    def _bg_fingerprint(self) -> str:
        # Identity of everything that determines the initial interpretations. A cached
        # agents_backgrounds.json is only reusable if this still matches: exp_id alone does
        # NOT change when the prompts, the article revision or the model change.
        h = hashlib.sha256()
        for part in (self.article_text, self.topic_question, init_interpretation_prompt,
                     self.gpt_model, str(self.temp), str(self.config.seed),
                     str(self.config.num_agents)):
            h.update(part.encode("utf-8"))
        for node in sorted(self._agent_init_params):   # personas: ESS draw x prompt template
            h.update(self._agent_init_params[node]["system_prompt"].encode("utf-8"))
        return h.hexdigest()[:16]

    def _bg_cache_stale(self, bg_path, fp_path, fingerprint) -> "str | None":
        # None -> cache is reusable; otherwise the human-readable reason it is not.
        if not os.path.exists(bg_path):
            return "No cached agent backgrounds"
        if not os.path.exists(fp_path):
            return "Cache predates fingerprinting (prompts/article may have changed since)"
        with open(fp_path, encoding="utf-8") as f:
            cached_fp = f.read().strip()
        if cached_fp != fingerprint:
            return f"Cache fingerprint {cached_fp} != current {fingerprint} (inputs changed)"
        with open(bg_path, encoding="utf-8") as f:
            n_cached = len(json.load(f))
        if n_cached != len(self.G.nodes()):
            return f"Cache has {n_cached} entries but network has {len(self.G.nodes())} nodes"
        return None

    def _init_agent_params(self):
        cfg = self.config
        pool = _load_ess_pool()

        # 1) Stubborn actors at the most central nodes (star -> degree; WS/ER -> betweenness).
        #    top-1 -> FCA, top-2 -> denial. Stubborn nodes get NO camp node attribute and are
        #    excluded from homophily rewiring / assortativity.
        stub_nodes = {}
        if cfg.fca_enabled or cfg.denial_enabled:
            if cfg.network_type == "centralized":
                ranked = sorted(self.G.nodes(), key=lambda n: self.G.degree(n), reverse=True)
            else:
                bc = nx.betweenness_centrality(self.G, normalized=True)
                ranked = sorted(bc, key=lambda n: bc[n], reverse=True)
            pending = (["fca"] if cfg.fca_enabled else []) + (["denial"] if cfg.denial_enabled else [])
            for role, node in zip(pending, ranked):
                stub_nodes[node] = role

        # 2) Citizens: sample each respondent, then assign a frozen camp LABEL.
        citizen_nodes = [n for n in sorted(self.G.nodes()) if n not in stub_nodes]
        # Sample each citizen from the FULL ESS pool (real unimodal distribution, wrclmch=3 kept);
        # camp is a post-hoc 3-tier LABEL derived from the sampled initial concern.
        citizen_rows = sample_citizen_rows(cfg.seed, citizen_nodes, pool)
        camp_of = {}
        for node in citizen_nodes:
            c0 = int(citizen_rows[node]["climate_concern"])
            camp_of[node] = ("high_concern" if c0 >= cfg.concern_high_min
                             else "low_concern" if c0 <= cfg.concern_low_max
                             else "mid")

        # camp label as a node attribute (post-hoc grouping + AUXILIARY categorical assortativity).
        nx.set_node_attributes(self.G, camp_of, "camp")

        # 3) Inject homophily by rewiring citizen-citizen edges toward target_r on NUMERIC concern
        #    (concern-distance driven, not categorical camp).
        concern_of = {node: int(citizen_rows[node]["climate_concern"]) for node in citizen_nodes}
        nx.set_node_attributes(self.G, concern_of, "concern")
        self.G = homophilous_rewire(self.G, concern_of, cfg.target_r, cfg.network_seed)

        self._camp_of    = camp_of
        self._concern_of = concern_of
        self._stub_nodes = stub_nodes

        # 4) Build per-node agent parameters.
        self._agent_init_params = {}
        for node, role in stub_nodes.items():
            self._agent_init_params[node] = {
                "name":                f"{role.upper()}_Agent",
                "camp":                role,
                "ess_row":             {},
                "system_prompt":       build_fca_system_prompt() if role == "fca"
                                       else build_denial_system_prompt(),
                "initial_concern":     3,          # dummy; excluded from metrics (is_stubborn)
                "demographic_profile": {"agent_type": role},
                "is_stubborn":         True,
            }
        for node in citizen_nodes:
            row = citizen_rows[node]
            system_prompt, demo_profile = build_ess_persona_prompt(row)
            self._agent_init_params[node] = {
                "name":                f"Agent_{node}",
                "camp":                camp_of[node],
                "ess_row":             row,
                "system_prompt":       system_prompt,
                "initial_concern":     int(row["climate_concern"]),
                "demographic_profile": demo_profile,
                "is_stubborn":         False,
            }

    def _camp_network_stats(self) -> dict:
        # MAIN homophily: NUMERIC assortativity of concern over the citizen subgraph
        # (stubborn excluded), post-rewire. Categorical camp assortativity + E-I kept as auxiliary.
        camp_of = self._camp_of
        citizens = [n for n in self.G.nodes() if n in camp_of]
        try:
            r_concern = nx.numeric_assortativity_coefficient(self.G.subgraph(citizens), "concern")
            if r_concern != r_concern:
                r_concern = None
        except Exception:
            r_concern = None
        try:
            r = nx.attribute_assortativity_coefficient(self.G.subgraph(citizens), "camp")
            if r != r:
                r = None
        except Exception:
            r = None
        internal = external = 0
        for a, b in self.G.edges():
            if a in camp_of and b in camp_of:
                if camp_of[a] == camp_of[b]:
                    internal += 1
                else:
                    external += 1
        stub_positions = {}
        if self._stub_nodes:
            bc = nx.betweenness_centrality(self.G, normalized=True)
            for node, role in self._stub_nodes.items():
                stub_positions[str(node)] = {
                    "role": role,
                    "degree": self.G.degree(node),
                    "betweenness": bc.get(node),
                }
        return {
            "target_r": self.config.target_r,
            "concern_assortativity_r": r_concern,        # MAIN homophily metric
            "camp_assortativity_r": r,                   # auxiliary
            "n_high_concern": sum(1 for c in camp_of.values() if c == "high_concern"),
            "n_low_concern": sum(1 for c in camp_of.values() if c == "low_concern"),
            "n_mid": sum(1 for c in camp_of.values() if c == "mid"),
            "citizen_internal_edges": internal,
            "citizen_external_edges": external,
            "stub_positions": stub_positions,
        }

    async def _async_generate_all_interpretations(self) -> dict:
        nodes = sorted(self.G.nodes())
        tasks = [self._async_generate_initial_interpretation(
                     self._agent_init_params[n]["system_prompt"],
                     self._agent_init_params[n]["initial_concern"])
                 for n in nodes]
        responses = await asyncio.gather(*tasks)
        return {n: {"opinion": o, "rationale": r, "llm_concern": b}
                for n, (o, r, b) in zip(nodes, responses)}

    async def _async_generate_initial_interpretation(self, system_prompt, initial_concern):
        user_msg = init_interpretation_prompt.format(
            article_text=self.article_text,
            topic_question=self.topic_question,
        )
        response = await async_get_completion_from_messages_structured(
            system_messages=system_prompt, messages=user_msg,
            model=self.gpt_model, temperature=self.temp, response_type=UpdateOpinionResponse,
        )
        if response:
            llm_concern = clamp_concern(response.concern)
            return response.opinion, response.rationale, llm_concern
        # LLM exhausted its retries: fall back to the sampled ESS value with no opinion text.
        # Counted by the caller -- a run initialised from fallbacks must not be cached as canonical.
        self._init_fallbacks += 1
        return "", "Initial interpretation (LLM fallback).", initial_concern

    def decide_agent_interactions(self):
        # Deffuant pairwise contact: each agent randomly selects max_interactions
        # neighbour(s) per step (Deffuant et al. 2000; Flache et al. 2017).
        # Selective receptivity is handled by the LLM persona.
        if self.config.isolation:                    # blank control: contact no one
            for agent in self.schedule.agents:
                agent.agent_interaction = []
            return
        for agent in self.schedule.agents:
            neighbors = self.grid.get_neighbors(agent.pos)
            if not neighbors:
                agent.agent_interaction = []
                continue
            # Sort before shuffling: a saved/reloaded graph returns neighbours in a different
            # adjacency order, and Fisher-Yates on a reordered list picks a different contact
            # from the same RNG state -- i.e. cached vs freshly-built networks would diverge.
            neighbor_list = sorted(neighbors, key=lambda a: a.unique_id)
            if self.max_interactions > 0 and len(neighbor_list) > self.max_interactions:
                self._contact_rng.shuffle(neighbor_list)
                neighbor_list = neighbor_list[:self.max_interactions]
            agent.agent_interaction = neighbor_list


    def _agent_state_dicts(self):
        concerns = {a.unique_id: a.concern for a in self.schedule.agents}
        camps    = {a.unique_id: a.camp for a in self.schedule.agents}
        stubborn = {a.unique_id: a.is_stubborn for a in self.schedule.agents}
        return concerns, camps, stubborn

    def compute_dispersion(self) -> float:
        concerns, _camps, stub = self._agent_state_dicts()
        return metric_dispersion(concerns, stub)

    def compute_mean_concern(self) -> float:
        concerns, _camps, stub = self._agent_state_dicts()
        return metric_mean_concern(concerns, stub)

    def compute_bimodality(self) -> float:
        concerns, _camps, stub = self._agent_state_dicts()
        return metric_bimodality(concerns, stub)

    def compute_cross_camp_rate(self, step_idx: int = -1) -> float:
        # Behavioural mixing: share of this step's citizen contacts that cross camps.
        # Reads frozen contact_camps; stubborn listeners excluded, stubborn speakers ignored.
        # All three citizen camps count on BOTH sides -- 'mid' is the largest group, so
        # scoring it as a speaker while dropping it as a listener would bias the denominator.
        cross, total = 0, 0
        for agent in self.schedule.agents:
            if agent.is_stubborn or not agent.contact_camps:
                continue
            try:
                contacts = agent.contact_camps[step_idx]
            except IndexError:
                continue
            for cc in contacts:
                if cc not in CITIZEN_CAMPS:
                    continue
                total += 1
                if cc != agent.camp:
                    cross += 1
        return cross / total if total > 0 else 0.0

    def step(self):
        self.decide_agent_interactions()
        reset_llm_semaphore()
        # Advance the clock BEFORE the round runs, so event_log rows carry the same step
        # number as the model_overview/model_summary row for that round (they are joined on it).
        self.schedule.steps += 1
        self.schedule.time  += 1
        asyncio.run(self._async_step_core())
        self.datacollector.collect(self)
        self.current_date += timedelta(days=1)
        self.save_model_data()

    async def _async_step_core(self):
        agents = list(self.schedule.agents)
        await asyncio.gather(*(self._async_update_agent(a) for a in agents))

    async def _async_intervener_response(self, speaker, listener):
        # Reactive intervener: reads the listener's current opinion and generates a targeted
        # correction (FCA) or rebuttal (denial). Stance is fixed by the system prompt; only the
        # wording adapts. Falls back to the fixed stance if the LLM returns nothing.
        if speaker.camp == "fca":
            user_msg = fca_response_prompt.format(
                listener_opinion=listener.opinions[-1], article_text=self.article_text)
        else:
            user_msg = denial_response_prompt.format(listener_opinion=listener.opinions[-1])
        resp = await async_get_completion_from_messages_structured(
            system_messages=speaker.system_prompt, messages=user_msg,
            model=self.gpt_model, temperature=self.temp,
            response_type=UpdateOpinionResponse,
        )
        return resp.opinion if resp and resp.opinion else speaker.opinions[-1]

    async def _async_update_agent(self, agent):
        # Stubborn actors (FCA / denial) never change their own position: carry the fixed
        # stance forward (used only as a fallback; their actual per-listener replies are
        # generated reactively in _async_intervener_response), and log empty contacts.
        if agent.is_stubborn:
            agent.opinions.append(agent.opinions[-1])
            agent.concerns.append(agent.concerns[-1])
            agent.rationales.append(agent.rationales[-1])
            agent.contact_ids.append([])
            agent.contact_camps.append([])
            return

        contacts = agent.agent_interaction
        # Reactive interveners tailor their reply to this listener; citizens are read as-is.
        contact_texts = [await self._async_intervener_response(c, agent) if c.is_stubborn
                         else c.opinions[-1] for c in contacts]
        agent.contact_ids.append([c.unique_id for c in contacts])
        agent.contact_camps.append([c.camp for c in contacts])

        c_before      = agent.concerns[-1]
        memory_text   = "\n".join(f"- {m}" for m in agent.memory) or "(nothing yet)"
        contact_block = "\n".join(f'- "{t}"' for t in contact_texts) or "(No one contacted you this round.)"
        user_msg = update_prompt.format(
            my_opinion=agent.opinions[-1], my_concern=round(c_before),
            memory=memory_text, contact_opinions=contact_block,
        )

        resp = await async_get_completion_from_messages_structured(
            system_messages=agent.system_prompt, messages=user_msg,
            model=agent.gpt_model, temperature=agent.temp,
            response_type=UpdateOpinionResponse,
        )
        if resp:
            new_op, new_con, rat = resp.opinion, clamp_concern(resp.concern), resp.rationale
        else:
            new_op, new_con, rat = agent.opinions[-1], c_before, agent.rationales[-1]

        agent.opinion, agent.concern = new_op, new_con
        agent.opinions.append(new_op)
        agent.concerns.append(new_con)
        agent.rationales.append(rat)

        # Event log: with max_interactions=1 each row is exactly one listener-speaker
        # dyad, so delta is cleanly attributable to that speaker (attribution rule).
        for sp, sp_text in zip(contacts, contact_texts):
            self.event_log.append({
                "run_id": self.config.exp_id, "step": self.schedule.time,
                "listener_id": agent.unique_id, "listener_camp": agent.camp,
                "speaker_id": sp.unique_id, "speaker_camp": sp.camp,
                # None (not False) for stubborn speakers: they are outside the citizen-mixing
                # denominator, so .mean() must skip them to match CrossCampRate.
                "cross_camp": (agent.camp != sp.camp) if sp.camp in CITIZEN_CAMPS else None,
                "concern_before": c_before, "concern_after": new_con,
                "delta": new_con - c_before,
                "opinion_after": new_op, "rationale": rat,
                "speaker_opinion": sp_text,
            })

        agent.memory.extend(contact_texts)
        agent.memory = agent.memory[-agent.memory_window:]

    def run_model(self, step_count: "int | None" = None):
        if step_count is None:
            step_count = self.config.step_count
        overview_path = os.path.join(self.run_dir, "model_overview.json")
        if os.path.exists(overview_path):
            os.remove(overview_path)
        print(f"[INFO] Running simulation steps 1-{step_count}...")
        for _ in tqdm(range(step_count), desc="Simulation"):
            self.step()
        self.save_agents_data(os.path.join(self.run_dir, "agents_data.json"))
        self.save_camp_distribution()
        # Three tidy long-format tables: (1) event log, (2) agent-step panel, (3) model-step summary.
        pd.DataFrame(self.event_log).to_csv(
            os.path.join(self.run_dir, "event_log.csv"), index=False)
        ap = self.datacollector.get_agent_vars_dataframe().reset_index()
        ap.insert(0, "run_id", self.config.exp_id)
        ap.to_csv(os.path.join(self.run_dir, "agent_panel.csv"), index=False)
        ms = self.datacollector.get_model_vars_dataframe().reset_index().rename(
            columns={"index": "step"})
        ms.insert(0, "run_id", self.config.exp_id)
        ms.to_csv(os.path.join(self.run_dir, "model_summary.csv"), index=False)

    def save_agents_data(self, file_path: str):
        agents_data = {}
        for agent in self.schedule.agents:
            agents_data[str(agent.unique_id)] = {
                "agent_type":    "stubborn" if agent.is_stubborn else "citizen",
                "camp":          agent.camp,
                "is_stubborn":   agent.is_stubborn,
                "opinions":      agent.opinions,
                "concerns":      agent.concerns,
                "rationales":    agent.rationales,
                "memory":        agent.memory,
                "contact_ids":   agent.contact_ids,
                "contact_camps": agent.contact_camps,
            }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(agents_data, f, indent=2, ensure_ascii=False)

    def save_model_data(self):
        concerns, _camps, stub = self._agent_state_dicts()
        bm = metric_bimodality(concerns, stub)
        model_data = {
            "run_id":             self.config.exp_id,
            "step":               self.schedule.time,
            "date":               str(self.current_date),
            "mean_concern":       metric_mean_concern(concerns, stub),
            "dispersion":         metric_dispersion(concerns, stub),
            "cross_camp_rate":    self.compute_cross_camp_rate(),
            "bimodality":         (None if bm != bm else bm),
            # Non-zero means some agents were frozen by an LLM failure, not by choice:
            # that biases mean/dispersion toward no-change. Cumulative over the run.
            "llm_failures":       LLM_STATS["failures"],
        }
        file_path = os.path.join(self.run_dir, "model_overview.json")
        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(model_data, f)
            f.write("\n")

    def save_camp_distribution(self):
        camp_of = getattr(self, "_camp_of", {})
        dist = {
            "total_agents": self.num_agents,
            "n_high_concern": sum(1 for c in camp_of.values() if c == "high_concern"),
            "n_low_concern": sum(1 for c in camp_of.values() if c == "low_concern"),
            "n_mid": sum(1 for c in camp_of.values() if c == "mid"),
            "concern_high_min": self.config.concern_high_min,
            "concern_low_max":  self.config.concern_low_max,
            "camps": {str(a.unique_id): a.camp for a in self.schedule.agents},
        }
        path = os.path.join(self.run_dir, "camp_distribution.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dist, f, indent=2)

