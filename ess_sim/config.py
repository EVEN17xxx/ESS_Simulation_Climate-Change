"""ExperimentConfig dataclass and the 3x3xseeds condition grid."""
from dataclasses import dataclass
from typing import Literal

@dataclass
class ExperimentConfig:
    # core
    num_agents: int = 30
    network_type: Literal["centralized", "small_world", "random"] = "centralized"
    gpt_model: str = "gpt-4o-mini"
    temperature: float = 0.7  # our choice: balances diversity vs coherence (API default is 1.0)
    step_count: int = 20

    # initial-concern camp labels (full unimodal sampling; wrclmch==3 kept, labelled 'mid')
    concern_high_min: int = 4   # >=4 -> high_concern
    concern_low_max: int = 2    # <=2 -> low_concern

    # network homophily as a target NUMERIC concern-assortativity r, injected at build time via
    # degree-preserving rewiring. 0 = no rewire (off in all main runs).
    target_r: float = 0.0

    # stubborn agents (3-level intervention: none / FCA / denial)
    fca_enabled: bool = False
    denial_enabled: bool = False

    # blank control: isolation mode -- agents contact no one (pure LLM drift baseline)
    isolation: bool = False

    # interaction & memory
    max_interactions: int = 1  # Deffuant pairwise convention (Deffuant et al. 2000; Flache et al. 2017)
    memory_window: int = 3     # raw sliding window, no summarization (Cowan 2001 ~3-4 chunks)
    max_concurrent_llm_calls: int = 5   # conservative default; caps parallel LLM calls per step

    seed: int = 0
    network_seed: int | None = None
    exp_dir: str = "./runs"
    exp_id: str = ""

    def __post_init__(self):
        if self.network_seed is None:
            self.network_seed = self.seed
        if not self.exp_id:
            tags = "".join([
                "_fca" if self.fca_enabled else "",
                "_denial" if self.denial_enabled else "",
                f"_r{self.target_r:g}" if self.target_r else "",
                "_iso" if self.isolation else "",
                f"_t{self.temperature:g}" if self.temperature != 0.7 else "",
                f"_{self.gpt_model}" if self.gpt_model != "gpt-4o-mini" else "",
                f"_steps{self.step_count}" if self.step_count != 20 else "",
                f"_mem{self.memory_window}" if self.memory_window != 3 else "",
                f"_k{self.max_interactions}" if self.max_interactions != 1 else "",
                f"_nseed{self.network_seed}" if self.network_seed != self.seed else "",
            ])
            self.exp_id = (
                f"net_{self.network_type}{tags}_"
                f"n{self.num_agents}_"
                f"seed_{self.seed}"
            )


def generate_ess_conditions(num_agents=30, step_count=20, seeds=None, target_r=0.0) -> list:
    """Experiment grid: 3 network topologies x 3-LEVEL intervention x len(seeds) seeds.

    Intervention is 3-level {none / FCA-only / denial-only}; the FCA x denial cross
    (both stubborn at once) is intentionally EXCLUDED and left as future work, matching the
    central claim. full = 3 x 3 x 3 = 27 runs; 1 seed = 9."""
    
    if seeds is None:
        seeds = [42, 123, 777]
    network_types = ["centralized", "small_world", "random"]
    configs = []
    for nt in network_types:
        for s in seeds:
            for fca, denial in [(False, False), (True, False), (False, True)]:   # none / FCA / denial
                configs.append(ExperimentConfig(
                    network_type=nt, seed=s, network_seed=s,
                    fca_enabled=fca, denial_enabled=denial, target_r=target_r,
                    num_agents=num_agents, step_count=step_count))
    return configs