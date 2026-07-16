# ESS-LLM: Climate-Concern Opinion Dynamics with LLM Agents

An agent-based simulation in which **LLM agents initialized from European Social Survey (ESS Round 10) respondents** interact on social networks and update a 1–5 climate-concern value. We study how network topology and a structural intervention (a fact-checker or a denier) shape the collective dynamics.

## What's in this repo

```
ess_sim/               # the simulation engine (import this)
  model.py             #   World: orchestrator -- builds the network, samples agents,
                       #   runs the step loop, drives the LLM, writes runs/<exp_id>/
  agent.py             #   ClimateAgent: one citizen's state only (no behaviour)
  llm_client.py        #   Azure OpenAI plumbing: async client, retry, concurrency, schema
  network.py           #   topology generation, stats, homophily rewiring
  ess_pool.py          #   ESS pool loader + the citizen draw + pool diagnostics
  prompt.py            #   all prompt templates (personas, FCA/denial, update)
  article.py           #   version-pinned Wikipedia stimulus fetch + cache
  metrics.py           #   the three dependent variables
  config.py            #   ExperimentConfig + the 3x3xseeds condition grid
  visualization.py     #   topology comparison, ESS-vs-sampled distribution, diary

run_full.py            # CLI: run the grid, resumable      <- batch entry point
run_ess_sim.ipynb      # interactive: one worked run + figures  <- exploratory entry point
requirements.txt
.env.example           # template for your Azure OpenAI credentials
LICENSE                # MIT
```


## Pipeline

  How one experiment condition goes from survey data to results:

  ```mermaid
  flowchart TD
      A[ESS Round 10<br/>respondent pool] -->|sample one respondent per node| B[30 LLM citizen agents<br/>age · gender · country · education · concern]
      C[Pick topology<br/>centralized / small-world / random] --> D[Build network<br/>N=30, ~60 edges, density-matched]
      B --> E[Place agents on the network]
      D --> E
      E --> F{Intervention?}
      F -->|FCA or denial| G[Add ONE stubborn intervener<br/>at the most-central node]
      F -->|none| H[LLM initialization]
      G --> H[LLM initialization<br/>each agent reads a pinned Wikipedia article<br/>and forms an initial concern 1-5]
      H --> I[Simulation loop — 20 steps]
      I --> J[Per-step metrics<br/>mean concern · variance · bimodality]
  
  ```

  **What happens in one simulation step** (repeated 20 times):

  ```mermaid
  flowchart LR
      S1[Each citizen picks<br/>1 random neighbour] --> S2{Is that neighbour<br/>an intervener?}
      S2 -->|yes| S3[Intervener reads the citizen's view<br/>and writes a tailored reply<br/>its stance never changes]
      S2 -->|no| S4[Read the neighbour's<br/>current opinion]
      S3 --> S5[Citizen re-evaluates and<br/>updates its concern via the LLM]
      S4 --> S5
      S5 --> S6[Update memory + write event log]
  ```


## Running it

Run from the repo root, so `import ess_sim` and `./data` resolve.

**The full grid** (3 topologies × 3 interventions × 3 seeds = 27 runs).  ≈2 hours and ≈$2 on `gpt-4o-mini`:

```bash
python run_full.py                 # all 27
python run_full.py --mode medium   # 1 seed x 3 topo x 3 interv = 9
python run_full.py --mode single --topology centralized --intervention denial --seed 42
```

**One condition, interactively** — network figures, a sample persona, the per-agent table
and a step-by-step interaction diary. The fastest way to see what the model does:
open `run_ess_sim.ipynb`.


## Setup

1. Python 3.11+ (developed on 3.13).
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and fill in your **Azure OpenAI** credentials (endpoint, API key, deployment name).

## Data

This is a code-only release: `data/` and `runs/` are not tracked.

- **ESS respondent pool** (`data/ess_respondent_pool.csv`): derived from the **European Social Survey Round 10**. ESS microdata redistribution can be restricted, so it is omitted; obtain ESS10 from the ESS Data Portal and generate the pool yourself. Please cite ESS. Expected columns: `climate_concern` (`wrclmch`, 1–5), `agea`, `gndr`, `eisced`, `cntry`.
- **Interaction graphs** and the **Wikipedia stimulus**: regenerated automatically (`generate_network`); the stimulus is fetched once as a **pinned revision** (`fetch_and_cache`) and cached under `data/`, so every run sees byte-identical material.
- **Run outputs** (`runs/<exp_id>/`): produced by `run_full.py`.



## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Yiwen Zhang.

## Acknowledgments

European Social Survey (ESS ERIC) for the ESS Round 10 data; Wikipedia contributors for the stimulus article.
