"""ess_sim — LLM-agent simulation of climate-concern dynamics on social networks.


------------------------------------------------------------------------------
ARCHITECTURE 
------------------------------------------------------------------------------

  run_ess_sim.ipynb / run_full.py        <- entry points (you run these)
        |
        v
  model.py        World: the orchestrator. Builds the network, samples agents from
                  the ESS pool, runs the step loop, drives the LLM updates, computes
                  metrics, writes all outputs to runs/<exp_id>/.
        |
        +--> agent.py        ClimateAgent: a single citizen's STATE only
        |                    (name, camp, concern/opinion history, memory). No behaviour.
        |
        +--> llm_client.py   Raw Azure-OpenAI plumbing: async client, retry, concurrency
        |                    semaphore, structured-output schema. Knows nothing about
        |                    climate/agents/networks.
        |
        +--> network.py      Topology generation (centralized / small-world / random),
        |                    network stats, homophily rewiring.
        |
        +--> ess_pool.py     The ESS respondent pool: built once from the ESS10 microdata
        |                    (`python -m ess_sim.ess_pool`), then loaded, validated, and
        |                    drawn from per node (sample_citizen_rows). Pool diagnostics.
        |
        +--> prompt.py       All prompt templates (personas, FCA/denial, update prompts).
        |
        +--> article.py      Version-pinned Wikipedia stimulus fetch + cache.
        |
        +--> metrics.py      The three dependent variables (mean concern, dispersion,
                             bimodality).

  config.py        ExperimentConfig dataclass + generate_ess_conditions() (the full
                   experiment grid: 3 topologies x 3 interventions x seeds).
  visualization.py Plotting helpers (topology comparison, ESS-vs-sampled concern
                   distribution, interaction diary).



Requires Azure OpenAI credentials in a .env file (see .env.example).
"""
