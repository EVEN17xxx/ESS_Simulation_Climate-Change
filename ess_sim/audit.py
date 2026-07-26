"""Offline stance audit: a blind LLM judge rates how worried each initial opinion sounds.

Manipulation check for the hard-assigned t=0 concern: the judge sees ONLY the opinion
text (no persona, no assigned value, no article). Never imported by the simulation loop.
"""
import asyncio

import pandas as pd
from pydantic import BaseModel

from .llm_client import (
    async_get_completion_from_messages_structured,
    clamp_concern,
    reset_llm_semaphore,
)

_JUDGE_SYSTEM = (
    "You are an independent text rater. You see only a short statement and know "
    "nothing about its author."
)
_JUDGE_PROMPT = (
    "Read this statement from an online discussion:\n\n"
    '"{opinion}"\n\n'
    "On a scale of 1 to 5, how worried does the author sound about climate change "
    "(1 = not at all worried, 5 = extremely worried)?\n\n"
    "Output:\n"
    "  concern: integer 1-5"
)


class _JudgeResponse(BaseModel):
    concern: int


async def _async_judge_all(opinions, model):
    tasks = [async_get_completion_from_messages_structured(
                 system_messages=_JUDGE_SYSTEM,
                 messages=_JUDGE_PROMPT.format(opinion=op),
                 model=model, temperature=0.0, response_type=_JudgeResponse)
             for op in opinions]
    return await asyncio.gather(*tasks)


def audit_initial_stance(backgrounds: dict, model: str, seed: int) -> pd.DataFrame:
    # Citizens only (stubborn agents carry an agent_type profile), non-empty opinions only
    # (empty = LLM init fallback, nothing to judge).
    rows = [(int(node), bg["initial_opinion"], int(bg["initial_concern"]))
            for node, bg in sorted(backgrounds.items(), key=lambda kv: int(kv[0]))
            if "agent_type" not in bg["demographic_profile"]
            and bg["initial_opinion"].strip()]
    reset_llm_semaphore()
    responses = asyncio.run(_async_judge_all([op for _, op, _ in rows], model))
    records = [
        {"seed": seed, "agent_id": node, "assigned": assigned,
         "judged": clamp_concern(resp.concern), "delta": clamp_concern(resp.concern) - assigned}
        for (node, _op, assigned), resp in zip(rows, responses) if resp is not None
    ]
    return pd.DataFrame(records)


def summarize_audit(df: pd.DataFrame) -> dict:
    return {
        "spearman_assigned_judged": df["assigned"].corr(df["judged"], method="spearman"),
        "mean_signed_delta": df["delta"].mean(),
        # asymmetry check: are low-assigned agents judged systematically higher?
        "mean_delta_by_assigned": df.groupby("assigned")["delta"].mean().to_dict(),
    }
