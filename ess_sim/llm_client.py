"""LLM client: Azure OpenAI async wrapper, concurrency semaphore, structured-output schema."""
import os
import asyncio
import random
import openai
from pydantic import BaseModel
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv(override=True)

# Azure OpenAI client (sync + async)
def _required_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill in your Azure OpenAI "
            f"credentials (endpoint, API key, deployment name), then re-run from the repo root."
        )
    return v

_azure_endpoint = _required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
_azure_key      = _required_env("AZURE_OPENAI_API_KEY")
_azure_deploy   = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

_az_kwargs = dict(
    api_key=_azure_key,
    base_url=f"{_azure_endpoint}/openai/deployments/{_azure_deploy}",
    default_headers={"api-key": _azure_key},
    default_query={"api-version": "2024-08-01-preview"},
)
async_client = openai.AsyncOpenAI(**_az_kwargs)

_llm_semaphore  = None
_max_concurrent = 5   # pre-World default; World overrides via config.max_concurrent_llm_calls
MAX_LLM_RETRIES = 30
# cumulative; reset per matrix run. `calls` counts SUCCEEDED calls only -- retried attempts
# still cost latency (and tokens, if the request reached the model) but are not tallied there.
# `failures` counts calls that exhausted every retry and returned None: each one silently
# becomes a "no change" for an agent, so a non-zero value biases every DV toward the null.
LLM_STATS = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "failures": 0}

def clamp_concern(v):
    return None if v is None else int(round(max(1.0, min(5.0, float(v)))))

def set_max_concurrent_llm_calls(n: int):
    global _max_concurrent, _llm_semaphore
    _max_concurrent = n
    _llm_semaphore  = None

def reset_llm_semaphore():
    global _llm_semaphore
    _llm_semaphore = None

async def _get_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(_max_concurrent)
    return _llm_semaphore

class UpdateOpinionResponse(BaseModel):
    opinion: str
    concern: float
    rationale: str

async def async_get_completion_from_messages_structured(
    messages, system_messages="You are a helpful assistant.",
    model="gpt-4o-mini", temperature=0.7,
    response_type=UpdateOpinionResponse, max_retries=MAX_LLM_RETRIES,
):
    sem = await _get_semaphore()
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await async_client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_messages},
                        {"role": "user",   "content": messages},
                    ],
                    temperature=temperature,
                    response_format=response_type,
                )
                LLM_STATS["calls"] += 1
                if response.usage:
                    LLM_STATS["prompt_tokens"] += response.usage.prompt_tokens
                    LLM_STATS["completion_tokens"] += response.usage.completion_tokens
                return response.choices[0].message.parsed
            except openai.RateLimitError:
                if attempt >= max_retries - 1:
                    LLM_STATS["failures"] += 1
                    print(f"[LLM ERROR] Rate-limited on all {max_retries} attempts; giving up.")
                    return None
                wait_time = min(2 ** attempt + random.random(), 60)
                print(f"[RATE LIMIT] Attempt {attempt+1}/{max_retries}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 60))   # cap: uncapped 2**29 would hang for years
                else:
                    LLM_STATS["failures"] += 1
                    print(f"[LLM ERROR] Failed after {max_retries} attempts: {e}")
                    return None
    LLM_STATS["failures"] += 1
    return None