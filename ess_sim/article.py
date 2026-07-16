"""Version-pinned Wikipedia stimulus fetcher + cache."""
import os
import json
import requests

import re
from html import unescape as _html_unescape
from datetime import datetime, timezone

HEADERS = {
    "User-Agent": (
        "ClimateFactCheckResearch/1.0 "
        "(Academic research on LLM agent opinion dynamics; "
        "contact: research@university.edu)"
    ),
    "Accept": "application/json",
}

WIKI_API = "https://en.wikipedia.org/w/api.php"

TOPIC_WIKI_MAPPING = {
    "climate_europe": {
        "title": "Climate_change_in_Europe",
        "question": (
            "What are your thoughts on climate change and its effects in Europe?"
        ),
    },
}



def _truncate_at_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chars // 2:
        return truncated[: last_period + 1]
    return truncated


def _html_to_plaintext(html: str) -> str:
    # Extract paragraph prose only (skips infoboxes, tables, navboxes, reference lists),
    # drop citation superscripts, strip tags, unescape entities, collapse whitespace.
    chunks = []
    for p in re.findall(r"<p\b[^>]*>(.*?)</p>", html, flags=re.S | re.I):
        p = re.sub(r"<sup\b[^>]*>.*?</sup>", "", p, flags=re.S | re.I)
        p = re.sub(r"<[^>]+>", "", p)
        p = _html_unescape(p)
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            chunks.append(p)
    return "\n\n".join(chunks)


def get_revision_meta(title: str, oldid=None) -> dict:
    """Revision metadata via the MediaWiki Action API. oldid=None -> current revision;
    otherwise the exact pinned revision. Returns a full version identity."""
    params = {
        "action": "query", "format": "json", "formatversion": "2",
        "prop": "revisions", "rvprop": "ids|timestamp|sha1|size", "rvslots": "main",
    }
    if oldid is None:
        params["titles"] = title
        params["rvlimit"] = 1
    else:
        params["revids"] = str(oldid)
    resp = requests.get(WIKI_API, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", [])
    if not pages:
        raise RuntimeError(f"No page returned for title={title!r} oldid={oldid!r}")
    page = pages[0]
    if page.get("missing"):
        raise RuntimeError(f"Wikipedia page not found: {title!r}")
    revs = page.get("revisions", [])
    if not revs:
        raise RuntimeError(f"No revision returned for title={title!r} oldid={oldid!r}")
    rev = revs[0]
    resolved_title = page.get("title", title)
    resolved_oldid = rev["revid"]
    return {
        "title": resolved_title,
        "oldid": resolved_oldid,
        "timestamp": rev["timestamp"],           # ISO-8601 UTC
        "sha1": rev.get("sha1"),
        "size": rev.get("size"),
        "permalink": (
            "https://en.wikipedia.org/w/index.php?title="
            f"{resolved_title.replace(' ', '_')}&oldid={resolved_oldid}"
        ),
    }


def fetch_revision_text(title: str, oldid, max_chars: int = 3000) -> str:
    """Fetch the EXACT revision's content by oldid and strip to plaintext.

    Uses action=parse&oldid=<oldid>&prop=text (rendered HTML of that revision), NOT
    TextExtracts (prop=extracts): extracts is page-scoped, cannot pin an oldid, and
    silently returns the current revision -- the bug this rewrite fixes."""
    params = {
        "action": "parse", "format": "json", "formatversion": "2",
        "oldid": str(oldid), "prop": "text", "disablelimitreport": "1",
    }
    resp = requests.get(WIKI_API, headers=HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    html = resp.json()["parse"]["text"]
    return _truncate_at_sentence(_html_to_plaintext(html), max_chars)


def fetch_and_cache(topic_key: str, pinned_oldid=None, cache_dir: str = "./data") -> dict:
    """Return a reproducible, version-pinned stimulus record for topic_key.

    If a cache file exists, read it and NEVER touch live Wikipedia. On first fetch,
    pin the revision (pinned_oldid, or the current revision if None), record its full
    identity (oldid/timestamp/sha1/size/permalink), and cache it."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"wiki_{topic_key}.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[INFO] Loaded cached article for '{topic_key}' (oldid={data.get('oldid')})")
        return data

    if topic_key not in TOPIC_WIKI_MAPPING:
        raise ValueError(f"Unknown topic '{topic_key}'. Available: {list(TOPIC_WIKI_MAPPING.keys())}")
    mapping  = TOPIC_WIKI_MAPPING[topic_key]
    title    = mapping["title"]
    question = mapping["question"]

    try:
        meta    = get_revision_meta(title, oldid=pinned_oldid)
        article = fetch_revision_text(title, meta["oldid"], max_chars=3000)
        if len(article) < 800:
            raise RuntimeError(f"parsed revision text too short ({len(article)} chars)")
        record = {
            "question":           question,
            "article":            article,
            "wiki_title":         meta["title"],
            "oldid":              meta["oldid"],
            "revision_timestamp": meta["timestamp"],
            "sha1":               meta["sha1"],
            "size":               meta["size"],
            "permalink":          meta["permalink"],
            "retrieved_at":       datetime.now(timezone.utc).isoformat(),
            "source":             "wikipedia_pinned_revision",
        }
        print(f"[INFO] Fetched '{topic_key}' from oldid={meta['oldid']} "
              f"({meta['timestamp']}, {len(article)} chars)")
    except Exception as e:
        # No hardcoded fallback: fail loudly rather than silently substitute a
        # different, hand-written stimulus. The cache file is shipped for reproducibility.
        raise RuntimeError(
            f"Live Wikipedia fetch failed for '{topic_key}' and no cache exists at "
            f"{cache_path}. Restore the cached stimulus or retry with network access."
        ) from e

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Cached to {cache_path}")
    return record