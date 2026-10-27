"""HTTP layer. Run:  uvicorn app:app --reload   (model text needs FOUNDRY_* env vars, see engine/llm.py; otherwise templates)
  GET  /insights?mode=casual_fan&lang=id        all insights for the replayed match (precomputed, zero LLM cost)
  GET  /stream?speed=30&mode=analyst&lang=en    SSE: insights released on the match clock
  POST /events                                  live ingest of ONE raw event -> insights emitted by it
  POST /reset                                   start a new live session
Personalization query params (all endpoints): favorite_team, favorite_player_id, player_focus, focus_metric
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Body, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from engine import Pipeline
from engine.render import MODES, LANGS

BASE = os.path.dirname(__file__)
EVENTS = os.environ.get("MATCH_FILE", os.path.join(BASE, "data", "synthetic_match_900001.json"))
INFO = os.environ.get("MATCH_INFO", os.path.join(BASE, "data", "match_info.json"))

# Multi-agent explain layer (Microsoft Agent Framework + Foundry). Optional: without credentials the engine serves
# the deterministic template text, so the demo never depends on model quota.
try:
    from engine.agents import AgentTeam, match_entities
    TEAM, LLM_STATUS = AgentTeam.from_env(cache_path=os.environ.get("INSIGHT_CACHE", os.path.join(BASE, "data", "llm_cache.json")))
except Exception as _ex:  # agent-framework not installed
    TEAM, LLM_STATUS, match_entities = None, f"agent framework unavailable: {type(_ex).__name__}: {_ex}", None
LLM_BUDGET_S = float(os.environ.get("INSIGHT_LLM_BUDGET_S", "45"))   # max wait per request; unfinished items stay template text


@asynccontextmanager
async def lifespan(_app):
    """Warm the cache for every mode x language in the background so the first viewer already sees model text."""
    task = None
    if TEAM:
        async def warm():
            p = replay_pipeline()
            for mode in MODES:
                for lang in LANGS:
                    await TEAM.enrich_many(render_pairs(p, mode, lang, {}), p.names, match_entities(p), {})
        task = asyncio.create_task(warm())
    yield
    if task:
        task.cancel()


app = FastAPI(title="Match Insight Engine", version="0.3.0", lifespan=lifespan,
              description="Synthetic football events -> explainable, personalized, overlay-ready insights.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
LIVE = {"pipe": Pipeline.from_file(INFO)}


@lru_cache(maxsize=1)
def replay_pipeline() -> Pipeline:
    """One offline pass over the match. Cached: this is also the cost-free fallback for judges/testing."""
    p = Pipeline.from_file(INFO)
    for e in json.load(open(EVENTS, encoding="utf-8")):
        p.feed(e)
    return p


def prefs(favorite_team, favorite_player_id, player_focus, focus_metric):
    d = {"favorite_team": favorite_team, "favorite_player_id": favorite_player_id,
         "player_focus": player_focus, "focus_metric": focus_metric}
    return {k: v for k, v in d.items() if v not in (None, False)}


def render_pairs(p: Pipeline, mode, lang, pr):
    out = [(m, i) for m in p.moments if (i := p.insight(m, mode=mode, lang=lang, prefs=pr))]
    return sorted(out, key=lambda t: t[1]["timing"]["show_at_stream_offset_s"])


def render_all(p: Pipeline, mode, lang, pr):
    return [i for _, i in render_pairs(p, mode, lang, pr)]


async def build_items(p: Pipeline, mode, lang, pr, llm=True):
    """Template insights, upgraded by the agent team when configured (items not finished in time keep template text)."""
    pairs = render_pairs(p, mode, lang, pr)
    if not (TEAM and llm):
        return [i for _, i in pairs]
    return await TEAM.enrich_many(pairs, p.names, match_entities(p), pr, budget_s=LLM_BUDGET_S)


@app.get("/meta")
def meta():
    return replay_pipeline().meta()


@app.get("/tracking")
def tracking():
    """SIMULATED 2 Hz tracking (all 22 players + ball) produced by the match simulator; see data/tracking_900001.json."""
    return FileResponse(os.path.join(BASE, "data", "tracking_900001.json"), media_type="application/json")


@app.get("/health")
def health():
    return {"ok": True, "synthetic": True, "match_id": replay_pipeline().match_id,
            "llm": {"enabled": TEAM is not None, "model": TEAM.label if TEAM else None, "status": LLM_STATUS}}


@app.get("/agents")
def agents():
    """Topology of the multi-agent layer (what runs, in which order, what happens on failure)."""
    return TEAM.topology() if TEAM else {"enabled": False, "reason": LLM_STATUS}


@app.get("/insights")
async def insights(mode: str = "analyst", lang: str = "en", favorite_team: str | None = None,
                   favorite_player_id: int | None = None, player_focus: bool = False, focus_metric: str | None = None,
                   llm: bool = True):
    return await build_items(replay_pipeline(), mode, lang, prefs(favorite_team, favorite_player_id, player_focus, focus_metric), llm)


@app.get("/recap")
async def recap(mode: str = "casual_fan", lang: str = "en"):
    """Full-time recap. RecapAgent output passes the same verifier; otherwise a deterministic template recap."""
    p = replay_pipeline()
    if not TEAM:
        return template_recap(p, lang)
    items = await build_items(p, mode, lang, {}, True)
    return await TEAM.recap(items, p.meta(), mode, lang, match_entities(p))


def template_recap(p, lang):
    from engine.agents import AgentTeam
    meta = p.meta()
    final = meta["goals"][-1]["score"] if meta["goals"] else "0-0"
    full = {c: t["name"] for c, t in meta["teams"].items()}
    return {"mode": None, "language": lang, "source": "template", "model": "deterministic-templates-v0",
            **AgentTeam._template_recap(meta, final, full, lang)}


@app.get("/stream")
async def stream(speed: float = Query(30.0, ge=0), mode: str = "analyst", lang: str = "en",
                 favorite_team: str | None = None, favorite_player_id: int | None = None,
                 player_focus: bool = False, focus_metric: str | None = None):
    items = await build_items(replay_pipeline(), mode, lang, prefs(favorite_team, favorite_player_id, player_focus, focus_metric))

    async def gen():
        prev = 0.0
        for i in items:
            t = i["timing"]["show_at_stream_offset_s"]
            if speed > 0:
                await asyncio.sleep(max(0.0, (t - prev) / speed))
            prev = t
            yield f"event: insight\ndata: {json.dumps(i, ensure_ascii=False)}\n\n"
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/events")
async def ingest(event: dict = Body(...), mode: str = "analyst", lang: str = "en",
           favorite_team: str | None = None, favorite_player_id: int | None = None,
           player_focus: bool = False, focus_metric: str | None = None):
    p = LIVE["pipe"]
    pr = prefs(favorite_team, favorite_player_id, player_focus, focus_metric)
    pairs = [(m, i) for m in p.feed(event) if (i := p.insight(m, mode=mode, lang=lang, prefs=pr))]
    out = await TEAM.enrich_many(pairs, p.names, match_entities(p), pr, budget_s=LLM_BUDGET_S) if (TEAM and pairs) else [i for _, i in pairs]
    return {"emitted": out, "counters": p.state.counters}


@app.post("/reset")
def reset():
    LIVE["pipe"] = Pipeline.from_file(INFO)
    return {"ok": True}


# Frontend (static): served at / ; the page falls back to static/snapshots.js if the API is unreachable.
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))
