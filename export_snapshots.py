#!/usr/bin/env python3
"""Pre-render every (mode, language) combination to static/snapshots.js.
The frontend uses it when the API is unreachable (zero-cost judge/demo fallback, works from file://).

  python export_snapshots.py          # deterministic template text (no credentials needed)
  python export_snapshots.py --llm    # run the agent team (ExplainAgent -> PersonaAgent -> VerifierAgent, RecapAgent) with the
                                      # model configured in the environment (see engine/llm.py); unverified text falls back to templates
Model output is cached in data/llm_cache.json, so re-exporting costs nothing and committing that file lets a deployment
serve model-written, verified text without calling the model at all.
"""
import asyncio, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from engine import Pipeline
from engine.render import MODES, LANGS

B = os.path.dirname(__file__)
USE_LLM = "--llm" in sys.argv
p = Pipeline.from_file(f"{B}/data/match_info.json")
for e in json.load(open(f"{B}/data/synthetic_match_900001.json", encoding="utf-8")):
    p.feed(e)

team = None
if USE_LLM:
    from engine.agents import AgentTeam, match_entities
    team, status = AgentTeam.from_env(cache_path=f"{B}/data/llm_cache.json")
    if not team:
        sys.exit(f"--llm requested but no model is available: {status}")
    ents = match_entities(p)
    print("model:", team.label)


def pairs(mode, lang):
    return sorted([(mo, i) for mo in p.moments if (i := p.insight(mo, mode=mode, lang=lang))],
                  key=lambda t: t[1]["timing"]["show_at_stream_offset_s"])


def template_recap(lang):
    from engine.agents import AgentTeam as T
    meta = p.meta()
    final = meta["goals"][-1]["score"]
    return {"language": lang, "source": "template", "model": "deterministic-templates-v0",
            **T._template_recap(meta, final, {c: t["name"] for c, t in meta["teams"].items()}, lang)}


async def build():
    out = {"meta": p.meta()}
    for m in MODES:
        for l in LANGS:
            pr = pairs(m, l)
            items = await team.enrich_many(pr, p.names, ents, {}) if team else [i for _, i in pr]
            out[f"{m}_{l}"] = items
            out[f"recap_{m}_{l}"] = await team.recap(items, out["meta"], m, l, ents) if team else template_recap(l)
            if team:
                degraded = sum(1 for i in items if i["provenance"]["degraded"])
                print(f"{m}_{l}: {len(items)} insights, {degraded} fell back to template; recap source = {out[f'recap_{m}_{l}']['source']}")
    return out


out = asyncio.run(build())
open(f"{B}/static/snapshots.js", "w", encoding="utf-8").write("window.SNAPSHOTS = " + json.dumps(out, ensure_ascii=False) + ";\n")
print("wrote static/snapshots.js", {k: len(v) for k, v in out.items() if isinstance(v, list)}, "duration", out["meta"]["duration_s"], "p1_end", out["meta"]["p1_end_s"])
