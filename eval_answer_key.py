#!/usr/bin/env python3
"""Answer-key harness: replay -> pipeline -> compare with match_info.answer_key_moments.

  python eval_answer_key.py                       # baseline run, writes report to reports/
  python eval_answer_key.py --dup 0.1 --seed 7    # failure test: 10% duplicate events
  python eval_answer_key.py --empty 0.05          # failure test: 5% empty/garbage events
Metrics: recall of key moments, automatic explanation checks, grounding (unbacked numbers), latency, extras.
"""
import argparse, json, os, random, re, statistics, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from engine import Pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--events", default="data/synthetic_match_900001.json")
ap.add_argument("--info", default="data/match_info.json")
ap.add_argument("--dup", type=float, default=0.0)
ap.add_argument("--empty", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="reports")
a = ap.parse_args()

info = json.load(open(a.info, encoding="utf-8"))
events = json.load(open(a.events, encoding="utf-8"))
rnd = random.Random(a.seed)
stream = []
for e in events:
    stream.append(e)
    if rnd.random() < a.dup:
        stream.append(dict(e))                       # exact duplicate
    if rnd.random() < a.empty:
        stream.append(rnd.choice([None, {}, {"id": "x"}, {"type": {"name": "Pass"}}]))

pipe = Pipeline(info)
t0 = time.perf_counter()
insights = pipe.run_file(stream, mode="analyst", lang="en")
wall = time.perf_counter() - t0

# schema validation (optional dependency)
schema_errors = None
try:
    from jsonschema import Draft202012Validator
    v = Draft202012Validator(json.load(open("insight.schema.json")))
    schema_errors = sum(1 for i in insights if list(v.iter_errors(i)))
except ImportError:
    pass

by_event = {i["trigger"]["event_ids"][0]: i for i in insights}
rows, hit = [], 0
for k in info["answer_key_moments"]:
    ins = by_event.get(k["event_ids"][0])
    want_score = re.findall(r"\d-\d", k["headline"])
    r = {"key": k["key"], "minute": k["match_minute"], "detected": ins is not None}
    if ins:
        hit += 1
        hl = ins["headline"]
        who = k["player"].split()[-1]
        r.update(headline=hl, why=ins["why_it_matters"],
                 names_player=who in hl,
                 score_ok=(want_score[0] in hl) if want_score else None,
                 grounded=ins["provenance"]["grounding"]["status"] == "verified",
                 unbacked=ins["provenance"]["grounding"]["unbacked_numbers"],
                 latency_ms=ins["provenance"]["latency_ms"]["event_to_insight"], tier=ins["importance"]["tier"])
    rows.append(r)
key_ids = {k["event_ids"][0] for k in info["answer_key_moments"]}
extras = [i for e, i in by_event.items() if e not in key_ids]
lat = sorted(i["provenance"]["latency_ms"]["event_to_insight"] for i in insights)
p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] if lat else None
summary = {
    "events_in_stream": len(stream), "counters": pipe.state.counters, "insights": len(insights),
    "key_moments_detected": f"{hit}/{len(rows)}", "extra_candidates": len(extras),
    "grounded": f"{sum(1 for i in insights if i['provenance']['grounding']['status']=='verified')}/{len(insights)}",
    "unbacked_numbers_total": sum(len(i["provenance"]["grounding"]["unbacked_numbers"]) for i in insights),
    "schema_invalid": schema_errors, "latency_ms_p50": p(0.5), "latency_ms_p95": p(0.95),
}
os.makedirs(a.out, exist_ok=True)
tag = f"dup{a.dup}_empty{a.empty}"
json.dump({"summary": summary, "rows": rows, "extras": [{"minute": i["timing"]["display_clock"], "headline": i["headline"]} for i in extras],
           "insight_ids": sorted(i["insight_id"] for i in insights)}, open(f"{a.out}/result_{tag}.json", "w"), indent=2, ensure_ascii=False)
md = [f"# Answer-key report ({tag})", "", "## Summary", ""] + [f"- **{k}**: {v}" for k, v in summary.items() if v is not None]
md += ["", "## Key moments", "", "| Key | Min | Detected | Player named | Score ok | Grounded | Tier | ms | Manual: explanation correct? |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['key']} | {r['minute']}' | {'yes' if r['detected'] else 'NO'} | {r.get('names_player','-')} | {r.get('score_ok','-')} | {r.get('grounded','-')} | {r.get('tier','-')} | {r.get('latency_ms','-')} | [ ] |")
md += ["", "## Generated text for key moments (review manually)", ""]
for r in rows:
    if r["detected"]:
        md.append(f"- **{r['key']}**: {r['headline']} — {r['why']}")
md += ["", "## Extra candidates (not in key; review: legitimate or false positive?)", ""] + [f"- {i['timing']['display_clock']}: {i['headline']}" for i in extras]
open(f"{a.out}/report_{tag}.md", "w", encoding="utf-8").write("\n".join(md))
print(json.dumps(summary, indent=2))
print(f"report -> {a.out}/report_{tag}.md")
