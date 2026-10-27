"""Multi-agent explain layer, built on Microsoft Agent Framework (Agent + Workflow).

                 analysis                 candidate                    verdict
 Moment ──▶ ExplainAgent ───────▶ PersonaAgent ───────▶ VerifierAgent ──── ok ───────▶ insight (LLM text)
 (facts,      (LLM, Foundry)      (LLM, Foundry)          (deterministic)   │
  factors,                              ▲                                    ├─ problems ─▶ back to PersonaAgent (feedback, max N)
  metrics)                              └────────────────────────────────────┘
                                                                             └─ exhausted / agent error ─▶ template text (provenance.degraded)

* ExplainAgent  - decides WHY the moment matters, in neutral English, using only the supplied factors and metrics.
* PersonaAgent  - turns that analysis into the viewer's voice: analyst / casual fan / broadcaster, English or Indonesian,
                  favourite club and player.
* VerifierAgent - not an LLM: engine.verify.verify_llm_text. Every digit and spelled number must be backed by metrics[],
                  every capitalised name must be on the roster. Failures go back to PersonaAgent as concrete feedback.
* RecapAgent    - writes the full-time recap from the verified insights, through the same verifier.
The deterministic template text from engine.render is always the fallback, so a model outage, quota error or
hallucination can never put an unverified sentence on screen.
"""
import asyncio
import copy
import json
import os
import re
import time
from dataclasses import dataclass, field

from agent_framework import Agent, Executor, WorkflowBuilder, WorkflowContext, handler

from .verify import verify_llm_text

MAX_ATTEMPTS = int(os.getenv("INSIGHT_LLM_ATTEMPTS", "2"))
TIMEOUT_S = float(os.getenv("INSIGHT_LLM_TIMEOUT", "25"))
LANG_NAME = {"en": "English", "id": "Indonesian (Bahasa Indonesia)"}
STYLE = {
    "analyst": "Precise and tactical. Two sentences. Cite one or two of the supplied metrics with their units (xG, % pass share, metres).",
    "casual_fan": "Plain words, no jargon (never say xG or pass share), warm and emotional, one or two short sentences, at most one number.",
    "broadcaster": "Commentary-ready: punchy, present tense, quotable on a lower-third. Two short sentences.",
}

EXPLAIN_INSTRUCTIONS = """You are ExplainAgent in a football match-intelligence pipeline (synthetic data).
You receive one JSON object describing a match moment: kind, clock, facts, factors (type + params) and metrics.
Write a 2-3 sentence ENGLISH analysis of WHY the moment matters: control vs chaos, tactical pressure shift, game state, rhythm, milestone.
Hard rules:
- Use only the supplied factors, facts and metrics. Never invent players, minutes, statistics or events.
- Every number you write must be one of "allowed_numbers", formatted exactly as listed. Do not spell numbers out as words.
- Refer to a moment's time only by writing the supplied "clock" string exactly.
- Mention names exactly as in "allowed_names".
Return ONLY JSON: {"analysis": "<text>"}"""

PERSONA_INSTRUCTIONS = """You are PersonaAgent in a football match-intelligence pipeline (synthetic data).
You turn a verified English analysis into an on-screen overlay for one specific viewer.
Input JSON: audience (mode, language, style), preferences, clock, analysis, template_headline (for reference only),
allowed_numbers, allowed_names, previous_problems (empty on the first try).
Write in the audience's language. Hard rules:
- headline: max 85 characters. why_it_matters: max 300 characters.
- Use ONLY facts from "analysis". Every number must be one of "allowed_numbers", exactly as listed (use a decimal comma in Indonesian).
  Never spell numbers as words (no "three", "tiga"). If you mention the time, write the clock string exactly as supplied.
- Names must be exactly as in "allowed_names"; do not introduce any other person, club or competition.
- If a favourite team or player is given, speak to that fan and mention them only if they are part of the moment.
- If previous_problems is not empty, fix every item in it.
Return ONLY JSON: {"headline": "<text>", "why_it_matters": "<text>"}"""

RECAP_INSTRUCTIONS = """You are RecapAgent in a football match-intelligence pipeline (synthetic data).
You receive the verified key moments of one match plus the final score and write the full-time recap for one audience.
Hard rules:
- Write 90-140 words in the audience's language, following its style. Narrative arc: how the match turned, not a list.
- Use only the supplied moments. Every number must be one of "allowed_numbers", exactly as listed. Never spell numbers as words.
- When mentioning a time, write the clock string exactly as supplied (for example 61'). Names exactly as in "allowed_names".
- If previous_problems is not empty, fix every item in it.
Return ONLY JSON: {"title": "<max 80 characters>", "recap": "<text>"}"""


def parse_json(text: str) -> dict:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        raise ValueError("no JSON object in model output")
    return json.loads(t[i:j + 1])


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def allowed_numbers(metrics: list) -> list[str]:
    from .verify import _allowed
    return sorted(_allowed(metrics), key=float)


@dataclass
class Job:
    """Shared state that travels through the workflow (the hand-off contract between agents)."""
    insight: dict                     # template insight: schema-valid, grounded, and the fallback
    moment: dict
    names: dict                       # team code -> short name
    entities: set
    mode: str
    lang: str
    prefs: dict
    analysis: str = ""
    candidate: dict | None = None
    problems: list = field(default_factory=list)
    attempts: int = 0
    retry: bool = False
    error: str | None = None
    final: dict | None = None
    verdict: dict | None = None
    trace: list = field(default_factory=list)


# ----------------------------------------------------------------------------- executors
class ExplainExecutor(Executor):
    def __init__(self, client):
        super().__init__(id="explain")
        self.client = client

    @handler
    async def handle(self, job: Job, ctx: WorkflowContext[Job]) -> None:
        t0 = time.perf_counter()
        mo = job.moment
        tn = lambda c: job.names.get(c, c)
        payload = {
            "kind": mo["kind"], "clock": mo["clock"], "facts": mo["facts"],
            "team": tn(mo["team"]),
            "factors": [{"type": f["type"], "params": {k: (tn(v) if k in ("team", "opp") else v) for k, v in f["params"].items()}}
                        for f in mo["factors"]],
            "metrics": [{"key": m["key"], "label": m["label"], "value": m["value"], "unit": m["unit"]} for m in mo["metrics"]],
            "allowed_numbers": allowed_numbers(mo["metrics"]),
            "allowed_names": sorted(job.entities),
        }
        try:
            agent = Agent(self.client, EXPLAIN_INSTRUCTIONS, name="ExplainAgent")
            r = await asyncio.wait_for(agent.run(json.dumps(payload, ensure_ascii=False)), TIMEOUT_S)
            job.analysis = str(parse_json(r.text)["analysis"]).strip()[:900]
            if not job.analysis:
                raise ValueError("empty analysis")
        except Exception as ex:
            job.error = f"ExplainAgent failed: {type(ex).__name__}: {ex}"[:300]
        job.trace.append({"stage": "explain", "agent": "ExplainAgent", "latency_ms": _ms(t0)})
        await ctx.send_message(job)


class PersonaExecutor(Executor):
    def __init__(self, client):
        super().__init__(id="persona")
        self.client = client

    @handler
    async def handle(self, job: Job, ctx: WorkflowContext[Job]) -> None:
        job.retry = False
        if job.error:                                  # upstream failure: pass through, the verifier will fall back
            await ctx.send_message(job)
            return
        t0 = time.perf_counter()
        pr = job.prefs
        payload = {
            "audience": {"mode": job.mode, "language": LANG_NAME[job.lang], "style": STYLE.get(job.mode, STYLE["analyst"])},
            "preferences": {"favorite_team": job.names.get(pr.get("favorite_team")) if pr.get("favorite_team") else None,
                            "favorite_player_id": pr.get("favorite_player_id"), "player_focus": bool(pr.get("player_focus")),
                            "focus_metric": pr.get("focus_metric")},
            "clock": job.moment["clock"], "analysis": job.analysis, "template_headline": job.insight["headline"],
            "allowed_numbers": allowed_numbers(job.insight["metrics"]), "allowed_names": sorted(job.entities),
            "previous_problems": job.problems,
        }
        try:
            agent = Agent(self.client, PERSONA_INSTRUCTIONS, name="PersonaAgent")
            r = await asyncio.wait_for(agent.run(json.dumps(payload, ensure_ascii=False)), TIMEOUT_S)
            d = parse_json(r.text)
            job.candidate = {"headline": str(d["headline"]).strip(), "why_it_matters": str(d["why_it_matters"]).strip()}
        except Exception as ex:
            job.candidate, job.error = None, f"PersonaAgent failed: {type(ex).__name__}: {ex}"[:300]
        job.attempts += 1
        job.trace.append({"stage": "personalize", "agent": "PersonaAgent", "latency_ms": _ms(t0)})
        await ctx.send_message(job)


class VerifyExecutor(Executor):
    def __init__(self):
        super().__init__(id="verify")

    @handler
    async def handle(self, job: Job, ctx: WorkflowContext[Job, Job]) -> None:
        t0 = time.perf_counter()
        if not job.error and job.candidate:
            v = verify_llm_text(job.candidate["headline"], job.candidate["why_it_matters"], job.insight["metrics"], job.entities)
            job.verdict = v
            job.trace.append({"stage": "verify", "agent": "VerifierAgent", "latency_ms": _ms(t0)})
            if v["status"] == "verified":
                job.final = job.candidate
            elif job.attempts < MAX_ATTEMPTS:
                job.problems, job.retry = v["problems"], True      # hand back to PersonaAgent with concrete feedback
                await ctx.send_message(job)
                return
            else:
                job.error = "verification failed after %d attempts: %s" % (job.attempts, "; ".join(v["problems"]))[:300]
        await ctx.yield_output(job)


def build_workflow(client):
    explain, persona, verify = ExplainExecutor(client), PersonaExecutor(client), VerifyExecutor()
    return (WorkflowBuilder(start_executor=explain, output_from=[verify])
            .add_edge(explain, persona)
            .add_edge(persona, verify)
            .add_edge(verify, persona, condition=lambda j: bool(getattr(j, "retry", False)))
            .build())


# ----------------------------------------------------------------------------- team
def apply_result(insight: dict, job: Job, label: str) -> dict:
    out = copy.deepcopy(insight)
    prov = out["provenance"]
    interpret = [t for t in prov.get("trace", []) if t["stage"] == "interpret"]
    agents_ms = round(sum(t["latency_ms"] for t in job.trace), 2)
    prov["trace"] = interpret + job.trace
    if job.final:
        hl, why = job.final["headline"], job.final["why_it_matters"]
        out["headline"], out["why_it_matters"] = hl, why
        out["overlay"]["payload"]["title"], out["overlay"]["payload"]["subtitle"] = hl[:60], why[:120]
        prov.update({"produced_by": ["InterpretAgent", "ExplainAgent", "PersonaAgent", "VerifierAgent"], "model": label,
                     "confidence": 0.94 if job.attempts == 1 else 0.88, "degraded": False})
        prov.pop("degraded_reason", None)
        prov["grounding"] = {k: job.verdict[k] for k in ("status", "numbers_checked", "unbacked_numbers")}
        prov["latency_ms"] = {"event_to_insight": round(sum(t["latency_ms"] for t in prov["trace"]), 2)}
    else:                                               # template text stays; say honestly why
        prov["degraded"], prov["degraded_reason"] = True, "llm_fallback: " + (job.error or "unknown")
        prov["latency_ms"] = {"event_to_insight": round(prov["latency_ms"]["event_to_insight"] + agents_ms, 2)}
    return out


class AgentTeam:
    def __init__(self, client, label: str, concurrency: int = 6, cache_path: str | None = None):
        self.client, self.label, self.concurrency = client, label, concurrency
        self.cache_path, self.cache = cache_path, {}
        if cache_path and os.path.exists(cache_path):
            try:
                self.cache = json.load(open(cache_path, encoding="utf-8"))
            except Exception:
                self.cache = {}

    @classmethod
    def from_env(cls, cache_path: str | None = None):
        from .llm import make_chat_client
        client, label = make_chat_client()
        return (cls(client, label, cache_path=cache_path), label) if client else (None, label)

    def save_cache(self):
        if self.cache_path:
            json.dump(self.cache, open(self.cache_path, "w", encoding="utf-8"), ensure_ascii=False)

    # -- one insight --------------------------------------------------------------------
    async def enrich(self, insight: dict, moment: dict, names: dict, entities: set, prefs: dict | None = None) -> dict:
        key = f"{insight['insight_id']}|{self.label}"
        hit = self.cache.get(key)
        if hit:
            job = Job(insight, moment, names, entities, insight["audience_mode"], insight["language"], prefs or {})
            job.final, job.verdict, job.trace, job.attempts = hit["final"], hit["verdict"], hit["trace"], hit["attempts"]
            return apply_result(insight, job, self.label)
        job = Job(insight, moment, names, entities, insight["audience_mode"], insight["language"], prefs or {})
        try:
            res = await build_workflow(self.client).run(job)
            outs = res.get_outputs()
            job = outs[0] if outs else job
            if not outs:
                job.error = "workflow produced no output"
        except Exception as ex:
            job.error = f"workflow error: {type(ex).__name__}: {ex}"[:300]
        if job.final:
            self.cache[key] = {"final": job.final, "verdict": job.verdict, "trace": job.trace, "attempts": job.attempts}
        return apply_result(insight, job, self.label)

    async def enrich_many(self, pairs: list, names: dict, entities: set, prefs: dict | None = None, budget_s: float | None = None) -> list:
        """pairs: [(moment, insight)] -> insights in the same order; anything not done within budget_s stays template text."""
        sem = asyncio.Semaphore(self.concurrency)

        async def one(mo, ins):
            async with sem:
                return await self.enrich(ins, mo, names, entities, prefs)
        tasks = [asyncio.ensure_future(one(mo, ins)) for mo, ins in pairs]
        done, pending = await asyncio.wait(tasks, timeout=budget_s)
        for t in pending:
            t.cancel()
        out = []
        for t, (_, ins) in zip(tasks, pairs):
            try:
                out.append(t.result() if t in done and not t.cancelled() else ins)
            except Exception:
                out.append(ins)
        self.save_cache()
        return out

    # -- full-time recap --------------------------------------------------------------------
    async def recap(self, insights: list, meta: dict, mode: str, lang: str, entities: set) -> dict:
        teams = {c: t["short"] for c, t in meta["teams"].items()}
        full = {c: t["name"] for c, t in meta["teams"].items()}
        final = meta["goals"][-1]["score"] if meta["goals"] else "0-0"
        pseudo = [m for i in insights for m in i["metrics"]] + [{"key": "final_score", "value": final}]
        base = {"mode": mode, "language": lang, "source": "template", "model": "deterministic-templates-v0"}
        template = self._template_recap(meta, final, full, lang)
        key = f"recap|{mode}|{lang}|{self.label}|{final}|{len(insights)}"
        if key in self.cache:
            return {**base, **self.cache[key], "source": "llm", "model": self.label}
        payload = {"audience": {"mode": mode, "language": LANG_NAME[lang], "style": STYLE.get(mode, STYLE["analyst"])},
                   "teams": full, "final_score": final,
                   "moments": [{"clock": i["timing"]["display_clock"], "headline": i["headline"], "why": i["why_it_matters"]} for i in insights],
                   "allowed_numbers": allowed_numbers(pseudo), "allowed_names": sorted(entities), "previous_problems": []}
        problems, reason = [], "not attempted"
        for attempt in range(MAX_ATTEMPTS):
            payload["previous_problems"] = problems
            try:
                agent = Agent(self.client, RECAP_INSTRUCTIONS, name="RecapAgent")
                r = await asyncio.wait_for(agent.run(json.dumps(payload, ensure_ascii=False)), TIMEOUT_S)
                d = parse_json(r.text)
                v = verify_llm_text(str(d["title"]), str(d["recap"]), pseudo, entities, limits={"headline": 90, "why_it_matters": 1100})
                if v["status"] == "verified":
                    res = {"title": d["title"], "recap": d["recap"], "attempts": attempt + 1,
                           "grounding": {k: v[k] for k in ("status", "numbers_checked", "unbacked_numbers")}}
                    self.cache[key] = res
                    self.save_cache()
                    return {**base, **res, "source": "llm", "model": self.label}
                problems, reason = v["problems"], "; ".join(v["problems"])[:300]
            except Exception as ex:
                reason = f"{type(ex).__name__}: {ex}"[:300]
        return {**base, **template, "degraded_reason": "llm_fallback: " + reason}

    @staticmethod
    def _template_recap(meta, final, full, lang):
        a, b = (full[c] for c in meta["order"])
        beats = "; ".join(f"{g['clock']} {g['score']}" for g in meta["goals"])
        if lang == "id":
            return {"title": f"Hasil akhir: {a} {final} {b}", "recap": f"{a} dan {b} bermain; skor akhir {final}. Gol-gol: {beats}."}
        return {"title": f"Full time: {a} {final} {b}", "recap": f"{a} against {b} finished {final}. Goals: {beats}."}

    def topology(self) -> dict:
        return {"framework": "Microsoft Agent Framework (Agent + Workflow)", "model": self.label, "max_attempts": MAX_ATTEMPTS,
                "agents": [{"name": "InterpretAgent", "kind": "deterministic", "role": "events -> moments with provenance"},
                           {"name": "ExplainAgent", "kind": "llm", "role": "why the moment matters, from factors + metrics only"},
                           {"name": "PersonaAgent", "kind": "llm", "role": "voice for analyst / casual fan / broadcaster, EN / ID, favourites"},
                           {"name": "VerifierAgent", "kind": "deterministic", "role": "numbers, names, length; returns concrete feedback"},
                           {"name": "RecapAgent", "kind": "llm", "role": "full-time recap from verified insights, same verifier"}],
                "edges": ["explain -> persona", "persona -> verify", "verify -> persona (while problems and attempts < max)",
                          "verify -> output (verified, or template fallback with provenance.degraded)"]}


def match_entities(pipeline) -> set:
    """Every name the model may legitimately mention: both clubs and all players that appear in a moment."""
    ents = set()
    for t in pipeline.info["teams"].values():
        ents.update(t["name"].split())
        ents.add(t["name"])
    for m in pipeline.moments:
        for pl in m["players"]:
            ents.update(pl["name"].split())
            ents.add(pl["name"])
    return ents
