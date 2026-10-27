"""Orchestrates the 5 stages. Agents are plain callables so they can be swapped for Agent Framework agents."""
import json
import time
from .state import MatchState
from .detect import detect
from .render import render
from .verify import verify


class Pipeline:
    def __init__(self, match_info: dict):
        self.info = match_info
        self.match_id = match_info["match_id"]
        self.codes = {t["id"]: c for c, t in match_info["teams"].items()}
        self.names = {c: t["name"].split()[0] for c, t in match_info["teams"].items()}
        self.state = MatchState(self.codes)
        self.moments: list[dict] = []

    @classmethod
    def from_file(cls, path: str):
        return cls(json.load(open(path, encoding="utf-8")))

    def feed(self, raw) -> list[dict]:
        """Ingest one raw event -> neutral Moments (may be empty). Never raises on bad input."""
        t0 = time.perf_counter()
        try:
            if not self.state.update(raw):
                return []
            ms = detect(self.state, raw)
        except Exception as ex:                      # an unexpected event must not kill the stream
            self.state.counters["invalid"] += 1
            return []
        dt = round((time.perf_counter() - t0) * 1000, 2)
        for m in ms:
            m["_feed_ms"] = dt
        self.moments += ms
        return ms

    def insight(self, mo: dict, mode="analyst", lang="en", prefs=None, agent_down: str | None = None):
        t0 = time.perf_counter()
        ins = render(mo, self.names, self.match_id, mode, lang, prefs)
        if ins is None:
            return None
        t1 = time.perf_counter()
        g = verify(ins)
        t2 = time.perf_counter()
        r_ms, v_ms, f_ms = round((t1 - t0) * 1000, 2), round((t2 - t1) * 1000, 2), mo.get("_feed_ms", 0.0)
        ins["provenance"] = {
            "produced_by": ["IngestAgent", "InterpretAgent", "ExplainTemplate", "PersonaTemplate", "VerifierAgent"],
            "trace": [{"stage": "interpret", "agent": "InterpretAgent", "latency_ms": f_ms},
                      {"stage": "personalize", "agent": "PersonaTemplate", "latency_ms": r_ms},
                      {"stage": "verify", "agent": "VerifierAgent", "latency_ms": v_ms}],
            "latency_ms": {"event_to_insight": round(f_ms + r_ms + v_ms, 2)},
            "model": "deterministic-templates-v0", "confidence": 0.9 if g["status"] == "verified" else 0.4,
            "degraded": g["status"] != "verified", "grounding": g}
        if g["status"] != "verified":
            ins["provenance"]["degraded_reason"] = "grounding check failed: " + ", ".join(g["unbacked_numbers"])
        return ins

    def meta(self) -> dict:
        """Everything the frontend needs besides insights: teams, clock mapping, score timeline, players."""
        goals, players = [], {}
        for m in self.moments:
            for pl in m["players"]:
                if pl["role"] in ("actor", "keeper"):
                    players[pl["player_id"]] = {"id": pl["player_id"], "name": pl["name"], "team": pl["team"]}
            if m["kind"] == "goal":
                goals.append({"stream_s": m["stream_s"], "score": m["facts"]["score"], "team": m["team"], "clock": m["clock"]})
        return {"match_id": self.match_id, "synthetic": True, "order": self.state.order,
                "teams": {c: {"name": t["name"], "short": self.names[c]} for c, t in self.info["teams"].items()},
                "p1_end_s": self.state.p1_end, "duration_s": round(self.state.last_s + 20, 1),
                "goals": goals, "players": sorted(players.values(), key=lambda x: (x["team"], x["name"]))}

    def run_file(self, events, **kw):
        for e in events:
            self.feed(e)
        return [i for m in self.moments if (i := self.insight(m, **kw))]
