"""Stage 1 (Ingest): validate/dedup events and keep running match state."""
from collections import defaultdict


def secs(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


class MatchState:
    def __init__(self, codes: dict[int, str]):
        self.codes = codes                      # {team_id: "KES"}
        self.order = list(codes.values())       # display order: first = "home"
        self.seen: set[str] = set()
        self.p1_end = 0.0
        self.score = {c: 0 for c in self.order}
        self.on_pitch = {c: 11 for c in self.order}
        self.xg = {c: 0.0 for c in self.order}
        self.shots = {c: 0 for c in self.order}
        self.max_deficit = {c: 0 for c in self.order}
        self.player_goals: dict[int, int] = defaultdict(int)
        self.player_goal_types: dict[int, list[str]] = defaultdict(list)
        self.player_shots: dict[int, int] = defaultdict(int)
        self.player_xg: dict[int, float] = defaultdict(float)
        self.passes: list[tuple[float, str]] = []
        self.shot_log: list[tuple[float, str, float]] = []
        self.red: tuple[float, str, str] | None = None   # (stream_s, team code, player)
        self.by_id: dict[str, dict] = {}
        self.keepers: dict[str, tuple[int, str]] = {}
        self.offsides: list[dict] = []                   # {s, team, player, margin_m, var_candidate}
        self.yellows: dict[str, int] = {c: 0 for c in self.order}
        self.var_log: list[dict] = []                    # VAR reviews seen so far
        self.fouls = {c: 0 for c in self.order}
        self.recent: list[dict] = []
        self.last_s = 0.0
        self.counters = {"invalid": 0, "duplicate": 0, "late": 0, "ok": 0}

    def code(self, team: dict) -> str:
        return self.codes[team["id"]]

    def other(self, code: str) -> str:
        return next(c for c in self.order if c != code)

    def stream_s(self, e: dict) -> float:
        return secs(e["timestamp"]) + (self.p1_end if e["period"] >= 2 else 0.0)

    def update(self, e) -> bool:
        """Returns False for invalid/duplicate events (they are counted, not processed)."""
        if not isinstance(e, dict) or not e.get("id") or "type" not in e or "timestamp" not in e:
            self.counters["invalid"] += 1
            return False
        if e["id"] in self.seen:
            self.counters["duplicate"] += 1
            return False
        self.seen.add(e["id"])
        self.by_id[e["id"]] = e
        t = e["type"]["name"]
        if t == "Starting XI":
            for pl in (e.get("tactics") or {}).get("lineup", []):
                if pl["position"]["name"] == "Goalkeeper":
                    self.keepers[self.code(e["team"])] = (pl["player"]["id"], pl["player"]["name"])
        if t == "Half End" and e["period"] == 1:
            self.p1_end = max(self.p1_end, secs(e["timestamp"]))
        s = self.stream_s(e)
        if s < self.last_s - 5:
            self.counters["late"] += 1
        self.last_s = max(self.last_s, s)
        self.counters["ok"] += 1
        if t == "Pass":
            self.passes.append((s, self.code(e["team"])))
        elif t == "Shot":
            c, sh, pid = self.code(e["team"]), e["shot"], e["player"]["id"]
            self.shots[c] += 1
            self.xg[c] += sh["statsbomb_xg"]
            self.player_shots[pid] += 1
            self.player_xg[pid] += sh["statsbomb_xg"]
            self.shot_log.append((s, c, sh["statsbomb_xg"]))
            if sh["outcome"]["name"] == "Goal":
                self.score[c] += 1
                self.player_goals[pid] += 1
                kind = ("penalty" if sh["type"]["name"] == "Penalty"
                        else "header" if sh["body_part"]["name"] == "Head" else "open play")
                self.player_goal_types[pid].append(kind)
                o = self.other(c)
                for team in (c, o):
                    self.max_deficit[team] = max(self.max_deficit[team], self.score[self.other(team)] - self.score[team])
        elif t == "Foul Committed" and (e.get("foul_committed") or {}).get("card", {}).get("name") == "Red Card":
            c = self.code(e["team"])
            self.on_pitch[c] -= 1
            self.red = (s, c, e["player"]["name"])
            self.fouls[c] += 1
        elif t == "Foul Committed":
            c = self.code(e["team"])
            self.fouls[c] += 1
            if (e.get("foul_committed") or {}).get("card", {}).get("name") == "Yellow Card":
                self.yellows[c] += 1
        elif t == "Offside":
            self.offsides.append({"s": s, "team": self.code(e["team"]), "player": (e.get("player") or {}).get("name"),
                                  "margin_m": e.get("x_margin_m"), "var_candidate": bool(e.get("x_var_candidate")), "event_id": e["id"]})
        elif t == "VAR":
            v = e.get("var") or {}
            self.var_log.append({"s": s, "team": self.code(e["team"]), "review": v.get("review"), "outcome": (v.get("outcome") or {}).get("name"),
                                 "subject": v.get("subject_event_id"), "event_id": e["id"]})
        self.recent.append(e)
        del self.recent[:-40]
        return True

    # ---- windowed stats ----
    def pass_share(self, code: str, lo: float, hi: float, min_passes: int = 15):
        w = [c for s, c in self.passes if lo < s <= hi]
        if len(w) < min_passes:
            return None
        return round(100 * sum(1 for c in w if c == code) / len(w), 1)

    def xg_between(self, code: str, lo: float, hi: float) -> float:
        return round(sum(x for s, c, x in self.shot_log if c == code and lo < s <= hi), 2)

    def offsides_against(self, code: str, lo: float, hi: float) -> int:
        return sum(1 for o in self.offsides if o["team"] == code and lo < o["s"] <= hi)

    def score_str(self, score: dict | None = None) -> str:
        sc = score or self.score
        return "-".join(str(sc[c]) for c in self.order)
