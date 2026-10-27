"""Match simulator: 22 players + ball on a 0.5 s step (2 Hz tracking), driven by the decision model, with scripted beats.
Deterministic: all randomness comes from one seeded random.Random; iteration order is fixed.
Global frame: KES attacks x=120, IRO attacks x=0. Event locations are written in the acting team's own attacking frame."""
import math, random, uuid, copy
from .model import (L, W, GOAL, att, hyp, xg_shot, pressure_at, seg_dist, offside_margin_att, lane_blockers, BOX_X, BOX_Y0, BOX_Y1)
from .decision import decide, plan_pass, flight_steps
from .roster import KES, KES_SUBS, IRO, IRO_SUBS, POS_ID, SLOT, role_of
from .beats import BEATS, SUBS

DT = 0.5
TEAMS = ("KES", "IRO")
DIR = {"KES": 1, "IRO": -1}
NAME = {"KES": "Kestrel Rovers", "IRO": "Ironvale Athletic"}
TID = {"KES": 9001, "IRO": 9002}
OPP = {"KES": "IRO", "IRO": "KES"}
TYPE_ID = {"Pass": 30, "Ball Receipt*": 42, "Carry": 43, "Pressure": 17, "Ball Recovery": 2, "Interception": 10, "Shot": 16, "Goal Keeper": 23,
           "Duel": 4, "Dribble": 14, "Dribbled Past": 39, "Miscontrol": 38, "Clearance": 9, "Foul Committed": 22, "Foul Won": 21,
           "Dispossessed": 3, "Block": 6, "Substitution": 19, "Tactical Shift": 36, "Half Start": 18, "Half End": 34, "Starting XI": 35,
           "Offside": 8, "Free Kick Awarded": 101, "VAR": 102, "Injury Stoppage": 40}
PATTERN = {"Regular Play": 1, "From Corner": 2, "From Free Kick": 3, "From Throw In": 4, "From Goal Kick": 7, "From Counter": 6, "From Kick Off": 9}
SHOT_OUT = {"Goal": 97, "Saved": 100, "Off T": 98, "Post": 99, "Blocked": 96, "Wayward": 102, "Goal Disallowed": 110}

# ---- tunables (see README "Decision model & simulator") ----
TEAM = {
    "KES": dict(pass_skill=1.0, fwd=0.9, temp=0.006, patience=1.0, shot_mult=1.0, direct=0, line=35, press=0.6, counter_p=0.25, dribble=1.0),
    "IRO": dict(pass_skill=-0.35, fwd=1.1, temp=0.006, patience=0.4, shot_mult=1.5, direct=1, line=28, press=0.55, counter_p=0.95, dribble=0.9),
}
SIGMA_OFF = 0.6        # passer's perception noise on the receiver's offside margin (m)
TRAP_P = 0.03           # chance the defensive line steps up as the pass is played
TRAP_STEP = 2.2
FOUL_RATE = 0.08
YELLOW_P = 0.12


class Pl:
    __slots__ = ("id", "team", "name", "pos", "no", "x", "y", "vx", "vy", "on", "role", "vmax", "run", "hold", "yc", "sprint", "cap")

    def __init__(self, tup, team, rng):
        self.id, self.name, self.pos, self.no = tup
        self.team, self.x, self.y, self.vx, self.vy, self.on = team, 0.0, 0.0, 0.0, 0.0, False
        self.role = role_of(self.pos)
        self.vmax = {"GK": 5.5, "DEF": 7.1, "MID": 7.4, "ATT": 7.9}[self.role] + rng.uniform(-0.25, 0.35)
        self.run, self.hold, self.yc, self.sprint, self.cap = 0, 1.0, 0, False, None


class World:
    def __init__(self, seed=2026):
        self.seed = seed
        self.rng = random.Random(seed)
        self.events, self.frames, self.decisions, self.beat_log = [], [], [], {}
        self.pl = {}
        for team, xi, subs in (("KES", KES, KES_SUBS), ("IRO", IRO, IRO_SUBS)):
            for tup in xi + subs:
                self.pl[tup[0]] = Pl(tup, team, self.rng)
        self.on = {t: [] for t in TEAMS}
        self.lineup = {t: [] for t in TEAMS}
        self.per, self.t = 1, 0.0
        self.state = "dead"            # dead | carried | flight
        self.carrier, self.fl, self.dead = None, None, None
        self.poss_team, self.poss_no, self.pattern = "KES", 0, "From Kick Off"
        self.next_t, self.counter_until, self.ft_since = 0.0, -1.0, {"KES": None, "IRO": None}
        self.press_cd, self.last_pass, self.var_until = {}, None, 0.0
        self.score = {"KES": 0, "IRO": 0}
        self.beats = [dict(b, done=False, armed=False, force=False, t_force=None) for b in BEATS]
        self.beat = None
        self.subs = [dict(s, done=False) for s in SUBS]
        self.p1_end = None
        self.stats = {"offsides": 0, "fouls": 0}
        self.pending_hook = None
        self.possession_time = {"KES": 0.0, "IRO": 0.0}
        self.ball = [60.0, 40.0, 0.0]
        self.red_done = False
        self.first_touch, self.ctx_header, self.last_pass, self.carry_goal, self.last_foul = False, False, None, None, None
        self.P_END = {1: 2790.0, 2: 3500.0}
        self.finish_t = {}
        self.last_choice = {}

    # ------------------------------------------------------------------ helpers
    def A(self, team, x, y):
        return att(DIR[team], x, y)

    def stream_t(self):
        return self.t + (self.p1_end if (self.per == 2 and self.p1_end is not None) else 0.0)

    def minute_second(self):
        m = int(self.t // 60) + (45 if self.per == 2 else 0)
        return m, int(self.t % 60)

    def ball_pos(self):
        if self.state == "flight":
            f = self.fl
            u = min(1.0, max(0.0, (self.t - f["t0"]) / max(DT, f["t1"] - f["t0"])))
            z = (3.2 * math.sin(math.pi * u) if f["high"] else 0.0) if f["kind"] == "pass" else (1.0 * u if f["kind"] == "shot" else 0.0)
            return [f["p0"][0] + (f["p1"][0] - f["p0"][0]) * u, f["p0"][1] + (f["p1"][1] - f["p0"][1]) * u, z]
        if self.carrier is not None and self.state == "carried":
            return [self.carrier.x, self.carrier.y, 0.0]
        return list(self.ball)

    def pdict(self, p, team):
        x, y = self.A(team, p.x, p.y)
        d = DIR[team]
        return {"id": p.id, "x": x, "y": y, "vx": p.vx * d, "vy": p.vy * d, "role": p.role, "pos": p.pos}

    # ------------------------------------------------------------------ events
    def emit(self, typ, team, p=None, loc=None, duration=0.0, **extra):
        idx = len(self.events) + 1
        m, s = self.minute_second()
        ts = self.t
        e = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"synthetic-{self.seed}-{idx}")), "index": idx, "period": self.per,
             "timestamp": f"{int(ts // 3600):02d}:{int(ts % 3600 // 60):02d}:{ts % 60:06.3f}", "minute": m, "second": s,
             "type": {"id": TYPE_ID[typ], "name": typ}, "possession": self.poss_no,
             "possession_team": {"id": TID[self.poss_team], "name": NAME[self.poss_team]},
             "play_pattern": {"id": PATTERN[self.pattern], "name": self.pattern}, "team": {"id": TID[team], "name": NAME[team]}}
        if p is not None:
            e["player"] = {"id": p.id, "name": p.name}
            e["position"] = {"id": POS_ID[p.pos], "name": p.pos}
        if loc is not None:
            x, y = self.A(team, loc[0], loc[1])
            e["location"] = [round(x, 1), round(y, 1)]
        e["duration"] = round(duration, 3)
        e.update(extra)
        self.events.append(e)
        return e

    def own(self, team, x, y):
        a, b = self.A(team, x, y)
        return [round(a, 1), round(b, 1)]

    def freeze(self, team, exclude=None):
        out = []
        for tm in TEAMS:
            for p in self.on[tm]:
                if p.id == exclude:
                    continue
                out.append({"location": self.own(team, p.x, p.y), "player": {"id": p.id, "name": p.name},
                            "position": {"id": POS_ID[p.pos], "name": p.pos}, "teammate": tm == team})
        return out

    # ------------------------------------------------------------------ lineups
    def set_lineup(self, team, ids, positions=None):
        for pid in self.lineup[team]:
            self.pl[pid].on = False
        self.lineup[team] = list(ids)
        for pid in ids:
            self.pl[pid].on = True
            if positions and pid in positions:
                self.pl[pid].pos = positions[pid]
        self.on[team] = [self.pl[i] for i in ids]

    def tactics_event(self, team, typ="Tactical Shift"):
        lineup = [{"player": {"id": p.id, "name": p.name}, "position": {"id": POS_ID[p.pos], "name": p.pos}, "jersey_number": p.no} for p in self.on[team]]
        form = {"KES": 4231, "IRO": 442 if len(self.on[team]) == 11 and not self.red_done else 441}[team]
        self.emit(typ, team, tactics={"formation": form, "lineup": lineup})

    def place_kickoff(self, kick_team):
        for team in TEAMS:
            for p in self.on[team]:
                sx, sy = SLOT[p.pos]
                sx = min(sx, 56 if p.role != "ATT" else 50)
                if p.role == "ATT" and p.pos == "Center Forward":
                    sx, sy = 52, 40
                p.x, p.y = self.A(team, sx, sy)
                p.vx = p.vy = 0.0
        k = None
        for p in self.on[kick_team]:
            if p.pos in ("Center Forward", "Right Center Forward"):
                k = p
        k = k or self.on[kick_team][-1]
        k.x, k.y = 60.0, 40.0
        self.ball = [60.0, 40.0, 0.0]

    # ------------------------------------------------------------------ possession control
    def new_possession(self, team, how):
        if team != self.poss_team or how != "same":
            self.poss_no += 1
        self.poss_team = team
        self.ft_since = {"KES": None, "IRO": None}
        cfg = TEAM[team]
        opp = OPP[team]
        pat = {"throw": "From Throw In", "corner": "From Corner", "fk": "From Free Kick", "gk": "From Goal Kick", "ko": "From Kick Off"}.get(how)
        if pat is None:
            adv = sum(self.A(opp, p.x, p.y)[0] for p in self.on[opp] if p.role != "GK") / max(1, len(self.on[opp]) - 1)
            bx = self.A(team, *self.ball_pos()[:2])[0]
            beatc = self.beat and self.beat["team"] == team and self.beat["kind"] == "counter"
            if (how in ("tackle", "intercept", "recovery", "gkcatch") and adv > 52 and bx < 78 and self.rng.random() < cfg["counter_p"]) or beatc:
                pat = "From Counter"
                self.counter_until = self.t + 14.0
            else:
                pat = "Regular Play"
        self.pattern = pat

    def give_ball(self, p, hold=None):
        self.carrier, self.state = p, "carried"
        self.ball = [p.x, p.y, 0.0]
        if hold is None:
            hold = 1 if (self.counter_until > self.t) else self.rng.choice((1, 2, 2, 3))
        self.next_t = self.t + hold * DT

    def set_dead(self, kind, team, loc, delay, how=None):
        loc = (min(L, max(0.0, loc[0])), min(W, max(0.0, loc[1])))
        vb = self.ball_pos()
        self.dead = {"kind": kind, "team": team, "loc": loc, "ready": self.t + delay, "how": how or kind, "vfrom": (vb[0], vb[1]), "vt0": self.t}
        self.state, self.carrier, self.fl = "dead", None, None
        self.dead["taker"] = self.pick_taker(kind, team, loc).id
        self.ball = [loc[0], loc[1], 0.0]

    # ------------------------------------------------------------------ one tick of the clock
    def step(self):
        self.process()
        self.record()
        self.advance()

    def process(self):
        t = self.t
        self.check_subs()
        self.check_beats()
        if t < self.var_until:
            return
        if self.state == "dead":
            if t >= self.dead["ready"] - 1e-9:
                self.run_restart()
        elif self.state == "flight":
            if t >= self.fl["t1"] - 1e-9:
                self.land()
        if self.state == "carried" and t >= self.next_t - 1e-9:
            self.tick()
        if self.state in ("carried",):
            self.possession_time[self.poss_team] += DT
        elif self.state == "flight":
            self.possession_time[self.fl["team"]] += DT

    # ------------------------------------------------------------------ subs / beats arming
    def check_subs(self):
        if self.state != "dead" or self.t < self.var_until:
            return
        for s in self.subs:
            if not s["done"] and s["per"] == self.per and self.t >= s["t"]:
                s["done"] = True
                o, i = self.pl[s["out"]], self.pl[s["inn"]]
                if not o.on:
                    continue
                i.x, i.y, i.pos = o.x, o.y, o.pos
                ids = [i.id if q == o.id else q for q in self.lineup[s["team"]]]
                self.emit("Substitution", s["team"], o, loc=None, substitution={"outcome": {"id": 103, "name": "Tactical"},
                          "replacement": {"id": i.id, "name": i.name}})
                self.set_lineup(s["team"], ids)
                self.tactics_event(s["team"])
                self.dead["ready"] = max(self.dead["ready"], self.t + 4)

    def check_beats(self):
        if self.beat is None:
            for b in self.beats:
                if not b["done"] and not b["armed"] and b["per"] == self.per and self.t >= b["arm"] and self.t >= self.var_until and self.state in ("carried", "dead"):
                    if self.state == "dead" and self.dead["kind"] in ("kickoff",):
                        continue
                    self.start_beat(b)
                    break
        elif not self.beat["force"] and self.t >= self.beat["t_force"]:
            self.beat["force"] = True
        if self.beat and self.t > self.beat["arm"] + 110 and not self.beat["done"]:      # safety net: abandon a beat that never resolved
            self.beat_log.setdefault("abandoned", []).append(self.beat["key"])
            self.beat["done"] = True
            self.beat = None

    def start_beat(self, b):
        b["armed"] = True
        b["t_force"] = self.t + b["build"]
        self.beat = b
        team = b["team"]
        if b["kind"] == "corner_goal":
            cx = 120 if DIR[team] > 0 else 0
            cy = 0.5 if self.rng.random() < 0.5 else 79.5
            opp = OPP[team]
            c = min((p for p in self.on[opp] if p.role == "DEF"), key=lambda p: hyp(p.x, p.y, cx, cy))
            self.emit("Clearance", opp, c, loc=(c.x, c.y), clearance={"body_part": {"id": 37, "name": "Head"}})
            b["force"] = True
            self.set_dead("corner", team, (cx, cy), 7.0, "corner")
            return
        if self.state == "dead":                         # a dead ball for the other team: let it be played, beat still forces turnover later
            return
        if self.poss_team != team and self.state == "carried":
            return
        self.force_turnover(team)

    def force_turnover(self, team):
        """The other team's carrier plays a poor pass that the nearest `team` player reads (normal flight + interception)."""
        c = self.carrier
        if c is None or c.team == team:
            return
        cands = [p for p in self.on[team] if p.role != "GK"]
        near = min(cands, key=lambda p: hyp(p.x, p.y, c.x, c.y))
        ex, ey = near.x + (c.x - near.x) * 0.15, near.y + (c.y - near.y) * 0.15
        d = hyp(c.x, c.y, ex, ey)
        steps = max(2, flight_steps(d, False))
        self.start_flight("pass", c, None, (c.x, c.y), (ex, ey), steps, high=False, outcome="Incomplete", interceptor=near,
                          plan={"kind": "pass", "pc": 0.0, "margin": -9.0, "line": 0.0, "to": None, "dist": d})

    # ------------------------------------------------------------------ flights
    def start_flight(self, kind, passer, recv, p0, p1, steps, high, outcome, interceptor=None, plan=None, ptype=None, tech=None, cross=False):
        team = passer.team
        d = hyp(p0[0], p0[1], p1[0], p1[1])
        o0, o1 = self.A(team, *p0), self.A(team, *p1)
        pc = plan.get("pc", 0.8) if plan else 0.8
        pdict = {"length": round(d, 2), "angle": round(math.atan2(o1[1] - o0[1], o1[0] - o0[0]), 3),
                 "height": {"id": 3 if high else 1, "name": "High Pass" if high else "Ground Pass"},
                 "end_location": [round(o1[0], 1), round(o1[1], 1)], "body_part": {"id": 40 if passer.role != "GK" else 68, "name": "Right Foot" if passer.id % 3 else "Left Foot"},
                 "x_difficulty": round(1 - pc, 2), "x_ball_speed_kmh": round(d / (steps * DT) * 3.6, 1)}
        if recv is not None:
            pdict["recipient"] = {"id": recv.id, "name": recv.name}
        if ptype:
            pdict["type"] = {"id": {"Kick Off": 65, "Throw-in": 67, "Goal Kick": 63, "Corner": 61, "Free Kick": 62}[ptype], "name": ptype}
        if outcome != "Complete":
            pdict["outcome"] = {"id": {"Incomplete": 9, "Out": 75, "Pass Offside": 76}[outcome], "name": outcome}
        if tech:
            pdict["technique"] = {"id": 104, "name": tech}
        if cross:
            pdict["cross"] = True
        if plan and plan.get("margin") is not None and recv is not None and self.A(team, *p0)[0] > 0 and plan.get("recv_x", 0) > 60:
            pdict["x_offside_check"] = {"receiver_margin_m": round(plan["margin_eff"], 2), "offside_line_x": round(plan["line"], 1),
                                       "receiver_offside": plan["margin_eff"] > 0 and not plan.get("exempt"),
                                       **({"exempt_restart": plan["exempt"]} if plan.get("exempt") else {})}
        under = plan.get("press_me", 0) > 0.45 if plan else False
        extra = {"under_pressure": True} if under else {}
        e = self.emit("Pass", team, passer, loc=p0, duration=steps * DT, **{"pass": pdict}, **extra)
        self.fl = dict(kind=kind, team=team, passer=passer, recv=recv, p0=p0, p1=p1, t0=self.t, t1=self.t + steps * DT, high=high,
                       outcome=outcome, interceptor=interceptor, plan=plan or {}, ptype=ptype, ev=e, cross=cross)
        self.last_pass = dict(ev=e, recv=recv.id if recv else None, t=self.t, poss=self.poss_no, margin=(plan or {}).get("margin_eff"))
        self.state, self.carrier = "flight", None
        return e

    def land(self):
        fl = self.fl
        self.ball = [fl["p1"][0], fl["p1"][1], 0.0]
        if fl["kind"] == "shot":
            return self.land_shot(fl)
        if fl["kind"] == "clear":
            return self.land_clear(fl)
        team, o = fl["team"], fl["outcome"]
        opp = OPP[team]
        pe = fl["ev"]
        if o in ("Complete", "Pass Offside"):
            r = fl["recv"]
            if hyp(r.x, r.y, *fl["p1"]) > 1.2:                       # keep the event consistent with the tracking: end where he actually is
                fl["p1"] = (r.x + (fl["p1"][0] - r.x) * 0.25, r.y + (fl["p1"][1] - r.y) * 0.25)
                pe["pass"]["end_location"] = self.own(team, *fl["p1"])
            r.x, r.y = fl["p1"]
            if o == "Pass Offside":
                self.stats["offsides"] += 1
                oe = self.emit("Offside", team, r, loc=(r.x, r.y), related_events=[pe["id"]], x_margin_m=round(fl["plan"].get("margin_eff", 0), 2),
                               x_var_candidate=abs(fl["plan"].get("margin_eff", 9)) < 0.5)
                pe["related_events"] = [oe["id"]]
                self.emit("Free Kick Awarded", opp, None, loc=(r.x, r.y), reason="Offside", related_events=[oe["id"]])
                self.set_dead("free_kick", opp, (r.x, r.y), self.rng.uniform(7, 11), "fk")
                return
            re_ = self.emit("Ball Receipt*", team, r, loc=(r.x, r.y), related_events=[pe["id"]])
            pe["related_events"] = [re_["id"]]
            self.give_ball(r, hold=0 if (fl["cross"] or fl["ptype"] == "Corner") else None)
            self.ctx_header = bool(fl["cross"] or fl["ptype"] == "Corner")
            self.first_touch = True
            return
        if o == "Out":
            return self.out_of_play(team, fl["p1"], by="pass")
        # Incomplete -> opponent wins it
        it = fl["interceptor"]
        it.x, it.y = fl["p1"]
        it.vx = it.vy = 0.0
        if fl["ptype"] == "Corner":
            self.emit("Clearance", opp, it, loc=(it.x, it.y), clearance={"body_part": {"id": 37, "name": "Head"}})
            return self.clear_ball(it, aimless=True)
        if self.rng.random() < 0.62:
            ev = self.emit("Interception", opp, it, loc=(it.x, it.y), interception={"outcome": {"id": 4, "name": "Won"}}, related_events=[pe["id"]])
            how = "intercept"
        else:
            self.emit("Ball Recovery", opp, it, loc=(it.x, it.y))
            how = "recovery"
        self.new_possession(opp, how)
        self.give_ball(it)

    def out_of_play(self, team, p1, by="pass"):
        """team = team that put the ball out. p1 = global ball location (may be inside the pitch: we project outwards)."""
        opp = OPP[team]
        x, y = p1
        ax, ay = self.A(team, x, y)
        if ay <= 1.5 or ay >= W - 1.5 or (3 < ax < 117):
            by_ = 0.0 if ay < 40 else W
            gx, gy = self.A(team, min(118.0, max(2.0, ax)), by_)
            return self.set_dead("throw", opp, (gx, gy), self.rng.uniform(7, 11), "throw")
        if ax >= 117:                                         # over the opponent end line -> goal kick for the defenders
            gx, gy = self.A(team, 114.0, 40.0)
            return self.set_dead("goal_kick", opp, self.A(team, 114.0, 40.0), self.rng.uniform(9, 14), "gk")
        cx, cy = self.A(team, 0.5, 0.5 if ay < 40 else 79.5)  # over own end line -> corner for the opponent
        self.set_dead("corner", opp, (cx, cy) if False else self.A(team, 0.5, 0.5 if ay < 40 else 79.5), self.rng.uniform(12, 17), "corner")

    def clear_ball(self, p, aimless=False):
        team = p.team
        ax, ay = self.A(team, p.x, p.y)
        dx = self.rng.uniform(22, 38)
        ex = min(112.0, ax + dx) if not aimless else min(100.0, max(6.0, ax + self.rng.uniform(-6, 24)))
        ey = min(78.0, max(2.0, ay + self.rng.uniform(-22, 22)))
        gx, gy = self.A(team, ex, ey)
        out = self.rng.random() < 0.42
        if out:
            gy = 0.0 if self.rng.random() < 0.5 else W
        d = hyp(p.x, p.y, gx, gy)
        steps = flight_steps(d, True)
        self.fl = dict(kind="clear", team=team, passer=p, p0=(p.x, p.y), p1=(gx, gy), t0=self.t, t1=self.t + steps * DT, high=True, out=out, plan={}, ev=None, recv=None)
        self.state, self.carrier = "flight", None

    def land_clear(self, fl):
        team, p1 = fl["team"], fl["p1"]
        if fl["out"]:
            return self.out_of_play(team, p1)
        cands = sorted((q for tm in TEAMS for q in self.on[tm] if q.role != "GK"), key=lambda q: hyp(q.x, q.y, p1[0], p1[1]))[:3]
        w = self.rng.choices(cands, weights=[3.0 if q.team == OPP[team] else 2.0 for q in cands])[0]
        if hyp(w.x, w.y, *p1) > 3.0:                                  # ball ends where he actually reached it
            w.x, w.y = w.x + (p1[0] - w.x) * 0.3, w.y + (p1[1] - w.y) * 0.3
        w.vx = w.vy = 0.0
        self.ball = [w.x, w.y, 0.0]
        if w.team != team:
            self.emit("Ball Recovery", w.team, w, loc=(w.x, w.y))
        self.new_possession(w.team, "recovery")
        self.give_ball(w)

    # ------------------------------------------------------------------ decisions
    def team_cfg(self, team):
        cfg = dict(TEAM[team])
        if team == "KES":
            if self.per == 2:
                cfg.update(line=38, press=0.7, patience=1.3)
            if self.red_done:                                   # a man up: pin Ironvale back
                cfg.update(line=46, press=0.9, patience=2.2, shot_mult=1.15, pass_skill=cfg["pass_skill"] + 0.3)
        else:
            if self.red_done:
                cfg.update(line=20, press=0.45, counter_p=0.6, fwd=1.2, pass_skill=cfg["pass_skill"] - 0.5)
        return cfg

    def build_ctx(self, c):
        team = c.team
        cfg = self.team_cfg(team)
        fts = self.ft_since[team]
        ctx = dict(pass_skill=cfg["pass_skill"], fwd=cfg["fwd"], sigma_off=SIGMA_OFF, temp=cfg["temp"], counter=self.counter_until > self.t and self.poss_team == team,
                   direct=cfg["direct"], patience_s=(self.t - fts) if fts is not None else 0.0, patience=cfg["patience"], shot_mult=cfg["shot_mult"],
                   dribble=cfg["dribble"], carried_last=(self.last_choice.get(c.id) == "carry"), first_time=self.first_touch, header=getattr(self, "ctx_header", False))
        if ctx["counter"]:                       # transition: opponent is stretched, space behind the line
            ctx.update(pass_skill=ctx["pass_skill"] + 0.6)
            if team == "IRO":                    # Ironvale are built to counter; Kestrel only rarely get this kind of space
                ctx.update(box_entry=0.95, box_carry_risk=0.08, shot_mult=ctx["shot_mult"] * 1.2, shot_floor=0.03)
        b = self.beat
        if b and b["team"] == team and b["kind"] != "counter_none":
            ctx["beat"] = {"shooter": b["shooter"], "min_xg": b.get("min_xg"), "shoot_now": b["kind"] == "corner_goal", "cap_x": 101.0 if b["kind"] == "box_edge_foul" else 118.0}
            if b["kind"] in ("foul_red", "foul_pen", "box_edge_foul"):
                ctx["beat"]["min_xg"] = 9.9          # never shoots: the beat ends in a foul
            if b["kind"] == "disallowed":
                ctx["no_offside_filter"] = False
        return ctx

    def views(self, c):
        team, opp = c.team, OPP[c.team]
        me = self.pdict(c, team)
        mates = [self.pdict(p, team) for p in self.on[team] if p is not c]
        opps = []
        for p in self.on[opp]:
            x, y = self.A(team, p.x, p.y)
            opps.append((x, y, p.role == "GK"))
        return me, mates, opps

    def tick(self):
        c = self.carrier
        team, opp = c.team, OPP[c.team]
        self.press_events(c)
        b = self.beat if (self.beat and self.beat["team"] == team) else None
        protect = bool(b and b["force"])
        ax, ay = self.A(team, c.x, c.y)
        if ax > 66 and self.ft_since[team] is None:
            self.ft_since[team] = self.t
        if b and protect and self.beat_tick(c, b):
            return
        if not protect and self.contest(c):
            return
        ctx = self.build_ctx(c)
        me, mates, opps = self.views(c)
        choice, top3, notes = decide(ctx, me, mates, opps, self.rng)
        self.decisions.append(dict(t=round(self.t, 1), per=self.per, team=team, player=c.id, chosen=choice["action"], to=choice.get("to"),
                                   top3=top3, beat=bool(ctx.get("beat")), **notes, x=round(ax, 1), y=round(ay, 1)))
        self.first_touch = False
        self.last_choice = {c.id: choice["action"]}
        self.exec_choice(c, choice, ctx, notes)

    def exec_choice(self, c, ch, ctx, notes):
        a = ch["action"]
        b = self.beat
        if a == "shoot" and b and b["team"] == c.team and b["shooter"] == c.id and self.t < b["window"][0]:
            self.next_t = self.t + DT                    # the scripted moment waits for its minute (he keeps the ball)
            return
        if a == "shoot":
            self.do_shoot(c, ch["xg"], ch.get("blockers", 0), notes)
        elif a == "pass":
            self.do_pass(c, ch)
        elif a == "carry":
            self.do_carry(c, ch["end"])
        elif a == "dribble":
            self.do_dribble(c, ch)
        elif a == "clear":
            self.emit("Clearance", c.team, c, loc=(c.x, c.y), clearance={"body_part": {"id": 40, "name": "Right Foot"}})
            self.clear_ball(c)
        else:
            self.next_t = self.t + DT * self.rng.choice((1, 2))
            c.sprint = False
        self.ctx_header = False

    def press_events(self, c):
        opp = OPP[c.team]
        near = min((p for p in self.on[opp] if p.role != "GK"), key=lambda p: hyp(p.x, p.y, c.x, c.y))
        if hyp(near.x, near.y, c.x, c.y) < 3.4 and self.press_cd.get(near.id, -9) < self.t - 2.5:
            self.press_cd[near.id] = self.t
            lvl = round(min(1.0, max(0.1, 1 - hyp(near.x, near.y, c.x, c.y) / 4.0)), 2)
            self.emit("Pressure", opp, near, loc=(near.x, near.y), duration=self.rng.uniform(1.0, 3.5), x_pressure_level=lvl)

    def in_box(self, team_att, x, y):
        """is global (x,y) inside the box that `team_att` is attacking?"""
        ax, ay = self.A(team_att, x, y)
        return ax >= BOX_X and BOX_Y0 <= ay <= BOX_Y1

    def contest(self, c):
        team, opp = c.team, OPP[c.team]
        near = min((p for p in self.on[opp] if p.role != "GK"), key=lambda p: hyp(p.x, p.y, c.x, c.y))
        d = hyp(near.x, near.y, c.x, c.y)
        if d > 2.8:
            return False
        press = self.team_cfg(opp)["press"]
        ax_ = self.A(team, c.x, c.y)[0]
        zone = 1.0 + 2.2 * max(0.0, min(1.0, (ax_ - 72) / 28.0))      # defenders are tighter near their own box
        if self.rng.random() > min(0.9, (1 - d / 2.8) * 0.55 * press * 1.25 * zone):
            return False
        r = self.rng.random()
        boxy = self.in_box(team, c.x, c.y)
        if r < FOUL_RATE and not boxy:
            return self.foul(near, c)
        if r < 0.55:                                  # tackle won
            self.emit("Dispossessed", team, c, loc=(c.x, c.y))
            self.emit("Duel", opp, near, loc=(near.x, near.y), duel={"type": {"id": 11, "name": "Tackle"}, "outcome": {"id": 4, "name": "Won"}})
            self.new_possession(opp, "tackle")
            self.give_ball(near, hold=1)
            return True
        self.emit("Dribble", team, c, loc=(c.x, c.y), dribble={"outcome": {"id": 8, "name": "Complete"}})
        self.emit("Dribbled Past", opp, near, loc=(near.x, near.y))
        c.vx = c.vy = 0.0
        self.next_t = self.t + DT
        return True

    def foul(self, fouler, victim, scripted=None):
        team, opp = victim.team, fouler.team
        self.stats["fouls"] += 1
        card = None
        if scripted == "red":
            card = {"id": 5, "name": "Red Card"}
        elif fouler.yc == 0 and self.rng.random() < YELLOW_P:
            card = {"id": 7, "name": "Yellow Card"}
            fouler.yc = 1
        fe = self.emit("Foul Committed", opp, fouler, loc=(fouler.x, fouler.y), foul_committed=({"card": card} if card else {}))
        we = self.emit("Foul Won", team, victim, loc=(victim.x, victim.y), foul_won={}, related_events=[fe["id"]])
        fe["related_events"] = [we["id"]]
        self.last_foul = (fe, we, card)
        if scripted in ("red", "pen", "box_edge"):
            return fe
        self.emit("Free Kick Awarded", team, None, loc=(victim.x, victim.y), reason="Foul", related_events=[fe["id"]])
        self.set_dead("free_kick", team, (victim.x, victim.y), self.rng.uniform(9, 15), "fk")
        return True

    def do_carry(self, c, end):
        team = c.team
        gx, gy = self.A(team, *end)
        d = hyp(c.x, c.y, gx, gy)
        steps = max(1, math.ceil(d / (5.5 * DT)))
        self.emit("Carry", team, c, loc=(c.x, c.y), duration=steps * DT, carry={"end_location": self.own(team, gx, gy), "x_speed_kmh": round(d / (steps * DT) * 3.6, 1)})
        c.sprint, c.cap = True, d / (steps * DT)
        self.carry_goal = (c, (gx, gy))
        self.next_t = self.t + steps * DT
        self.ball = [c.x, c.y, 0.0]

    def do_dribble(self, c, ch):
        team, opp = c.team, OPP[c.team]
        near = min((p for p in self.on[opp] if p.role != "GK"), key=lambda p: hyp(p.x, p.y, c.x, c.y))
        if self.rng.random() < ch["p_beat"] or (self.beat and self.beat["team"] == team and self.beat["force"]):
            self.emit("Dribble", team, c, loc=(c.x, c.y), dribble={"outcome": {"id": 8, "name": "Complete"}})
            self.emit("Dribbled Past", opp, near, loc=(near.x, near.y))
            self.do_carry(c, ch["end"])
        else:
            self.emit("Dribble", team, c, loc=(c.x, c.y), dribble={"outcome": {"id": 9, "name": "Incomplete"}})
            self.emit("Duel", opp, near, loc=(near.x, near.y), duel={"type": {"id": 11, "name": "Tackle"}, "outcome": {"id": 4, "name": "Won"}})
            self.new_possession(opp, "tackle")
            self.give_ball(near, hold=1)

    # ------------------------------------------------------------------ passing
    def do_pass(self, c, ch, ptype=None, force_complete=False, plan_override=None):
        team, opp = c.team, OPP[c.team]
        rng = self.rng
        r = self.pl[ch["to"]]
        plan = dict(ch)
        ex, ey = ch["end"]
        p1 = self.A(team, ex, ey)
        margin = ch["margin"]
        trapped = rng.random() < TRAP_P and ptype is None
        plan["margin_eff"] = margin
        if trapped:                                    # offside trap: the back line steps up AS the ball is played (visible in the tracking frame)
            for q in self.on[opp]:
                if q.role == "DEF":
                    qx, qy = self.A(team, q.x, q.y)
                    q.x, q.y = self.A(team, max(1.0, qx - TRAP_STEP), qy)
                    q.vx = q.vy = 0.0
            def_xs = [self.A(team, q.x, q.y)[0] for q in self.on[opp]]
            plan["margin_eff"], plan["line"] = offside_margin_att(self.A(team, r.x, r.y)[0], self.A(team, c.x, c.y)[0], def_xs)
        plan["recv_x"] = self.A(team, r.x, r.y)[0]
        plan["press_me"] = 0.0
        exempt = ptype in ("Throw-in", "Goal Kick", "Corner")
        plan["exempt"] = ptype if exempt else None
        d = hyp(c.x, c.y, p1[0], p1[1])
        steps = ch["steps"]
        outcome, it = "Complete", None
        protect = force_complete or (self.beat and self.beat["team"] == team and self.beat["force"])
        if plan["margin_eff"] > 0 and plan["recv_x"] > 60 and not exempt and not (self.beat and self.beat["kind"] == "disallowed" and protect):
            outcome = "Pass Offside"
        elif self.beat and self.beat["kind"] == "disallowed" and protect and plan["margin_eff"] > 0:
            outcome = "Complete"                       # undetected on the field: VAR will rule it out later
        elif not protect and rng.random() > ch["pc"]:
            out_p = 0.10 if (ch["high"] or d > 30) else 0.025
            if rng.random() < out_p:
                outcome = "Out"
                p1 = self.ray_out(c, p1)
                steps = max(1, math.ceil(hyp(c.x, c.y, *p1) / ((22.0 if ch["high"] else 14.0) * DT)))
            else:
                outcome = "Incomplete"
                opps = sorted(self.on[opp], key=lambda q: seg_dist(q.x, q.y, c.x, c.y, p1[0], p1[1])[0])
                it = opps[0]
                dd, u = seg_dist(it.x, it.y, c.x, c.y, p1[0], p1[1])
                u = min(0.95, max(0.4, u))
                q = (c.x + (p1[0] - c.x) * u, c.y + (p1[1] - c.y) * u)
                steps = max(2, math.ceil(hyp(c.x, c.y, *q) / ((22.0 if ch["high"] else 14.0) * DT)))
                if hyp(it.x, it.y, *q) > 6.5 * steps * DT * 0.85 + 1.5:      # nobody can reach it in time: the ball runs out instead
                    outcome, it = "Out", None
                    q = self.ray_out(c, p1)
                    steps = max(1, math.ceil(hyp(c.x, c.y, *q) / ((22.0 if ch["high"] else 14.0) * DT)))
                p1 = q
        tech = "Through Ball" if ch["kind"] == "through" else None
        self.start_flight("pass", c, r if outcome != "Incomplete" else (r if rng.random() < .5 else None), (c.x, c.y), p1, steps, ch["high"], outcome, interceptor=it,
                          plan=plan, ptype=ptype, tech=tech, cross=ch["kind"] == "cross")
        if outcome == "Incomplete" and self.fl["recv"] is None:
            pass
        c.sprint = False
        self.carry_goal = None

    def ray_out(self, c, p1):
        """continue the ray c -> p1 until it leaves the pitch (global frame)"""
        dx, dy = p1[0] - c.x, p1[1] - c.y
        n = hyp(0, 0, dx, dy) + 1e-9
        px, py = p1
        for _ in range(60):
            if px < 0.2 or px > 119.8 or py < 0.2 or py > 79.8:
                break
            px, py = px + dx / n, py + dy / n
        return (min(L, max(0.0, px)), min(W, max(0.0, py)))

    # ------------------------------------------------------------------ shooting
    def do_shoot(self, c, xg, blockers, notes, kind="Open Play", force_outcome=None, penalty=False):
        team, opp = c.team, OPP[c.team]
        rng = self.rng
        ax, ay = self.A(team, c.x, c.y)
        key = None
        lp = self.last_pass
        if lp and lp["recv"] == c.id and self.t - lp["t"] <= 9 and lp["poss"] == self.poss_no:
            key = lp["ev"]["id"]
        b = self.beat if (self.beat and self.beat["team"] == team and self.beat.get("outcome") and self.beat["shooter"] == c.id and self.beat["force"] is not None) else None
        if force_outcome:
            outcome = force_outcome
        elif b and b["kind"] in ("counter", "chance", "disallowed", "corner_goal"):
            outcome = b["outcome"]
        else:
            pb = 0.10 + 0.17 * min(blockers, 2) + (0.06 if notes.get("pressure", 0) > .5 else 0)
            u = rng.random()
            ps = 0.30 + 0.5 * xg
            if u < pb:
                outcome = "Blocked"
            elif u < pb + (1 - pb) * ps:
                outcome = "Saved"
            elif u < pb + (1 - pb) * (ps + 0.03):
                outcome = "Post"
            else:
                outcome = "Off T" if rng.random() < 0.55 else "Wayward"
        gy = {"Goal": rng.uniform(37.5, 42.5), "Goal Disallowed": rng.uniform(37.5, 42.5), "Saved": rng.uniform(37.2, 42.8),
              "Post": rng.choice((36.34, 43.66)), "Off T": rng.choice((rng.uniform(30, 36), rng.uniform(44, 50))), "Wayward": rng.choice((rng.uniform(20, 31), rng.uniform(49, 60)))}.get(outcome, 40.0)
        gz = {"Goal": rng.uniform(0.2, 1.9), "Goal Disallowed": rng.uniform(0.2, 1.9), "Saved": rng.uniform(0.2, 1.8), "Post": 1.1, "Off T": rng.uniform(0.3, 2.2), "Wayward": rng.uniform(2.6, 6.0)}.get(outcome, 0.5)
        if outcome == "Blocked":
            near = min((q for q in self.on[opp] if q.role != "GK"), key=lambda q: seg_dist(*self.A(team, q.x, q.y), ax, ay, 120, 40)[0])
            bx_, by_ = self.A(team, *self.A(team, near.x, near.y))
            end_att = (self.A(team, near.x, near.y))
            ex, ey = end_att
            gz = 0.4
            blocker = near
        else:
            ex, ey, blocker = 120.0, gy, None
        p1 = self.A(team, ex, ey)
        steps = 1 if hyp(ax, ay, ex, ey) < 16 else 2
        part = "Head" if (getattr(self, "ctx_header", False) and kind == "Open Play") else "Right Foot" if c.id % 2 else "Left Foot"
        shot = {"statsbomb_xg": round(xg, 6), "end_location": [round(ex, 1), round(ey, 1), round(gz, 1)], "outcome": {"id": SHOT_OUT[outcome], "name": outcome},
                "first_time": bool(self.first_touch), "technique": {"id": 93, "name": "Normal"}, "body_part": {"id": 37 if part == "Head" else 40, "name": part},
                "type": {"id": {"Open Play": 87, "Free Kick": 62, "Penalty": 88}[kind], "name": kind}, "freeze_frame": self.freeze(team, c.id),
                "x_shot_speed_kmh": round(rng.uniform(70, 115) if part != "Head" else rng.uniform(45, 75), 1)}
        if key:
            shot["key_pass_id"] = key
        extra = {"under_pressure": True} if notes.get("pressure", 0) > .45 else {}
        e = self.emit("Shot", team, c, loc=(c.x, c.y), duration=steps * DT, shot=shot, **extra)
        self.fl = dict(kind="shot", team=team, passer=c, recv=None, p0=(c.x, c.y), p1=p1, t0=self.t, t1=self.t + steps * DT, high=False, outcome=outcome,
                       ev=e, xg=xg, blocker=blocker, key=key, plan={}, shot_kind=kind)
        self.state, self.carrier = "flight", None
        self.carry_goal = None
        if b:
            self.beat_log.setdefault(b["key"], {}).update(shot_id=e["id"], key_pass_id=key, xg=round(xg, 3))
        self.ctx_header = False

    def land_shot(self, fl):
        team, opp = fl["team"], OPP[fl["team"]]
        o, shooter, se = fl["outcome"], fl["passer"], fl["ev"]
        gk = next(p for p in self.on[opp] if p.role == "GK")
        b = self.beat if (self.beat and self.beat["team"] == team and not self.beat["done"] and self.beat_log.get(self.beat["key"], {}).get("shot_id") == se["id"]) else None
        if o == "Goal":
            gke = self.emit("Goal Keeper", opp, gk, loc=(gk.x, gk.y), related_events=[se["id"]], goalkeeper={"type": {"id": 26, "name": "Goal Conceded"}, "outcome": {"id": 47, "name": "No Touch"}})
            self.score[team] += 1
            dur = self.rng.uniform(40, 70)
            subject = se
            v = self.var("Goal Check", team, se, "Goal Confirmed", [("Offside", "No offence"), ("Foul in build-up", "No offence"), ("Handball", "No offence")], dur, shooter=shooter)
            if b:
                self.finish_beat(b)
            self.set_dead("kickoff", opp, (60, 40), dur + 22, "ko")
            for p in self.on[team] + self.on[opp]:
                p.sprint = False
            return
        if o == "Goal Disallowed":
            lp = self.pass_by_id(b.get("key_pass") if b else None) or self.pass_before(se)
            dur = self.rng.uniform(70, 100)
            margin = lp["x_offside_check"]["receiver_margin_m"] if lp else 0.4
            v = self.var("Goal Check", team, se, "Goal Disallowed", [("Offside", "Offside offence"), ("Foul in build-up", "No offence")], dur, shooter=shooter,
                         margin=margin, line=(lp or {}).get("x_offside_check", {}).get("offside_line_x"), key_pass=lp)
            oe = self.emit("Offside", team, shooter, loc=(shooter.x, shooter.y), related_events=[lp["_id"]] if lp else [], x_margin_m=margin, x_var_candidate=True, x_overturned_by_var=True)
            self.emit("Free Kick Awarded", opp, None, loc=(shooter.x, shooter.y), reason="Offside (VAR)", related_events=[oe["id"]])
            if b:
                self.finish_beat(b)
            self.set_dead("free_kick", opp, (shooter.x, shooter.y), dur + 10, "fk")
            return
        if o == "Saved":
            self.emit("Goal Keeper", opp, gk, loc=(gk.x, gk.y), related_events=[se["id"]], goalkeeper={"type": {"id": 33, "name": "Shot Saved"}, "position": {"id": 44, "name": "Set"},
                      "technique": {"id": 45, "name": "Diving"}, "body_part": {"id": 35, "name": "Both Hands"}, "outcome": {"id": 15, "name": "Success"}})
            if b:
                self.finish_beat(b)
            if self.rng.random() < 0.62 or (b and b["key"] == "breakaway"):
                gk.vx = gk.vy = 0.0
                self.new_possession(opp, "gkcatch")
                self.give_ball(gk, hold=3)
            else:
                cx, cy = self.A(team, 119.5, 0.5 if fl["p1"][1] * (1 if DIR[team] > 0 else -1) < 40 else 79.5)
                cy = 0.5 if fl["p1"][1] < 40 else 79.5
                self.set_dead("corner", team, (120 if DIR[team] > 0 else 0, cy), self.rng.uniform(12, 17), "corner")
            return
        if o == "Blocked":
            bl = fl["blocker"]
            bl.x, bl.y = fl["p1"]
            self.emit("Block", opp, bl, loc=(bl.x, bl.y), block={"deflection": self.rng.random() < .3}, related_events=[se["id"]])
            if self.rng.random() < 0.62:
                self.new_possession(opp, "recovery")
                self.give_ball(bl, hold=2)
            else:
                cy = 0.5 if bl.y < 40 else 79.5
                self.set_dead("corner", team, (120 if DIR[team] > 0 else 0, cy), self.rng.uniform(12, 17), "corner")
            if b:
                self.finish_beat(b)
            return
        if b:
            self.finish_beat(b)
        self.set_dead("goal_kick", opp, self.A(team, 114.0, 40.0), self.rng.uniform(9, 14), "gk")

    def pass_by_id(self, pid):
        for e in reversed(self.events):
            if e["id"] == pid:
                x = dict(e["pass"])
                x["_id"] = e["id"]
                return x
        return None

    def pass_before(self, shot_ev):
        kid = shot_ev["shot"].get("key_pass_id")
        for e in reversed(self.events[-60:]):
            if e["id"] == kid:
                x = dict(e["pass"])
                x["_id"] = e["id"]
                return x
        return None

    def finish_beat(self, b):
        b["done"] = True
        if self.beat is b:
            self.beat = None
        self.beat_log.setdefault(b["key"], {})["t"] = self.t
        self.beat_log[b["key"]]["per"] = self.per

    # ------------------------------------------------------------------ VAR
    def var(self, review, team, subject, outcome, checks, dur, shooter=None, margin=None, line=None, key_pass=None, extra=None):
        data = {"review": review, "outcome": {"name": outcome}, "subject_event_id": subject["id"], "duration_s": round(dur, 1),
                "checks": [{"name": n, "result": r} for n, r in checks]}
        if margin is not None:
            data["x_offside_margin_m"] = round(margin, 2)
        if line is not None:
            data["x_offside_line_x"] = line
        if key_pass:
            data["key_pass_event_id"] = key_pass["_id"]
        if extra:
            data.update(extra)
        e = self.emit("VAR", team, shooter, loc=None, var=data, x_key_frame_period_t=round(self.t, 1), x_freeze_frame=self.freeze(team))
        self.var_until = self.t + dur
        return e

    # ------------------------------------------------------------------ scripted beat actions
    def beat_tick(self, c, b):
        """Returns True when the beat handled this tick (the model is bypassed for the forced action only)."""
        team, k = c.team, b["kind"]
        if k == "corner_goal":
            return False
        shooter = self.pl[b["shooter"]]
        me, mates, opps = self.views(c)
        if c.id == shooter.id and k == "disallowed" and not b.get("key_pass"):
            # the offside call is judged on the pass that reaches him: lay it off first so a real pass to him exists
            m2 = min((q for q in mates if q["id"] != shooter.id and q["role"] != "GK"), key=lambda q: hyp(q["x"], q["y"], me["x"], me["y"]) if hyp(q["x"], q["y"], me["x"], me["y"]) > 5 else 99)
            ctx = dict(self.build_ctx(c), sigma_off=0.0, no_offside_filter=True, press_me=0.0)
            plan = plan_pass(me, m2, opps, ctx, self.rng)
            if plan and "excluded" not in plan:
                self.do_pass(c, plan, force_complete=True)
                return True
        if c.id != shooter.id:
            m = next((q for q in mates if q["id"] == shooter.id), None)
            if m is None:
                return False
            margin, line = offside_margin_att(m["x"], me["x"], [o[0] for o in opps])
            want = 0.5 if k == "disallowed" else -0.9
            if (k == "disallowed" and not (0.15 <= margin <= 0.9)) or (k != "disallowed" and margin > -0.5):
                dx = (line + want) - m["x"]                    # he is steering to the line already; only a small final correction is applied
                b["wait_n"] = b.get("wait_n", 0) + 1
                if abs(dx) > 1.5 and b["wait_n"] <= 14:
                    self.next_t = self.t + DT
                    return True
                dx = max(-1.5, min(1.5, dx))
                shooter.x, shooter.y = self.A(team, m["x"] + dx, m["y"])
                m = self.pdict(shooter, team)
            ctx = dict(self.build_ctx(c), sigma_off=0.0, no_offside_filter=True, press_me=pressure_at(me["x"], me["y"], [(o[0], o[1]) for o in opps]))
            plan = plan_pass(me, m, opps, ctx, self.rng)
            if not plan or "excluded" in plan:
                self.next_t = self.t + DT
                return True
            plan["kind"] = "through" if plan["kind"] == "pass" and m["x"] > me["x"] + 6 else plan["kind"]
            self.do_pass(c, plan, force_complete=True)
            b["key_pass"] = self.last_pass["ev"]["id"]
            return True
        ax, ay = self.A(team, c.x, c.y)
        f = self.pl.get(b.get("fouler")) if b.get("fouler") else None
        if k == "box_edge_foul":
            if ax >= 99.5 and (hyp(f.x, f.y, c.x, c.y) <= 2.7 or ax >= 100.8):       # contact right on the edge of the area (VAR will measure it)
                self.scripted_box_edge(f, c, b)
                return True
            return False
        if k not in ("foul_red", "foul_pen"):
            return False
        trig = {"foul_red": ax >= 86, "foul_pen": ax >= 104}[k]
        if trig or self.t - b["t_force"] > 16:
            dist = hyp(f.x, f.y, c.x, c.y)
            if dist <= 2.7:
                (self.scripted_red if k == "foul_red" else self.scripted_pen)(f, c, b)
                return True
            b["hold_n"] = b.get("hold_n", 0) + 1           # shield the ball for a moment until the chasing defender arrives
            if b["hold_n"] > 12:                            # safety: he has been chasing for 6 s - bring him within reach
                f.x, f.y = c.x - (1.8 if DIR[team] > 0 else -1.8), c.y + 0.5
            self.carry_goal = None
            c.cap = None
            self.next_t = self.t + DT
            return True
        return False

    def scripted_red(self, f, c, b):
        team, opp = c.team, f.team
        fe = self.foul(f, c, scripted="red")
        self.emit("Free Kick Awarded", team, None, loc=(c.x, c.y), reason="Foul", related_events=[fe["id"]])
        self.var("Red Card Review", opp, fe, "Red Card Upheld", [("Denial of obvious goal-scoring opportunity", "Confirmed"), ("Point of contact", "Confirmed")],
                 self.rng.uniform(55, 80), shooter=f)
        self.beat_log.setdefault(b["key"], {}).update(foul_id=fe["id"], player=f.id)
        self.red_done = True
        ids = [i for i in self.lineup[opp] if i != f.id]
        self.set_lineup(opp, ids, positions={92011: "Left Center Back", 92010: "Center Forward"})
        self.tactics_event(opp)
        self.finish_beat(b)
        self.set_dead("free_kick", team, (c.x, c.y), self.rng.uniform(70, 85), "fk")

    def scripted_pen(self, f, c, b):
        team, opp = c.team, f.team
        fe = self.foul(f, c, scripted="pen")
        fe["foul_committed"]["penalty"] = True
        self.events[-1]["foul_won"] = {"penalty": True}
        self.var("Penalty Review", team, fe, "Penalty Confirmed", [("Contact", "Confirmed"), ("Inside the area", "Confirmed")], self.rng.uniform(40, 60), shooter=c)
        self.beat_log.setdefault(b["key"], {}).update(foul_id=fe["id"], player=c.id)
        b["outcome"] = "Goal"
        b["kind"] = "penalty_kick"
        self.set_dead("penalty", team, self.A(team, 108.0, 40.0), self.var_until - self.t + 14.0, "fk")

    def scripted_box_edge(self, f, c, b):
        team, opp = c.team, f.team
        ax, ay = self.A(team, c.x, c.y)
        fe = self.foul(f, c, scripted="box_edge")
        self.events[-1]["foul_won"] = {"penalty": True, "x_awarded_on_field": True}
        self.var("Penalty Review", team, fe, "Penalty Overturned", [("Contact", "Confirmed"), ("Inside the area", f"No: contact {max(0.0, BOX_X - ax):.1f} m outside")],
                 self.rng.uniform(60, 85), shooter=c, extra={"x_original_decision": "Penalty", "x_final_decision": "Free Kick", "x_contact_outside_box_m": round(max(0.0, BOX_X - ax), 2)})
        self.emit("Free Kick Awarded", team, None, loc=(c.x, c.y), reason="Foul (penalty overturned by VAR)", related_events=[fe["id"]])
        self.beat_log.setdefault(b["key"], {}).update(foul_id=fe["id"], player=c.id)
        self.finish_beat(b)
        self.set_dead("free_kick", team, (c.x, c.y), self.var_until - self.t + 10.0, "fk")

    # ------------------------------------------------------------------ restarts
    def pick_taker(self, kind, team, loc):
        ps = [p for p in self.on[team]]
        out = [p for p in ps if p.role != "GK"]
        b = self.beat
        if kind == "kickoff":
            return next((p for p in out if p.pos in ("Center Forward", "Right Center Forward")), out[-1])
        if kind == "goal_kick":
            return next(p for p in ps if p.role == "GK")
        if kind == "penalty":
            return self.pl[b["shooter"]] if b else min(out, key=lambda p: -p.vmax)
        if kind == "corner":
            pool = [p for p in out if p.role in ("ATT", "MID") and not (b and p.id == b["shooter"])]
            return min(pool, key=lambda p: hyp(p.x, p.y, loc[0], loc[1]) - (6 if "Wing" in p.pos else 0))
        if kind == "throw":
            pool = [p for p in out if p.role != "DEF" or "Back" in p.pos and "Center" not in p.pos]
            return min(pool, key=lambda p: hyp(p.x, p.y, loc[0], loc[1]))
        pool = [p for p in out if p.role in ("MID", "DEF") and not (b and p.id == b["shooter"])]
        return min(pool, key=lambda p: hyp(p.x, p.y, loc[0], loc[1]) - (0 if p.role == "MID" else -5))

    def run_restart(self):
        d = self.dead
        team, kind, loc = d["team"], d["kind"], d["loc"]
        if "taker" not in d:
            d["taker"] = self.pick_taker(kind, team, loc).id
        taker = self.pl[d["taker"]]
        if hyp(taker.x, taker.y, *loc) > 3.5 and d.get("wait", 0) < 6:
            d["wait"] = d.get("wait", 0) + 1
            d["ready"] += 1.0
            return
        taker.x, taker.y = loc
        taker.vx = taker.vy = 0.0
        how = d["how"]
        self.new_possession(team, how)
        self.dead = None
        self.carrier, self.state = taker, "carried"
        self.ball = [loc[0], loc[1], 0.0]
        self.counter_until = -1.0
        self.first_touch = False
        self.ctx_header = False
        ctx = self.build_ctx(taker)
        me, mates, opps = self.views(taker)
        ax, ay = me["x"], me["y"]
        if kind == "penalty":
            self.do_shoot(taker, 0.78, 0, {}, kind="Penalty", force_outcome="Goal")
            return
        if kind == "corner":
            return self.corner_kick(taker, mates, opps, ctx)
        if kind == "free_kick":
            dg = hyp(ax, ay, 120, 40)
            if dg <= 25 and 18 < ay < 62 and not (self.beat and self.beat["team"] == team and self.beat["force"]) and self.rng.random() < 0.3:
                xg = xg_shot(ax, ay, "foot") * 0.6
                return self.do_shoot(taker, xg, 1, {"pressure": 0.0}, kind="Free Kick")
        ptype = {"kickoff": "Kick Off", "throw": "Throw-in", "goal_kick": "Goal Kick", "free_kick": "Free Kick"}[kind]
        ctx = dict(ctx, first_time=False)
        ch, top3, notes = decide(ctx, me, mates, opps, self.rng, only_pass=True)
        if "to" not in ch:
            ch, top3, notes = decide(dict(ctx, sigma_off=-99), me, mates, opps, self.rng, only_pass=True)
        self.decisions.append(dict(t=round(self.t, 1), per=self.per, team=team, player=taker.id, chosen="pass", to=ch.get("to"), restart=ptype, top3=top3, **notes,
                                   x=round(ax, 1), y=round(ay, 1)))
        if "to" not in ch:
            self.next_t = self.t + DT
            return
        self.do_pass(taker, ch, ptype=ptype)

    def corner_kick(self, taker, mates, opps, ctx):
        team = taker.team
        b = self.beat if (self.beat and self.beat["team"] == team and self.beat["kind"] == "corner_goal") else None
        me = self.pdict(taker, team)
        cands = [m for m in mates if m["x"] >= 98 and 14 <= m["y"] <= 66 and m["role"] != "GK"]
        tgt = None
        if b:
            tgt = next(m for m in mates if m["id"] == b["shooter"])
        elif cands and self.rng.random() > 0.15:
            tgt = self.rng.choices(cands, weights=[2.0 if m["role"] != "MID" else 1.0 for m in cands])[0]
        if tgt is None:
            ch, top3, notes = decide(ctx, me, mates, opps, self.rng, only_pass=True)
            if "to" not in ch:
                self.next_t = self.t + DT
                return
            return self.do_pass(taker, ch, ptype="Corner")
        ex, ey = tgt["x"] + self.rng.uniform(-1.2, 1.2), tgt["y"] + self.rng.uniform(-2, 2)
        d = hyp(me["x"], me["y"], ex, ey)
        pc = 0.42 + 0.1 * ctx["pass_skill"]
        plan = {"to": tgt["id"], "end": (ex, ey), "margin": -5.0, "line": 0.0, "steps": flight_steps(d, True), "high": True, "kind": "cross", "pc": pc, "dist": d}
        self.decisions.append(dict(t=round(self.t, 1), per=self.per, team=team, player=taker.id, chosen="pass", to=tgt["id"], restart="Corner", top3=[], x=round(me["x"], 1), y=round(me["y"], 1)))
        self.do_pass(taker, plan, ptype="Corner", force_complete=bool(b))

    # ------------------------------------------------------------------ movement
    def team_targets(self, team, ball, tg):
        opp = OPP[team]
        cfg = self.team_cfg(team)
        bxA, byA = self.A(team, ball[0], ball[1])
        dead = self.dead if self.state == "dead" else None
        poss = (dead["team"] == team) if dead else (self.poss_team == team)
        opps = [self.A(team, p.x, p.y) for p in self.on[opp]]
        xs = sorted((o[0] for o in opps), reverse=True)
        off = max(xs[1] if len(xs) > 1 else xs[0], 60.0)
        counter = self.counter_until > self.t and poss
        line = max(26.0, min(62.0, bxA - (30 if counter else 24))) if poss else max(25.0, min(56.0, cfg["line"] + 0.30 * (bxA - 60)))
        comp, wf, sh = (0.8, 1.05, 0.28) if poss else (0.62, 0.78, 0.5)
        outfield = [p for p in self.on[team] if p.role != "GK"]
        near = sorted(outfield, key=lambda p: hyp(p.x, p.y, ball[0], ball[1]))
        pressers = [] if poss else [p.id for p in near[:2] if hyp(p.x, p.y, ball[0], ball[1]) < 20][: (2 if cfg["press"] > 0.7 else 1)]
        taker = dead.get("taker") if dead else None
        beat_f = self.beat and self.beat["force"]
        for p in self.on[team]:
            if p.role == "GK":
                gx = 4.0 + 0.08 * min(bxA, 70) + (4.0 if poss and bxA > 60 else 0.0)
                gy = 40 + max(-6.0, min(6.0, (byA - 40) * 0.18))
                tg[p.id] = (self.A(team, gx, gy), False)
                continue
            sx, sy = SLOT[p.pos]
            depth = sx - 26
            tx = line + depth * comp
            ty = 40 + (sy - 40) * wf + (byA - 40) * sh
            spr = False
            if dead and dead["kind"] == "kickoff":
                tx, ty = min(sx, 56 if p.role != "ATT" else 50), sy
            elif dead and dead["kind"] == "penalty" and p.id != taker:
                tx, ty = 96.0 if poss else 97.0, 10 + (p.id % 12) * 5.5
            elif poss and p.role == "ATT":
                tx = max(min(tx, line + 40), min(bxA + 4, off - p.hold))
                tx = min(tx, off - p.hold)
                if p.run > 0:
                    p.run -= 1
                    tx, spr = min(117.0, off + 6.5), True
                elif bxA > 45 and self.rng.random() < 0.05 and not dead:
                    p.run = 3
            elif not poss:
                gd = [o for o in opps if hyp(tx, ty, o[0], o[1]) < 9 and o[0] > tx - 3]
                if gd and p.role in ("DEF", "MID"):
                    o = min(gd, key=lambda o: hyp(tx, ty, o[0], o[1]))
                    tx, ty = tx * 0.65 + (o[0] - 1.5) * 0.35, ty * 0.65 + o[1] * 0.35
            if p.id in pressers:
                tx, ty, spr = bxA - 0.8, byA, True
            if dead and dead["kind"] == "corner" and poss and self.beat and self.beat["kind"] == "corner_goal" and p.id == self.beat["shooter"]:
                tx, ty = 109.0, 42.0
            if taker == p.id:
                tx, ty = self.A(team, *dead["loc"])
                spr = True
            if beat_f and self.beat["team"] != team and self.beat.get("fouler") == p.id:
                c = self.carrier
                if c is not None:
                    tx, ty = self.A(team, c.x, c.y)
                    spr = True
            if beat_f and self.beat["team"] == team and p.id == self.beat["shooter"] and self.carrier is not p and not dead and self.beat["kind"] in ("counter", "chance", "disallowed", "foul_red", "foul_pen", "box_edge_foul"):
                tx, ty, spr = min(117.0, off + (0.5 if self.beat["kind"] == "disallowed" else -0.9)), ty, True
            tx = max(1.0, min(119.0, tx))
            ty = max(1.0, min(79.0, ty))
            tg[p.id] = (self.A(team, tx, ty), spr)

    def advance(self):
        ball = self.ball_pos()
        tg = {}
        for team in TEAMS:
            self.team_targets(team, ball, tg)
        fl = self.fl if self.state == "flight" else None
        if fl and fl["kind"] == "pass":
            if fl["recv"] is not None:
                tg[fl["recv"].id] = (fl["p1"], True)
            if fl["interceptor"] is not None:
                tg[fl["interceptor"].id] = (fl["p1"], True)
        if fl and fl["kind"] == "clear":                           # nearest players chase the cleared ball
            chasers = sorted((q for tm in TEAMS for q in self.on[tm] if q.role != "GK"), key=lambda q: hyp(q.x, q.y, *fl["p1"]))[:4]
            for q in chasers:
                tg[q.id] = (fl["p1"], True)
        if self.state == "carried" and self.carrier is not None:
            c = self.carrier
            if self.carry_goal and self.carry_goal[0] is c:
                tg[c.id] = (self.carry_goal[1], True)
            else:
                ax, ay = self.A(c.team, c.x, c.y)
                tg[c.id] = (self.A(c.team, min(118.0, ax + 1.5), ay), False)
        if self.rng.random() < 0.2:
            for tm in TEAMS:
                for p in self.on[tm]:
                    p.hold = max(0.2, min(2.6, p.hold + self.rng.gauss(0, 0.45)))
        for tm in TEAMS:
            for p in self.on[tm]:
                (tx, ty), spr = tg[p.id]
                dx, dy = tx - p.x, ty - p.y
                d = math.hypot(dx, dy)
                cap = p.cap if (p.cap and self.carrier is p) else (p.vmax if spr else min(p.vmax, 5.0))
                sp = min(cap, 1.0 + 1.4 * d)
                dvx, dvy = (dx / d * sp, dy / d * sp) if d > 1e-6 else (0.0, 0.0)
                p.vx, p.vy = 0.5 * p.vx + 0.5 * dvx, 0.5 * p.vy + 0.5 * dvy
                step = math.hypot(p.vx, p.vy) * DT
                if step > d and d > 0:
                    k = d / step
                    p.vx, p.vy = p.vx * k, p.vy * k
                p.x = max(0.3, min(L - 0.3, p.x + p.vx * DT))
                p.y = max(0.3, min(W - 0.3, p.y + p.vy * DT))
        if self.carrier is None or self.state != "carried":
            pass
        self.t = round(self.t + DT, 3)

    def visual_ball(self):
        """Ball for the tracking frames only (game logic keeps using ball_pos): while the ball is dead it rolls from where it went
        out / was saved to the restart spot at ~10 m/s instead of jumping there."""
        d = self.dead if self.state == "dead" else None
        if not d or "vfrom" not in d:
            return self.ball_pos()
        dist = hyp(d["vfrom"][0], d["vfrom"][1], *d["loc"])
        dur = max(0.5, min(dist / 10.0, max(0.5, d["ready"] - d["vt0"] - 0.5)))
        u = min(1.0, max(0.0, (self.t - d["vt0"]) / dur))
        u = u * u * (3 - 2 * u)
        return [d["vfrom"][0] + (d["loc"][0] - d["vfrom"][0]) * u, d["vfrom"][1] + (d["loc"][1] - d["vfrom"][1]) * u, 0.0]

    def record(self):
        b = self.visual_ball()
        self.frames.append((self.per, self.t, round(b[0], 2), round(b[1], 2), round(b[2], 1),
                            [(p.id, round(p.x, 2), round(p.y, 2)) for tm in TEAMS for p in self.on[tm]], self.poss_team if self.state != "dead" else None))

    # ------------------------------------------------------------------ periods
    def start_period(self, per, kick):
        self.per, self.t = per, 0.0
        self.var_until, self.beat, self.counter_until = 0.0, None, -1.0
        if per == 1:
            for team, xi in (("KES", KES), ("IRO", IRO)):
                self.set_lineup(team, [p[0] for p in xi])
            for team in TEAMS:
                self.emit("Starting XI", team, None, tactics={"formation": 4231 if team == "KES" else 442, "lineup": [
                    {"player": {"id": p.id, "name": p.name}, "position": {"id": POS_ID[p.pos], "name": p.pos}, "jersey_number": p.no} for p in self.on[team]]})
        for team in TEAMS:
            self.emit("Half Start", team)
        if per == 2:
            for team in TEAMS:
                self.tactics_event(team)
        self.poss_team = kick
        self.place_kickoff(kick)
        self.set_dead("kickoff", kick, (60.0, 40.0), 1.0, "ko")

    def period_done(self):
        if self.t < self.P_END[self.per]:
            if self.per == 2 and self.finish_t.get("winner") is not None and self.t >= self.finish_t["winner"] + 40 and self.state == "dead" and self.t >= self.var_until:
                return True
            return False
        if self.per == 1:
            return (self.state == "dead" and self.t >= self.var_until) or self.t >= self.P_END[1] + 25
        return True

    def run(self):
        for per, kick in ((1, "KES"), (2, "IRO")):
            self.start_period(per, kick)
            while not self.period_done():
                self.step()
                for k, v in list(self.beat_log.items()):
                    if k in ("abandoned",):
                        continue
                    if "t" in v and k not in self.finish_t:
                        self.finish_t[k] = v["t"]
            if per == 2:
                self.record()
            for team in TEAMS:
                self.emit("Half End", team)
            if per == 1:
                self.p1_end = self.t
        return self
