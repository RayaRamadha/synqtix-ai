#!/usr/bin/env python3
"""Generate the synthetic match with the decision model + simulator.

  python -m sim.generate                 # seed 2026 -> data/synthetic_match_900001.json, data/tracking_900001.json, data/match_info.json
  python -m sim.generate --decisions     # also data/decisions_900001.json (explainability log: top-3 alternatives per decision)
Same seed -> byte-identical output.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.world import World, NAME, TID
from sim.beats import BEATS
from sim.roster import KES, KES_SUBS, IRO, IRO_SUBS
from sim.metrics import acceptance

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = {"Kestrel Rovers": "KES", "Ironvale Athletic": "IRO"}


def dump(obj, path, **kw):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, **kw)
        f.write("\n")


def build_tracking(w):
    roster = [(p, "KES") for p in KES + KES_SUBS] + [(p, "IRO") for p in IRO + IRO_SUBS]
    idx = {p[0]: i for i, (p, _) in enumerate(roster)}
    p1_end = w.p1_end
    frames = []
    for per, t, bx, by, bz, plist, poss in w.frames:
        row = [round(t + (p1_end if per == 2 else 0.0), 2), per, bx, by, bz, {"KES": 0, "IRO": 1}.get(poss, -1)]
        flat = [-1.0] * (2 * len(roster))
        for pid, x, y in plist:
            i = idx[pid]
            flat[2 * i], flat[2 * i + 1] = x, y
        frames.append(row + flat)
    return {"synthetic": True, "match_id": 900001, "seed": w.seed, "frame_rate_hz": 2,
            "note": "SIMULATED positions (decision-model simulator), not measured tracking. Fictional players.",
            "coord_space": "global pitch 120x80 m, y down; Kestrel attack towards x=120, Ironvale towards x=0 (event files use each team's own attacking frame). z = ball height in m.",
            "frame_layout": "[stream_seconds, period, ball_x, ball_y, ball_z, possession(0=KES,1=IRO,-1=dead ball), x0,y0,x1,y1,...] players in `players` order; -1,-1 = not on the pitch",
            "p1_end": p1_end, "players": [{"id": p[0], "team": tm, "name": p[1], "no": p[3], "pos": p[2]} for p, tm in roster], "frames": frames}


def moment_text(w, ev, key):
    by = {e["id"]: e for e in ev}
    L = w.beat_log[key]
    return by[L.get("shot_id") or L.get("foul_id")]


def check_windows(w):
    """Every scripted beat's key event must land inside its minute window (period-local seconds)."""
    by = {e["id"]: e for e in w.events}
    bad = []
    for b in BEATS:
        L = w.beat_log.get(b["key"], {})
        e = by.get(L.get("shot_id") or L.get("foul_id"))
        if e is None:
            bad.append((b["key"], "missing"))
            continue
        loc_t = e["minute"] * 60 + e["second"] - (45 * 60 if e["period"] == 2 else 0)
        if not (b["window"][0] <= loc_t < b["window"][1]):
            bad.append((b["key"], loc_t, b["window"]))
    return bad


def build_info(w, acc):
    ev = w.events
    shots = [e for e in ev if e["type"]["name"] == "Shot"]
    by = {e["id"]: e for e in ev}
    stats = {}
    for code, name in (("KES", "Kestrel Rovers"), ("IRO", "Ironvale Athletic")):
        sh = [e for e in shots if e["team"]["name"] == name]
        ps = [e for e in ev if e["type"]["name"] == "Pass" and e["team"]["name"] == name]
        h = lambda per: round(sum(e["shot"]["statsbomb_xg"] for e in sh if e["period"] == per), 2)
        stats[code] = {"name": name, "goals": sum(1 for e in sh if e["shot"]["outcome"]["name"] == "Goal"), "shots": len(sh),
                       "xg": round(sum(e["shot"]["statsbomb_xg"] for e in sh), 2), "passes": len(ps),
                       "pass_accuracy_pct": round(100 * sum(1 for e in ps if "outcome" not in e["pass"]) / max(1, len(ps)), 1), "xg_h1": h(1), "xg_h2": h(2)}
    # possession = share of simulated ball-control time per half (from the frames)
    for per, key in ((1, "possession_h1_pct"), (2, "possession_h2_pct")):
        c = {"KES": 0, "IRO": 0}
        for f in w.frames:
            if f[0] == per and f[6] in c:
                c[f[6]] += 1
        tot = max(1, c["KES"] + c["IRO"])
        stats["KES"][key], stats["IRO"][key] = round(100 * c["KES"] / tot, 1), round(100 * c["IRO"] / tot, 1)
    ks, ir = stats["KES"], stats["IRO"]
    # --- answer key (scenario moments; ids come from the simulated events)
    def mk(key, kind, headline, why, player, team, ev_):
        d = {"key": key, "match_minute": ev_["minute"], "headline": headline, "why_it_matters": why, "event_ids": [ev_["id"]], "event_index": ev_["index"],
             "kind": kind, "player": player, "team": team}
        return d
    def poss_len(e):
        seq = [x for x in ev if x["possession"] == e["possession"] and x["index"] <= e["index"]]
        t0 = seq[0]
        sec = (e["minute"] * 60 + e["second"]) - (t0["minute"] * 60 + t0["second"])
        return sum(1 for x in seq if x["type"]["name"] == "Pass" and x["team"]["name"] == e["team"]["name"]), sec
    s = {k: moment_text(w, ev, k) for k in ("counter1", "post", "counter2", "red", "goal61", "penalty", "breakaway", "winner")}
    n1, sec1 = poss_len(s["counter1"])
    pw = "pass" if n1 == 1 else "passes"
    iro_sh = [e for e in shots if e["team"]["name"] == "Ironvale Athletic" and e["index"] <= s["counter2"]["index"]]
    iro_goals = sum(1 for e in iro_sh if e["shot"]["outcome"]["name"] == "Goal")
    iro_xg = round(sum(e["shot"]["statsbomb_xg"] for e in iro_sh), 2)
    ak = [
        mk("counter1", "goal", "Ironvale counter-attack goal: 0-1",
           f"Kestrel had the ball but lost it; Ironvale scored on a counter ({n1} {pw}, ~{sec1} seconds from the first touch of the move).", "Dario Oquendo", "IRO", s["counter1"]),
        mk("post", "big_chance", "Valdrin hits the post", "Biggest first-half Kestrel chance. Territorial dominance without a goal: control is not the same as danger.", "Mateo Valdrin", "KES", s["post"]),
        mk("counter2", "goal", "Ironvale second counter goal: 0-2",
           f"{len(iro_sh)} Ironvale shots, {iro_goals} goals (xG {iro_xg}) versus Kestrel's high possession. Pattern: clinical on transitions.", "Stellan Kriek", "IRO", s["counter2"]),
        mk("red", "red_card", "Red card: Dostal sent off (last-man foul on Valdrin)",
           "Ironvale drop to 10 men and switch to 4-4-1. Game state flips: Kestrel pressure and territory should rise sharply.", "Marek Dostal", "IRO", s["red"]),
        mk("goal61", "goal", "Valdrin pulls one back: 1-2", "First goal after the red card. Momentum swing begins; Valdrin's 1st of 3.", "Mateo Valdrin", "KES", s["goal61"]),
        mk("penalty", "goal", "Valdrin converts the penalty: 2-2", "Penalty won by Valdrin himself (VAR-checked), 2nd of his hat-trick. Ironvale's depleted defence is giving away fouls in the box.", "Mateo Valdrin", "KES", s["penalty"]),
        mk("breakaway", "big_save", "Rovik saves one-on-one", "Ironvale (10 men) break once more; a goal here would have put them 3-2 up. The save keeps Kestrel's comeback alive.", "Anselm Rovik", "KES", None) if False else None,
    ]
    sv = s["breakaway"]
    ak[6] = mk("breakaway", "big_save", "Rovik saves one-on-one",
               "Ironvale (10 men) break once more; a goal here would have put them 3-2 up. The save keeps Kestrel's comeback alive.", "Anselm Rovik", "KES", sv)
    ak[5]["foul_event_id"] = w.beat_log["penalty"]["foul_id"]
    wn = mk("winner", "goal", "Valdrin completes the hat-trick: 3-2 (90+3')",
            "Header from a corner in injury time. Comeback from 0-2, third Valdrin goal: three different goal types (open play, penalty, header).", "Mateo Valdrin", "KES", s["winner"])
    wn["milestone"] = "hat-trick"
    ak.append(wn)
    ak.sort(key=lambda k: k["event_index"])
    cards = [{"minute": e["minute"], "team": CODE[e["team"]["name"]], "player": e["player"]["name"], "card": e["foul_committed"]["card"]["name"]}
             for e in ev if e["type"]["name"] == "Foul Committed" and e.get("foul_committed", {}).get("card")]
    subs = [{"minute": e["minute"], "team": CODE[e["team"]["name"]], "out": e["player"]["name"], "in": e["substitution"]["replacement"]["name"]} for e in ev if e["type"]["name"] == "Substitution"]
    var = [{"minute": e["minute"], "review": e["var"]["review"], "outcome": e["var"]["outcome"]["name"], "team": CODE[e["team"]["name"]],
            "event_id": e["id"], "subject_event_id": e["var"]["subject_event_id"]} for e in ev if e["type"]["name"] == "VAR"]
    return {
        "synthetic": True, "note": "Fully synthetic data. Clubs, players and results are fictional; any resemblance to real people is coincidental.",
        "match_id": 900001, "seed": w.seed, "generator": "sim/generate.py (decision model + match simulator)",
        "teams": {"KES": {"id": 9001, "name": "Kestrel Rovers", "formation": 4231}, "IRO": {"id": 9002, "name": "Ironvale Athletic", "formation": 442}},
        "final_score": {"Kestrel Rovers": w.score["KES"], "Ironvale Athletic": w.score["IRO"]},
        "scenario": "Comeback: dominant-but-toothless Kestrel go 0-2 down to counters, Ironvale lose a defender to a red card (55'), Valdrin hat-trick wins it 3-2 in injury time.",
        "stats": stats, "answer_key_moments": ak, "cards": cards, "substitutions": subs, "var_reviews": var,
        "simulation": {
            "method": "Seeded utility-based decision model (shoot / pass / carry / dribble / clear / hold) + 2 Hz movement simulator; scripted beats fix WHEN the answer-key moments happen, not how the play around them looks.",
            "scripted_beats": [{"key": b["key"], "kind": b["kind"], "period": b["per"], "team": b["team"], "window_s": list(b["window"])} for b in BEATS],
            "unscripted_shots_never_score": True,
            "possession_basis": "share of simulated ball-control time (tracking frames)",
            "acceptance": acc},
        "x_extensions": {
            "pass.x_difficulty": "0-1 pass difficulty rating (1 - modelled completion probability)",
            "pass.x_ball_speed_kmh": "ball speed in km/h", "shot.x_shot_speed_kmh": "shot speed in km/h", "carry.x_speed_kmh": "player speed while carrying the ball, km/h",
            "pressure.x_pressure_level": "0-1 pressure intensity",
            "pass.x_offside_check": "{receiver_margin_m, offside_line_x, receiver_offside}: receiver vs the offside line at the moment the pass was played (attacking frame, + = offside)",
            "Offside": "type Offside: offside player, x_margin_m, x_var_candidate (|margin|<0.5 m), x_overturned_by_var; pass outcome 'Pass Offside' + 'Free Kick Awarded' follow",
            "Free Kick Awarded": "type Free Kick Awarded: team awarded, reason (Offside/Foul/...)",
            "VAR": "type VAR: var.review, var.outcome, var.checks[], var.duration_s, var.subject_event_id, x_offside_margin_m, x_freeze_frame; shot outcome 'Goal Disallowed' = ball in the net, ruled out by VAR",
            "tracking": "data/tracking_900001.json: SIMULATED 2 Hz positions of all 22 players + ball"},
        "event_count": len(ev)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--decisions", action="store_true", help="also write the explainability log data/decisions_900001.json")
    a = ap.parse_args()
    w = World(a.seed).run()
    acc = acceptance(w.events, w.decisions)
    bad = check_windows(w)
    if bad:
        print("BEAT WINDOW MISS:", bad, file=sys.stderr)
        sys.exit(3)
    os.makedirs(a.out, exist_ok=True)
    dump(w.events, f"{a.out}/synthetic_match_900001.json", separators=(",", ":"))
    dump(build_tracking(w), f"{a.out}/tracking_900001.json", separators=(",", ":"))
    dump(build_info(w, acc), f"{a.out}/match_info.json", indent=2)
    if a.decisions:
        dump({"synthetic": True, "seed": a.seed, "note": "One record per decision: chosen action and the top-3 alternatives with values (xG-like units).", "decisions": w.decisions},
             f"{a.out}/decisions_900001.json", separators=(",", ":"))
    print(json.dumps(acc, indent=1))
    if w.beat_log.get("abandoned"):
        print("WARNING: abandoned beats:", w.beat_log["abandoned"], file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
