#!/usr/bin/env python3
"""Compact raw events -> static/tracks.js (window.TRACKS) so the frontend can animate the FULL match (players + ball + commentary)
without the 2.3 MB raw file. Coordinates stay in each team's own attacking frame (x->120); the page flips the second team."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from engine.state import secs
B = os.path.dirname(os.path.abspath(__file__))
info = json.load(open(f"{B}/data/match_info.json", encoding="utf-8"))
codes = {t["id"]: c for c, t in info["teams"].items()}
ev = sorted(json.load(open(f"{B}/data/synthetic_match_900001.json", encoding="utf-8")), key=lambda e: e["index"])
p1_end = max(secs(e["timestamp"]) for e in ev if e["type"]["name"] == "Half End" and e["period"] == 1)
def r(v): return round(v, 1)
SID = {e["id"]: round(secs(e["timestamp"]) + (p1_end if e["period"] >= 2 else 0), 2) for e in ev}
lineups, jerseys, names, pos = {}, {}, {}, {}
events, subs = [], []
for e in ev:
    t = e["type"]["name"]
    if t == "Starting XI":
        c = codes[e["team"]["id"]]
        lineups[c] = {"formation": e["tactics"]["formation"], "players": []}
        for p in e["tactics"]["lineup"]:
            pid = p["player"]["id"]
            lineups[c]["players"].append({"id": pid, "pos": p["position"]["name"], "no": p["jersey_number"]})
            names[pid] = p["player"]["name"]; jerseys[pid] = p["jersey_number"]; pos[pid] = p["position"]["name"]
for e in ev:
    t = e["type"]["name"]
    if t in ("Starting XI", "Half Start", "Tactical Shift", "Half End"): continue
    s = round(secs(e["timestamp"]) + (p1_end if e["period"] >= 2 else 0), 2)
    c = codes[e["team"]["id"]]
    pl = e.get("player") or {}
    if pl.get("id"): names.setdefault(pl["id"], pl["name"])
    if t == "Substitution":
        rep = e["substitution"]["replacement"]; names[rep["id"]] = rep["name"]
        subs.append({"s": s, "team": c, "out": pl["id"], "in": rep["id"], "pos": pl.get("position", {}).get("name")}); continue
    if t == "VAR":
        v = e["var"]; sub = v.get("subject_event_id")
        subj = next((x for x in ev if x["id"] == sub), None)
        kp = SID.get(v.get("key_pass_event_id")) or (SID.get((subj or {}).get("shot", {}).get("key_pass_id")) if subj else None)
        events.append({"s": s, "k": "VAR", "t": c, "p": pl.get("id"), "x": 0, "y": 0, "d": 0, "pc": codes[e["possession_team"]["id"]], "rv": v["review"],
                       "o": v["outcome"]["name"], "dur": v.get("duration_s", 0), "ck": [[q["name"], q["result"]] for q in v.get("checks", [])],
                       "mg": v.get("x_offside_margin_m"), "cm": v.get("x_contact_outside_box_m"), "sj": SID.get(sub, s), "kp": kp, "sp": (subj or {}).get("player", {}).get("id")})
        continue
    loc = e.get("location")
    if not loc: continue
    o = {"s": s, "k": t, "t": c, "p": pl.get("id"), "x": r(loc[0]), "y": r(loc[1]), "d": round(e.get("duration") or 0, 2),
         "pc": codes[e["possession_team"]["id"]]}
    if t == "Pass":
        ps = e["pass"]; o["ex"], o["ey"] = r(ps["end_location"][0]), r(ps["end_location"][1])
        o["to"] = (ps.get("recipient") or {}).get("id"); o["o"] = (ps.get("outcome") or {}).get("name", "Complete")
        o["h"] = (ps.get("height") or {}).get("name"); o["ty"] = (ps.get("type") or {}).get("name")
        if (ps.get("recipient") or {}).get("id"): names.setdefault(ps["recipient"]["id"], ps["recipient"]["name"])
    elif t == "Carry":
        o["ex"], o["ey"] = r(e["carry"]["end_location"][0]), r(e["carry"]["end_location"][1])
    elif t == "Shot":
        sh = e["shot"]; o["ex"], o["ey"] = r(sh["end_location"][0]), r(sh["end_location"][1])
        o["o"] = sh["outcome"]["name"]; o["xg"] = round(sh["statsbomb_xg"], 2); o["ty"] = sh["type"]["name"]
    elif t == "Foul Committed":
        o["o"] = ((e.get("foul_committed") or {}).get("card") or {}).get("name")
    elif t == "Duel":
        o["o"] = e["duel"]["outcome"]["name"]; o["ty"] = e["duel"]["type"]["name"]
    elif t == "Goal Keeper":
        o["ty"] = e["goalkeeper"]["type"]["name"]
    elif t == "Dribble":
        o["o"] = e["dribble"]["outcome"]["name"]
    elif t == "Pressure":
        o["lv"] = e.get("x_pressure_level")
    elif t == "Offside":
        o["mg"] = e.get("x_margin_m"); o["vo"] = bool(e.get("x_overturned_by_var"))
        o["ps"] = SID.get((e.get("related_events") or [None])[0], s)
    elif t == "Free Kick Awarded":
        o["o"] = e.get("reason")
    events.append(o)
# ---- SIMULATED tracking frames (data/tracking_900001.json) -> compact delta-encoded ints (decimetres) ----
trk = None
tp = f"{B}/data/tracking_900001.json"
if os.path.exists(tp):
    T = json.load(open(tp, encoding="utf-8"))
    n = len(T["players"]); W = 3 + 2 * n
    prev, d, ts, poss = [0] * W, [], [], []
    for f in T["frames"]:
        cur = [round(f[2] * 10), round(f[3] * 10), round(f[4] * 10)] + [round(v * 10) for v in f[6:]]
        d += [a - b for a, b in zip(cur, prev)]; prev = cur
        ts.append(round(f[0], 2)); poss.append(f[5] if f[5] >= 0 else 2)
    for p in T["players"]: names.setdefault(p["id"], p["name"])
    trk = {"hz": T["frame_rate_hz"], "n": len(ts), "w": W, "ids": [p["id"] for p in T["players"]], "teams": [p["team"] for p in T["players"]],
           "no": [p["no"] for p in T["players"]], "pos": [p["pos"] for p in T["players"]], "ts": ts, "poss": "".join(map(str, poss)), "d": d,
           "note": "SIMULATED positions, not measured tracking"}
out = {"p1_end": p1_end, "duration": round(events[-1]["s"] + 20, 1), "lineups": lineups, "names": names, "events": events, "subs": subs, "trk": trk}
open(f"{B}/static/tracks.js", "w", encoding="utf-8").write("window.TRACKS = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("wrote static/tracks.js", len(events), "events,", len(subs), "subs, p1_end", p1_end)
