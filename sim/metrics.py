"""Acceptance metrics measured from a generated match (events + decision log). Used by the generator report and the tests."""
import math, collections


def acceptance(events, decisions):
    ev = events
    by_id = {e["id"]: e for e in ev}
    out = {}
    # 1) shot rate for central attackers within 12 m with a clear lane (decision log, open play, ball carrier decisions)
    sit = [d for d in decisions if "xg_here" in d and not d.get("beat") and d.get("d_goal", 99) <= 12 and abs(d["y"] - 40) <= 14 and d.get("lane_clear")]
    out["central_close_situations"] = len(sit)
    out["central_close_shot_rate"] = round(sum(1 for d in sit if d["chosen"] == "shoot") / max(1, len(sit)), 3)
    # 2) passes starting <=12 m from goal, central (event data)
    near = [e for e in ev if e["type"]["name"] == "Pass" and math.hypot(120 - e["location"][0], 40 - e["location"][1]) <= 12 and abs(e["location"][1] - 40) <= 14]
    out["passes_le12m_central"] = len(near)
    out["passes_in_box"] = sum(1 for e in ev if e["type"]["name"] == "Pass" and e["location"][0] >= 102 and 18 <= e["location"][1] <= 62)
    out["shots_in_box"] = sum(1 for e in ev if e["type"]["name"] == "Shot" and e["location"][0] >= 102 and 18 <= e["location"][1] <= 62)
    # 3) shots
    shots = [e for e in ev if e["type"]["name"] == "Shot"]
    out["shots_total"] = len(shots)
    for t in ("Kestrel Rovers", "Ironvale Athletic"):
        s = [e for e in shots if e["team"]["name"] == t]
        out[f"shots_{t.split()[0]}"] = len(s)
        out[f"xg_{t.split()[0]}"] = round(sum(e["shot"]["statsbomb_xg"] for e in s), 2)
    # 4) offside integrity: a completed pass whose receiver was in an offside position must be followed by an Offside event that references it
    off_ev = [e for e in ev if e["type"]["name"] == "Offside"]
    referenced = {r for e in off_ev for r in e.get("related_events", [])}
    bad = [e for e in ev if e["type"]["name"] == "Pass" and "outcome" not in e["pass"] and e["pass"].get("x_offside_check", {}).get("receiver_offside")
           and e["id"] not in referenced]
    out["completed_passes_to_offside_receivers"] = len(bad)
    out["offside_events"] = len(off_ev)
    out["pass_offside_outcomes"] = sum(1 for e in ev if e["type"]["name"] == "Pass" and e["pass"].get("outcome", {}).get("name") == "Pass Offside")
    # 5) VAR coverage
    var = [e for e in ev if e["type"]["name"] == "VAR"]
    rev = {e["var"]["subject_event_id"] for e in var}
    goals = [e for e in shots if e["shot"]["outcome"]["name"] == "Goal"]
    reds = [e for e in ev if e["type"]["name"] == "Foul Committed" and e.get("foul_committed", {}).get("card", {}).get("name") == "Red Card"]
    out["goals"] = len(goals)
    out["goals_with_var"] = sum(1 for g in goals if g["id"] in rev)
    out["reds"] = len(reds)
    out["reds_with_var"] = sum(1 for r in reds if r["id"] in rev)
    out["var_events"] = len(var)
    out["var_overturned"] = [e["var"]["outcome"]["name"] for e in var if "Overturned" in e["var"]["outcome"]["name"] or "Disallowed" in e["var"]["outcome"]["name"]]
    out["yellow_cards"] = sum(1 for e in ev if e["type"]["name"] == "Foul Committed" and e.get("foul_committed", {}).get("card", {}).get("name") == "Yellow Card")
    out["passes"] = sum(1 for e in ev if e["type"]["name"] == "Pass")
    out["events"] = len(ev)
    return out
