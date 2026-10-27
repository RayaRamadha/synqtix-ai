"""Stages 2-3 (Interpret + Explain, deterministic): turn events into language-neutral Moments.

A Moment = what happened + computed metrics + explanation factors (type + params + evidence keys).
No prose here: wording is done in render.py, so the same Moment can be rendered per audience/language.
Every number that may appear in text is exported in `metrics` with provenance.
"""
from .state import MatchState

BIG_CHANCE_XG = 0.30          # conventional "big chance" threshold (not tuned on the answer key)
WINDOW_S = 600                # 10-minute control window
RED_WINDOW_S = 2400           # how long after a red card we describe its effect


def M(key, label, value, unit, scope, entity=None, calc="", ids=None, src="computed", baseline=None, delta=None):
    m = {"key": key, "label": label, "value": value, "unit": unit, "scope": scope,
         "source": {"type": src}}
    if calc:
        m["source"]["calc"] = calc
    if ids:
        m["source"]["event_ids"] = ids
    if entity is not None:
        m["entity"] = str(entity)
    if baseline is not None:
        m["baseline"] = baseline
    if delta is not None:
        m["delta"] = delta
    return m


def win(m, s):
    m["window"] = {"from_stream_s": round(s - WINDOW_S, 1), "to_stream_s": round(s, 1)}
    return m


def clock(e) -> str:
    m = e["minute"]
    if e["period"] == 1 and m > 45:
        return f"45+{m - 45}'"
    if e["period"] == 2 and m > 90:
        return f"90+{m - 90}'"
    return f"{m}'"


def _factor(ftype, params, keys):
    return {"type": ftype, "params": params, "metric_keys": keys}


def _base(st: MatchState, e, kind, rule):
    s = st.stream_s(e)
    return {"kind": kind, "rule_id": rule, "event_id": e["id"], "event_index": e["index"],
            "period": e["period"], "minute": e["minute"], "second": e["second"],
            "clock": clock(e), "stream_s": round(s, 3), "team": st.code(e["team"]),
            "metrics": [], "factors": [], "players": [], "annotations": [], "facts": {}}


def _shot_common(st, e, mo, code, opp, s):
    sh = e["shot"]
    eid = e["id"]
    mo["metrics"] += [
        M("shot_xg", "Shot xG", round(sh["statsbomb_xg"], 2), "xg", "player", e["player"]["id"],
          "shot.statsbomb_xg", [eid], "event_field"),
        M("team_xg_so_far", f"{code} xG so far", round(st.xg[code], 2), "xg", "team", code,
          "sum(shot.statsbomb_xg, team, up to event)"),
        M("team_shots_so_far", f"{code} shots so far", st.shots[code], "count", "team", code,
          "count(Shot, team, up to event)"),
        M("team_goals", f"{code} goals", st.score[code], "count", "team", code, "running score"),
    ]
    if sh.get("x_shot_speed_kmh") is not None:
        mo["metrics"].append(M("shot_speed", "Shot speed", sh["x_shot_speed_kmh"], "km/h", "player",
                               e["player"]["id"], "shot.x_shot_speed_kmh", [eid], "event_field"))
    mo["players"].append({"player_id": e["player"]["id"], "name": e["player"]["name"], "team": code,
                          "role": "actor", "tag_metric_keys": [k for k in ("shot_xg", "shot_speed")
                                                               if any(m["key"] == k for m in mo["metrics"])]})
    kp = st.by_id.get(sh.get("key_pass_id", ""))
    if kp and kp["player"]["id"] != e["player"]["id"]:
        mo["players"].append({"player_id": kp["player"]["id"], "name": kp["player"]["name"], "team": code, "role": "assist"})
    end = sh.get("end_location")
    if e.get("location") and end:
        mo["annotations"].append({"kind": "arrow", "from": e["location"][:2], "to": end[:2], "label": sh["outcome"]["name"]})
    mo["facts"].update(shooter=e["player"]["name"], shooter_id=e["player"]["id"], outcome=sh["outcome"]["name"],
                       play_pattern=e["play_pattern"]["name"], body_part=sh["body_part"]["name"],
                       shot_type=sh["type"]["name"])
    # transition vulnerability: opponent had the ball but conceded a counter chance
    if e["play_pattern"]["name"] == "From Counter":
        share = st.pass_share(opp, s - WINDOW_S, s)
        if share is not None and share >= 55:
            mo["metrics"].append(win(M("opp_pass_share_10m", f"{opp} pass share, last 10 min", share, "pct", "window", opp,
                                       "passes by team / all passes in window"), s))
            mo["factors"].append(_factor("transition_vulnerability", {"team": code, "opp": opp, "share": share},
                                         ["opp_pass_share_10m"]))


def _red_effect(st, s, code, mo):
    """Pressure shift for the team that gained a numerical advantage (honest about direction)."""
    if not st.red or st.red[1] == code or s - st.red[0] > RED_WINDOW_S or s <= st.red[0]:
        return
    rs = st.red[0]
    since = st.pass_share(code, rs, s)
    before = st.pass_share(code, rs - WINDOW_S, rs)
    if since is None or before is None:
        return
    delta = round(since - before, 1)
    xg_since = st.xg_between(code, rs, s)
    mins = int((s - rs) // 60)
    mo["metrics"] += [
        M("pass_share_since_red", f"{code} pass share since red card", since, "pct", "window", code,
          "passes by team / all passes since red card", baseline=before, delta=delta),
        M("pass_share_before_red", f"{code} pass share, 10 min before red card", before, "pct", "window", code,
          "passes by team / all passes in 10 min before red card"),
        M("xg_since_red", f"{code} xG since red card", xg_since, "xg", "window", code, "sum(shot.statsbomb_xg) since red card"),
        M("minutes_since_red", "Minutes since red card", mins, "count", "match", None, "floor((now - red card)/60)"),
    ]
    direction = "rose" if delta >= 3 else "fell" if delta <= -3 else "held"
    mo["factors"].append(_factor("tactical_pressure_shift",
                                 {"team": code, "since": since, "before": before, "dir": direction, "xg": xg_since, "mins": mins},
                                 ["pass_share_since_red", "pass_share_before_red", "xg_since_red", "minutes_since_red"]))


def _var_moment(st, e):
    """VAR decisions that CHANGE something (goal ruled out, penalty overturned). Confirmations are not moments."""
    v = e.get("var") or {}
    out_name = (v.get("outcome") or {}).get("name", "")
    if out_name not in ("Goal Disallowed", "Penalty Overturned"):
        return None
    subj = st.by_id.get(v.get("subject_event_id", ""))
    code = st.code(e["team"])
    sub = "goal_disallowed" if out_name == "Goal Disallowed" else "penalty_overturned"
    mo = _base(st, e, "var_overturn", "var." + sub)
    eid = e["id"]
    if sub == "goal_disallowed":
        margin = v.get("x_offside_margin_m")
        if margin is None or not subj or "shot" not in subj:
            return None
        mo["metrics"] += [
            M("shot_xg", "Disallowed shot xG", round(subj["shot"]["statsbomb_xg"], 2), "xg", "player", subj["player"]["id"], "shot.statsbomb_xg", [subj["id"]], "event_field"),
            M("offside_margin_m", "Offside margin when the pass was played", round(abs(margin), 2), "m", "player", subj["player"]["id"], "var.x_offside_margin_m", [eid], "event_field"),
            M("var_review_s", "VAR review length", round(v.get("duration_s", 0)), "s", "match", None, "var.duration_s", [eid], "event_field"),
            M("score_now", "Score (unchanged)", st.score_str(), "text", "match", None, "running score"),
        ]
        mo["players"].append({"player_id": subj["player"]["id"], "name": subj["player"]["name"], "team": code, "role": "actor", "tag_metric_keys": ["shot_xg"]})
        mo["factors"].append(_factor("var_review", {"sub": sub, "team": code, "margin": round(abs(margin), 2), "secs": round(v.get("duration_s", 0)), "score": st.score_str()},
                                     ["offside_margin_m", "var_review_s", "score_now"]))
        mo["facts"].update(sub=sub, shooter=subj["player"]["name"], shooter_id=subj["player"]["id"], score=st.score_str())
    else:
        contact = v.get("x_contact_outside_box_m")
        if contact is None:
            return None
        mo["metrics"] += [
            M("contact_outside_box_m", "Contact point outside the penalty area", round(contact, 2), "m", "match", None, "var.x_contact_outside_box_m", [eid], "event_field"),
            M("var_review_s", "VAR review length", round(v.get("duration_s", 0)), "s", "match", None, "var.duration_s", [eid], "event_field"),
            M("score_now", "Score (unchanged)", st.score_str(), "text", "match", None, "running score"),
        ]
        if subj and subj.get("player"):
            mo["players"].append({"player_id": subj["player"]["id"], "name": subj["player"]["name"], "team": st.code(subj["team"]), "role": "actor"})
        mo["factors"].append(_factor("var_review", {"sub": sub, "team": code, "contact": round(contact, 2), "secs": round(v.get("duration_s", 0)), "score": st.score_str()},
                                     ["contact_outside_box_m", "var_review_s", "score_now"]))
        mo["facts"].update(sub=sub, score=st.score_str())
    return mo


OFFSIDE_PATTERN_N = 3


def _offside_pattern(st, e):
    """Third offside flag against the same team inside 10 minutes (fires once per run: only when the count hits exactly N)."""
    code = st.code(e["team"])
    s = st.stream_s(e)
    n = st.offsides_against(code, s - WINDOW_S, s)
    if n != OFFSIDE_PATTERN_N:
        return None
    mo = _base(st, e, "pattern", "pattern.offside_flags")
    mo["metrics"].append(win(M("offsides_10m", f"{code} offside flags, last 10 min", n, "count", "window", code, "count(Offside events, team, window)",
                               [o["event_id"] for o in st.offsides if o["team"] == code][-n:]), s))
    mo["factors"].append(_factor("offside_pattern", {"team": code, "n": n}, ["offsides_10m"]))
    if e.get("player"):
        mo["players"].append({"player_id": e["player"]["id"], "name": e["player"]["name"], "team": code, "role": "actor"})
    if e.get("location"):
        mo["annotations"].append({"kind": "point", "from": e["location"][:2], "label": "offside"})
    mo["facts"].update(team=code, n=n)
    return mo


def detect(st: MatchState, e: dict) -> list[dict]:
    """Call AFTER st.update(e). Returns zero or more neutral Moments."""
    t = e["type"]["name"]
    out = []
    s = st.stream_s(e)
    if t == "VAR":
        mo = _var_moment(st, e)
        return [mo] if mo else out
    if t == "Offside":
        mo = _offside_pattern(st, e)
        return [mo] if mo else out
    if t == "Shot":
        sh = e["shot"]
        code = st.code(e["team"])
        opp = st.other(code)
        outcome = sh["outcome"]["name"]
        xg = sh["statsbomb_xg"]
        if outcome == "Goal Disallowed":              # ball in the net but ruled out: the VAR event carries the story
            return out
        if outcome == "Goal":
            mo = _base(st, e, "goal", "goal.scored")
        elif xg >= BIG_CHANCE_XG and outcome == "Saved":
            mo = _base(st, e, "big_save", "chance.big_save")
        elif xg >= BIG_CHANCE_XG:
            mo = _base(st, e, "big_chance", "chance.big_miss")
        else:
            return out
        _shot_common(st, e, mo, code, opp, s)
        f = mo["facts"]
        score_after = dict(st.score)
        if outcome == "Goal":
            before = {**score_after, code: score_after[code] - 1}
            b_diff, a_diff = before[code] - before[opp], score_after[code] - score_after[opp]
            pid = e["player"]["id"]
            n = st.player_goals[pid]
            situation = ("opener" if sum(score_after.values()) == 1 else "equalizer" if a_diff == 0
                         else "go_ahead" if b_diff <= 0 < a_diff else "pull_back" if a_diff < 0 else "extend")
            sub = ("hat_trick" if n == 3 else "penalty" if sh["type"]["name"] == "Penalty" else situation)
            f.update(sub=sub, situation=situation, goals=n)
            mo["metrics"].append(M("score_after", "Score after goal", st.score_str(), "text", "match", None, "running score", [e["id"]]))
            mo["metrics"].append(M("scorer_goals", f"{e['player']['name']} goals", n, "count", "player", pid, "count(Shot outcome=Goal, player)"))
            gap = abs(a_diff)
            mo["metrics"].append(M("lead_margin", "Margin after goal", gap, "count", "match", None, "abs(goal difference)"))
            if n == 3:
                types = st.player_goal_types[pid]
                mo["metrics"].append(M("scorer_goal_types", "Goal types", ", ".join(types), "text", "player", pid, "classify(goal events)"))
                f["goal_types"] = types
                mo["factors"].append(_factor("milestone", {"player": e["player"]["name"], "goals": n, "types": types},
                                             ["scorer_goals", "scorer_goal_types"]))
            deficit = st.max_deficit[code]
            if situation in ("equalizer", "go_ahead", "pull_back") and deficit >= 1:
                mo["metrics"].append(M("deficit_overturned", f"Largest {code} deficit", deficit, "count", "team", code, "max(opponent goals - team goals) so far"))
                mo["factors"].append(_factor("game_state", {"situation": situation, "team": code, "deficit": deficit, "score": st.score_str(), "gap": gap},
                                             ["deficit_overturned", "score_after"] + (["lead_margin"] if situation == "pull_back" else [])))
            elif situation == "extend":
                mo["factors"].append(_factor("game_state", {"situation": "extend", "team": code, "gap": gap}, ["lead_margin"]))
            _red_effect(st, s, code, mo)
            tg = st.score[code]
            if tg >= 2 and tg - st.xg[code] >= 1.0:
                mo["factors"].append(_factor("finishing_efficiency", {"team": code, "goals": tg, "shots": st.shots[code], "xg": round(st.xg[code], 2)},
                                             ["team_goals", "team_shots_so_far", "team_xg_so_far"]))
            f["score"] = st.score_str()
        else:
            # keeper comes from the lineup (the keeper's own event arrives AFTER the shot in a live stream)
            kid, kname = st.keepers.get(opp, (None, None))
            if kname:
                f["keeper"], f["keeper_id"] = kname, kid
                mo["players"].append({"player_id": kid, "name": kname, "team": opp, "role": "keeper"})
            now, if_scored = st.score_str(), st.score_str({**score_after, code: score_after[code] + 1})
            rel = "ahead" if score_after[code] + 1 > score_after[opp] else "level" if score_after[code] + 1 == score_after[opp] else "behind"
            f.update(score_now=now, score_if=if_scored, relation=rel)
            if outcome == "Saved" and rel in ("ahead", "level"):
                mo["metrics"] += [M("score_now", "Score now", now, "text", "match", None, "running score"),
                                  M("score_if_scored", "Score if the shot went in", if_scored, "text", "match", None, "running score + 1 for shooting team")]
                mo["factors"].append(_factor("game_state", {"situation": "counterfactual", "team": code, "relation": rel, "score_now": now, "score_if": if_scored},
                                             ["score_now", "score_if_scored"]))
            elif outcome != "Saved":
                share = st.pass_share(code, s - WINDOW_S, s)
                if share is not None and share >= 55 and st.xg[code] - st.score[code] >= 0.5:
                    mo["metrics"].append(win(M("team_pass_share_10m", f"{code} pass share, last 10 min", share, "pct", "window", code, "passes by team / all passes in window"), s))
                    mo["factors"].append(_factor("control_vs_chaos", {"team": code, "share": share, "xg": round(st.xg[code], 2), "goals": st.score[code]},
                                                 ["team_pass_share_10m", "team_xg_so_far", "team_goals"]))
            f["outcome_label"] = outcome
        out.append(mo)
    elif t == "Foul Committed" and (e.get("foul_committed") or {}).get("card", {}).get("name") == "Red Card":
        code = st.code(e["team"])
        opp = st.other(code)
        mo = _base(st, e, "red_card", "discipline.red_card")
        diff = st.score[code] - st.score[opp]
        state = "leading" if diff > 0 else "trailing" if diff < 0 else "level"
        share = st.pass_share(opp, s - WINDOW_S, s)
        mo["metrics"] += [
            M("players_on_pitch", f"{code} players on pitch", st.on_pitch[code], "count", "team", code, "11 - red cards", [e["id"]]),
            M("score_now", "Score at red card", st.score_str(), "text", "match", None, "running score"),
        ]
        mo["factors"].append(_factor("game_state", {"situation": "red", "team": code, "n": st.on_pitch[code], "state": state, "score": st.score_str()},
                                     ["players_on_pitch", "score_now"]))
        if share is not None:
            mo["metrics"].append(win(M("opp_pass_share_10m", f"{opp} pass share, last 10 min", share, "pct", "window", opp, "passes by team / all passes in window"), s))
            mo["factors"].append(_factor("control_vs_chaos", {"team": opp, "share": share, "context": "red"}, ["opp_pass_share_10m"]))
        mo["players"].append({"player_id": e["player"]["id"], "name": e["player"]["name"], "team": code, "role": "actor"})
        mo["facts"].update(player=e["player"]["name"], player_id=e["player"]["id"], team_down=code, state=state, n=st.on_pitch[code], score=st.score_str())
        out.append(mo)
    return out
