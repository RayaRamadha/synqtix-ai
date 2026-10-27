"""Explainable utility model for the player in possession.

Every option gets a value in "expected xG-like units". The player picks argmax(value + small Gumbel noise) from a seeded RNG,
and the top-3 alternatives with their values and key terms are returned so every decision can be logged.

  shoot  : xG(distance, angle, body part, pressure, first-time, lane blockers) x role shot multiplier (+ patience term for long shots)
  pass   : p_complete x CF x threat(receiver) - (1-p) x loss + progression bonus      (offside is a HARD constraint, see below)
  carry  : (1 - p_tackle) x CF x threat(destination) + progression bonus
  dribble: p_beat x CF x threat(beyond defender) - (1-p) x loss
  clear  : fixed value that rises with pressure in the own third
  hold   : small value (shield the ball)
CF = P["conv"] is the share of positional threat that is realised as a shot after the next action (the "delay factor"),
so shooting yourself beats passing to a team-mate in the same spot, and only a clearly better-placed team-mate wins.

Offside law: the passer *perceives* the receiver's offside margin with Gaussian noise (sigma_off). Receivers perceived offside are
removed from the option list. Perception errors (and defensive step-ups at the moment of the pass, applied by the world) are what
produce the occasional real offside.
"""
import math
from .model import (P, hyp, threat, xg_shot, pressure_at, lane_block, lane_blockers, pass_completion, offside_margin_att, GOAL)

SHOT_COST = 0.046          # a shot gives the ball away: expected value of the lost possession
SHOT_FLOOR = 0.05          # below this xG a shot is only taken as a speculative effort after long possession
PASS_KEEP = 0.012          # value of keeping the ball with a completed pass
ROLE_SHOT = {"ATT": 1.2, "MID": 0.95, "DEF": 0.55, "GK": 0.0}


def loss_cost(x):
    return 0.004 + 0.06 * (1 - x / 120.0) ** 2


def flight_steps(dist, high):
    v = 22.0 if high else 14.0
    return max(1, math.ceil(dist / (v * 0.5)))


def plan_pass(me, m, opps, ctx, rng):
    """Returns dict describing a pass from `me` to mate `m` (all attacking frame) or None."""
    d = hyp(me["x"], me["y"], m["x"], m["y"])
    if d < 4.5 or d > 62:
        return None
    wide = abs(me["y"] - 40) > 17
    cross = me["x"] > 78 and wide and m["x"] >= 98 and 20 <= m["y"] <= 60 and d > 12
    through = (not cross) and m["role"] == "ATT" and m["x"] > me["x"] + 9 and m["x"] > 55
    high = cross or d > 34 or (ctx.get("direct", 0) > 0 and d > 26 and rng.random() < 0.5)
    steps = flight_steps(d, high)
    T = steps * 0.5
    lead_f = 4.0 if through else 0.0
    ex = min(118.0, max(2.0, m["x"] + m["vx"] * T * 0.7 + lead_f))
    ey = min(78.0, max(2.0, m["y"] + m["vy"] * T * 0.7))
    # offside (hard law constraint on the option list)
    def_xs = [o[0] for o in opps]
    margin, line = offside_margin_att(m["x"], me["x"], def_xs)
    perceived = margin + rng.gauss(0, ctx["sigma_off"])
    if m["x"] > 60 and perceived > -0.3 and not ctx.get("no_offside_filter"):
        return {"excluded": "offside", "to": m["id"], "margin": margin}
    o2 = [(o[0], o[1]) for o in opps]
    lane = lane_block(me["x"], me["y"], ex, ey, o2) * (1.5 if ex > 92 else 1.0)
    pr = pressure_at(ex, ey, o2)
    pc = pass_completion(hyp(me["x"], me["y"], ex, ey), high, ctx["press_me"], pr, lane, ctx["pass_skill"], through)
    if ex > 102 and 14 < ey < 66:                       # a compact block makes box entries hard
        pc *= ctx.get("box_entry", 0.42)
    gain = threat(ex, ey) * (1 - 0.45 * pr)
    prog = ctx["fwd"] * 0.02 * max(0.0, ex - me["x"]) / 30.0
    value = pc * (P["conv"] * gain + prog + PASS_KEEP) - (1 - pc) * loss_cost(ex)
    if cross:
        value *= 0.9 + ctx.get("cross_bias", 0.0)
    return {"to": m["id"], "value": value, "pc": pc, "end": (ex, ey), "kind": "cross" if cross else "through" if through else "pass",
            "high": high, "steps": steps, "margin": margin, "line": line, "lane": lane, "press_recv": pr, "dist": d}


def decide(ctx, me, mates, opps, rng, only_pass=False, shoot_only_fk=False):
    """ctx: dict(pass_skill, fwd, sigma_off, temp, press_me, counter, direct, patience_s, first_time, header, beat, shot_mult, min_shot_d)
    Returns (choice, top3, notes). choice = dict(action=..., ...)."""
    opt = []
    notes = {"excluded_offside": 0}
    press_me = pressure_at(me["x"], me["y"], [(o[0], o[1]) for o in opps])
    ctx = dict(ctx, press_me=press_me)
    fwd = ctx["fwd"] * (1.4 if ctx.get("counter") else 1.0)
    ctx["fwd"] = fwd
    d_goal = hyp(me["x"], me["y"], *GOAL)
    nogk = [(o[0], o[1]) for o in opps if not o[2]]
    beat = ctx.get("beat")
    is_shooter = bool(beat and beat["shooter"] == me["id"])
    xg_here = 0.0
    lane_clear = False
    if not only_pass and d_goal <= 34 and ROLE_SHOT.get(me["role"], 0) > 0:
        part = "head" if ctx.get("header") else "foot"
        blk = lane_blockers(me["x"], me["y"], GOAL[0], GOAL[1], nogk)
        lane_clear = blk == 0
        xg_here = xg_shot(me["x"], me["y"], part, press_me, ctx.get("first_time", False), blk)
        v = xg_here * ROLE_SHOT[me["role"]] * ctx["shot_mult"] - SHOT_COST * (1 - xg_here) + 0.003
        if d_goal > 16:
            v += 0.004 * min(ctx["patience_s"], 30) / 10.0 * ctx["patience"] * (1 if d_goal <= 29 else 0.5)
        floor = ctx.get("shot_floor", SHOT_FLOOR)
        speculative = xg_here < floor and ctx["patience_s"] > 12 and rng.random() < 0.15
        if xg_here >= floor or speculative or ctx.get("header") or ctx.get("first_time") and xg_here >= 0.03:
            opt.append({"action": "shoot", "value": v, "xg": xg_here, "blockers": blk})
    for m in mates:
        pl = plan_pass(me, m, opps, ctx, rng)
        if not pl:
            continue
        if "excluded" in pl:
            notes["excluded_offside"] += 1
            continue
        if m["role"] == "GK" and not (me["x"] < 38 and press_me > 0.4):
            continue
        v = pl["value"]
        if beat and beat["shooter"] == m["id"] and not is_shooter:
            v += 0.12 * pl["pc"]
        opt.append({"action": "pass", **pl})
    if not only_pass and me["role"] != "GK":
        for dx, dy in ((6, 0), (5, 5), (5, -5), (3, 8), (3, -8)):
            tx = me["x"] + dx
            ty = min(77.0, max(3.0, me["y"] + dy))
            if is_shooter and beat.get("min_xg"):                  # scripted chance: run at goal
                tx, ty = min(beat.get("cap_x", 118.0), me["x"] + 7), me["y"] + max(-5, min(5, 40 - me["y"]))
            pr = pressure_at(tx, ty, [(o[0], o[1]) for o in opps])
            pt = 0.10 + 0.28 * pr + (ctx.get("box_carry_risk", 0.4) if (tx > 102 and 14 < ty < 66) else 0.0)
            v = (1 - pt) * (P["conv"] * threat(tx, ty) + fwd * 0.02 * dx / 30.0) - pt * loss_cost(me["x"]) - 0.004 - (0.012 if ctx.get("carried_last") else 0.0)
            if is_shooter and beat.get("min_xg"):
                v += 0.12
            opt.append({"action": "carry", "value": v, "end": (min(118.0, tx), ty), "p_tackle": pt})
        ahead = [o for o in opps if not o[2] and 0 < o[0] - me["x"] < 4.5 and abs(o[1] - me["y"]) < 3.5]
        if ahead and me["x"] > 40:
            pb = 0.42 * ctx.get("dribble", 1.0)
            dst = (min(118.0, ahead[0][0] + 4.0), ahead[0][1])
            opt.append({"action": "dribble", "value": pb * P["conv"] * threat(*dst) - (1 - pb) * 2 * loss_cost(me["x"]),
                        "end": dst, "p_beat": pb})
    if not only_pass and (me["x"] < 34 or me["role"] == "GK") and press_me > 0.35:
        opt.append({"action": "clear", "value": 0.009 + 0.03 * press_me + (0.01 if ctx.get("direct") else 0)})
    if not only_pass and me["role"] != "GK":
        opt.append({"action": "hold", "value": 0.002 + 0.4 * P["conv"] * threat(me["x"], me["y"]) - 0.01 * press_me})
    # scripted-beat gate: designated shooter only shoots from a good-enough position
    if is_shooter and beat.get("min_xg"):
        for o in opt:
            if o["action"] == "shoot":
                o["value"] = 2.0 if xg_here >= beat["min_xg"] else -9.0
    elif is_shooter and beat.get("shoot_now"):
        for o in opt:
            if o["action"] == "shoot":
                o["value"] += 1.0
    if not opt:
        return {"action": "hold"}, [], notes
    for o in opt:
        o["u"] = o["value"] + ctx["temp"] * -math.log(-math.log(max(1e-9, rng.random())))
    opt.sort(key=lambda o: -o["u"])
    top3 = [{"action": o["action"], **({"to": o["to"]} if "to" in o else {}), "value": round(o["value"], 4),
             **({"xg": round(o["xg"], 3)} if "xg" in o else {}), **({"p_complete": round(o["pc"], 3)} if "pc" in o else {})}
            for o in opt[:3]]
    notes.update(xg_here=round(xg_here, 3), lane_clear=lane_clear, d_goal=round(d_goal, 1), pressure=round(press_me, 2))
    return opt[0], top3, notes
