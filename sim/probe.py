"""Decision-model probe: runs the production `decide()` over randomised close-range situations (same code path as the match simulator).
Measures: share of 'central attacker <=12 m, clear shooting lane' situations that end in a shot, and how the passes chosen there are justified."""
import random, math
from .decision import decide
from .model import xg_shot, GOAL, hyp

CTX = dict(pass_skill=0.8, fwd=0.9, sigma_off=0.8, temp=0.006, counter=False, direct=0, patience_s=0.0, patience=1.0, shot_mult=1.0, dribble=1.0,
           first_time=False, header=False)


def situation(rng, central_only=True):
    while True:
        x, y = rng.uniform(106, 118.5), rng.uniform(40 - 14, 40 + 14)
        if hyp(x, y, *GOAL) <= 12:
            break
    me = {"id": 1, "x": x, "y": y, "vx": 0, "vy": 0, "role": "ATT", "pos": "Center Forward"}
    opps = [(rng.uniform(112, 119), 40 + rng.uniform(-2, 2), True)]                      # keeper
    for _ in range(rng.randint(3, 8)):                                                    # outfield defenders, none on the shot lane
        ox, oy = rng.uniform(88, 117), rng.uniform(8, 72)
        if ox > x and abs(oy - 40) < 4:
            oy += 9 * (1 if rng.random() < .5 else -1)
        opps.append((ox, oy, False))
    # lane must be clear
    for ox, oy, gk in opps[1:]:
        if x < ox and abs((oy - y) - (40 - y) * (ox - x) / max(1e-6, 120 - x)) < 1.7:
            oy += 4
    mates = []
    for i in range(2, 11):
        mx, my = rng.uniform(55, 116), rng.uniform(6, 74)
        mates.append({"id": i, "x": mx, "y": my, "vx": rng.uniform(0, 3), "vy": rng.uniform(-1, 1), "role": rng.choice(["ATT", "MID", "ATT", "DEF"]), "pos": "x"})
    return me, mates, opps


def run(n=800, seed=7):
    rng = random.Random(seed)
    shoot = passes = clear = 0
    other = {}
    justified = 0
    for _ in range(n):
        me, mates, opps = situation(rng)
        ctx = dict(CTX)
        ch, top3, notes = decide(ctx, me, mates, opps, rng)
        if not notes.get("lane_clear"):
            continue
        clear += 1
        if ch["action"] not in ("shoot", "pass"):
            other[ch["action"]] = other.get(ch["action"], 0) + 1
        if ch["action"] == "shoot":
            shoot += 1
        elif ch["action"] == "pass":
            passes += 1
            mate = next(m for m in mates if m["id"] == ch["to"])
            if xg_shot(ch["end"][0], ch["end"][1]) >= 1.6 * xg_shot(me["x"], me["y"]):
                justified += 1
    return {"sampled": n, "clear_lane_situations": clear, "shoot_rate": round(shoot / max(1, clear), 3), "pass_rate": round(passes / max(1, clear), 3),
            "other_actions": other, "passes_justified_by_much_better_teammate": justified}


if __name__ == "__main__":
    print(run())
