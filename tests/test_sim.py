"""Decision model + simulator tests: football logic, acceptance numbers, determinism."""
import json, math, os, subprocess, sys, tempfile, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from sim import probe
from sim.world import World
from sim.metrics import acceptance
from sim.generate import check_windows
from sim.model import xg_shot, offside_margin_att, goal_angle, pass_completion

B = os.path.join(os.path.dirname(__file__), "..")
EV = json.load(open(f"{B}/data/synthetic_match_900001.json"))
INFO = json.load(open(f"{B}/data/match_info.json"))
TRK = json.load(open(f"{B}/data/tracking_900001.json"))


@pytest.fixture(scope="module")
def world():
    return World(2026).run()


# ---------- pure model ----------
def test_xg_is_monotone_in_distance_and_angle():
    assert xg_shot(110, 40) > xg_shot(100, 40) > xg_shot(90, 40)
    assert xg_shot(110, 40) > xg_shot(110, 25)                  # central beats tight angle
    assert xg_shot(108, 40, "head") < xg_shot(108, 40)          # header penalty
    assert xg_shot(108, 40, pressure=1.0) < xg_shot(108, 40)
    assert xg_shot(108, 40, blockers=2) < xg_shot(108, 40)
    assert xg_shot(108, 40, penalty=True) == pytest.approx(0.78)


def test_pass_completion_drops_with_distance_pressure_lane():
    base = pass_completion(15, False, 0, 0, 0)
    assert pass_completion(45, False, 0, 0, 0) < base and pass_completion(15, False, 1, 0, 0) < base
    assert pass_completion(15, False, 0, 0, 1) < base and pass_completion(15, True, 0, 0, 0) < base


def test_offside_law():
    # second-last defender at x=100 (keeper at 118): receiver at 103 is offside, at 99 onside, level is onside
    assert offside_margin_att(103, 90, [118, 100, 95])[0] > 0
    assert offside_margin_att(99, 90, [118, 100, 95])[0] < 0
    assert offside_margin_att(100, 90, [118, 100, 95])[0] == 0
    # own half is never offside (line is never below the halfway line)
    assert offside_margin_att(55, 40, [118, 50, 45])[0] < 0
    # not offside if level with / behind the ball
    assert offside_margin_att(103, 105, [118, 100, 95])[0] < 0


# ---------- decision model ----------
def test_probe_central_close_shot_rate():
    r = probe.run(n=800, seed=7)
    assert r["clear_lane_situations"] > 500
    assert r["shoot_rate"] >= 0.60, r
    assert r["pass_rate"] <= 0.10, r


def test_decisions_are_explainable(world):
    assert len(world.decisions) > 1000
    for d in world.decisions[:500]:
        assert 1 <= len(d["top3"]) <= 3 and d["chosen"] in ("shoot", "pass", "carry", "dribble", "clear", "hold")
        assert d["top3"][0]["value"] >= d["top3"][-1]["value"] - 0.05 or d["chosen"] != d["top3"][0]["action"]


# ---------- the generated match (acceptance criteria) ----------
def test_acceptance_numbers(world):
    a = acceptance(world.events, world.decisions)
    assert a["passes_le12m_central"] < 15
    assert 20 <= a["shots_total"] <= 30
    assert a["completed_passes_to_offside_receivers"] == 0
    assert a["offside_events"] >= 4
    assert a["goals_with_var"] == a["goals"] == 5 and a["reds_with_var"] == a["reds"] == 1
    assert 2.0 <= a["xg_Kestrel"] <= 4.2 and 0.4 <= a["xg_Ironvale"] <= 2.0
    assert world.score == {"KES": 3, "IRO": 2}
    assert not world.beat_log.get("abandoned") and check_windows(world) == []


def test_committed_dataset_matches_simulator(world):
    assert len(EV) == len(world.events) == INFO["event_count"]
    assert [e["id"] for e in EV] == [e["id"] for e in world.events]


def test_offside_events_are_consistent():
    by = {e["id"]: e for e in EV}
    off = [e for e in EV if e["type"]["name"] == "Offside"]
    for o in off:
        pe = by[o["related_events"][0]]
        assert pe["type"]["name"] == "Pass"
        # either a "Pass Offside" free-kick, or a VAR-overturned goal
        assert pe["pass"].get("outcome", {}).get("name") == "Pass Offside" or o.get("x_overturned_by_var")
        assert pe["pass"]["x_offside_check"]["receiver_offside"] is True
    idx = {e["id"]: i for i, e in enumerate(EV)}
    for o in off:                                  # a Free Kick Awarded to the other team follows
        nxt = [e for e in EV[idx[o["id"]] + 1: idx[o["id"]] + 3] if e["type"]["name"] == "Free Kick Awarded"]
        assert nxt and nxt[0]["team"]["id"] != o["team"]["id"]


def test_set_pieces_and_cards_exist():
    types = {}
    for e in EV:
        if e["type"]["name"] == "Pass" and "type" in e["pass"]:
            types[e["pass"]["type"]["name"]] = types.get(e["pass"]["type"]["name"], 0) + 1
    assert all(types.get(k, 0) > 0 for k in ("Throw-in", "Goal Kick", "Corner", "Free Kick", "Kick Off"))
    names = {e["type"]["name"] for e in EV}
    assert {"Foul Won", "Free Kick Awarded", "Offside", "VAR", "Substitution"} <= names
    cards = [e["foul_committed"]["card"]["name"] for e in EV if e["type"]["name"] == "Foul Committed" and e.get("foul_committed", {}).get("card")]
    assert "Red Card" in cards and "Yellow Card" in cards


def test_var_reviews_and_overturns():
    var = [e["var"] for e in EV if e["type"]["name"] == "VAR"]
    outs = [v["outcome"]["name"] for v in var]
    assert "Goal Disallowed" in outs and "Penalty Overturned" in outs and "Red Card Upheld" in outs and "Penalty Confirmed" in outs
    d = next(v for v in var if v["outcome"]["name"] == "Goal Disallowed")
    assert 0 < d["x_offside_margin_m"] < 1.0           # marginal, VAR-worthy
    ids = {e["id"] for e in EV}
    assert all(v["subject_event_id"] in ids for v in var)


# ---------- tracking ----------
def test_tracking_is_consistent_with_events():
    fr, n = TRK["frames"], len(TRK["players"])
    assert TRK["frame_rate_hz"] == 2 and all(len(f) == 6 + 2 * n for f in fr)
    assert all(-0.5 <= f[6 + 2 * i] <= 120.5 for f in fr[::50] for i in range(n) if f[6 + 2 * i] >= 0)
    # frames are 0.5 s apart inside a period, strictly increasing
    ts = [f[0] for f in fr]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    # players do not teleport (a handful of fast chases are tolerated) and nobody runs faster than 11 m/s
    jumps = 0
    for a, b in zip(fr, fr[1:]):
        if b[1] != a[1]:
            continue
        for i in range(n):
            ax, ay, bx, by = a[6 + 2 * i], a[7 + 2 * i], b[6 + 2 * i], b[7 + 2 * i]
            if ax < 0 or bx < 0:
                continue
            d = math.hypot(bx - ax, by - ay)
            assert d <= 12.0
            jumps += d > 5.5
    assert jumps <= 40
    # a pass is played on the frame grid: the ball is at the passer's feet at that frame
    pid_idx = {p["id"]: i for i, p in enumerate(TRK["players"])}
    p1 = TRK["p1_end"]
    byt = {round(f[0], 2): f for f in fr}
    checked = 0
    for e in EV:
        if e["type"]["name"] != "Pass" or "outcome" in e["pass"]:
            continue
        t = round(e["timestamp"].split(":")[2] and (int(e["timestamp"][3:5]) * 60 + float(e["timestamp"][6:])) + (p1 if e["period"] == 2 else 0), 2)
        f = byt.get(t)
        if not f:
            continue
        i = pid_idx[e["player"]["id"]]
        px, py = f[6 + 2 * i], f[7 + 2 * i]
        assert math.hypot(px - f[2], py - f[3]) < 3.0
        checked += 1
        if checked > 80:
            break
    assert checked > 40


def test_defensive_line_moves_as_a_unit():
    """Back four of the defending side stay within ~14 m of each other in x while the other team attacks in open play."""
    fr = TRK["frames"]
    idx = {p["id"]: i for i, p in enumerate(TRK["players"])}
    back = [92002, 92003, 92004, 92005]
    spreads = []
    for f in fr[::10]:
        if f[5] != 0:
            continue
        xs = [f[6 + 2 * idx[b]] for b in back if f[6 + 2 * idx[b]] >= 0]
        if len(xs) == 4:
            spreads.append(max(xs) - min(xs))
    assert spreads and sum(spreads) / len(spreads) < 12


# ---------- determinism ----------
def test_same_seed_is_byte_identical():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        for d in (a, b):
            subprocess.run([sys.executable, "-m", "sim.generate", "--out", d, "--decisions"], cwd=B, check=True, capture_output=True)
        for f in ("synthetic_match_900001.json", "tracking_900001.json", "match_info.json", "decisions_900001.json"):
            assert hashlib.sha256(open(f"{a}/{f}", "rb").read()).digest() == hashlib.sha256(open(f"{b}/{f}", "rb").read()).digest(), f
        # and identical to the committed data
        for f in ("synthetic_match_900001.json", "tracking_900001.json", "match_info.json"):
            assert open(f"{a}/{f}", "rb").read() == open(f"{B}/data/{f}", "rb").read(), f


def test_other_seed_still_hits_all_beats():
    w = World(11).run()
    assert w.score == {"KES": 3, "IRO": 2} and not w.beat_log.get("abandoned") and check_windows(w) == []


def _frame_at(t, byt):
    return byt.get(round(t, 2))


def test_offside_margins_match_the_tracking_frames():
    """The recorded offside margin of every pass into the opponent half can be re-derived from the tracking frame at the moment of the pass."""
    fr, n = TRK["frames"], len(TRK["players"])
    team = [p["team"] for p in TRK["players"]]
    ids = [p["id"] for p in TRK["players"]]
    byt = {round(f[0], 2): f for f in fr}
    p1 = TRK["p1_end"]
    checked = bad = 0
    for e in EV:
        if e["type"]["name"] != "Pass" or "x_offside_check" not in e["pass"] or e["pass"].get("recipient") is None:
            continue
        chk = e["pass"]["x_offside_check"]
        if chk.get("exempt_restart"):
            continue
        ts = e["timestamp"]
        t = round(int(ts[3:5]) * 60 + float(ts[6:]) + (p1 if e["period"] == 2 else 0), 2)
        f = byt.get(t)
        if not f:
            continue
        att = "KES" if e["team"]["name"].startswith("Kest") else "IRO"
        sgn = 1 if att == "KES" else -1
        ax = lambda i: f[6 + 2 * i] if sgn > 0 else 120 - f[6 + 2 * i]
        r = ids.index(e["pass"]["recipient"]["id"]); p = ids.index(e["player"]["id"])
        defs = sorted((ax(i) for i in range(n) if team[i] != att and f[6 + 2 * i] >= 0), reverse=True)
        line = max(defs[1], ax(p), 60.0)
        m = ax(r) - line
        checked += 1
        bad += abs(m - chk["receiver_margin_m"]) > 0.35
    assert checked > 100 and bad <= max(2, checked * 0.02), (checked, bad)
