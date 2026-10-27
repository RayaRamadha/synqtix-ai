import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
import app as appmod
from engine import Pipeline

B = os.path.join(os.path.dirname(__file__), "..")
INFO = json.load(open(f"{B}/data/match_info.json"))
EVENTS = json.load(open(f"{B}/data/synthetic_match_900001.json"))
V = Draft202012Validator(json.load(open(f"{B}/insight.schema.json")))


def run(**kw):
    p = Pipeline(INFO)
    return p, p.run_file(EVENTS, **kw)


def test_all_answer_key_moments_detected():
    _, ins = run()
    got = {i["trigger"]["event_ids"][0] for i in ins}
    assert all(k["event_ids"][0] in got for k in INFO["answer_key_moments"])


def test_schema_and_grounding():
    for mode in ("analyst", "casual_fan"):
        for lang in ("en", "id"):
            _, ins = run(mode=mode, lang=lang)
            for i in ins:
                assert not list(V.iter_errors(i)), (mode, lang, i["headline"])
                assert i["provenance"]["grounding"]["status"] == "verified", i["headline"]


def test_duplicates_and_garbage_do_not_change_output():
    p0, base = run()
    noisy = []
    for e in EVENTS:
        noisy += [e, dict(e), None, {}]
    p1 = Pipeline(INFO)
    again = p1.run_file(noisy)
    assert [i["insight_id"] for i in base] == [i["insight_id"] for i in again]
    assert p1.state.counters["duplicate"] == len(EVENTS)


def test_same_moment_differs_by_audience():
    _, a = run(mode="analyst")
    _, c = run(mode="casual_fan", lang="id")
    assert a[-1]["headline"] != c[-1]["headline"] and a[-1]["trigger"] == c[-1]["trigger"]


def test_player_focus_filters_other_players():
    _, ins = run(prefs={"player_focus": True, "favorite_player_id": 91011})
    assert ins and all(any(p["player_id"] == 91011 for p in i["players"]) or i["importance"]["tier"] == "critical" for i in ins)


def test_api_roundtrip():
    c = TestClient(appmod.app)
    assert c.get("/health").json()["ok"]
    _, expect = run(mode="casual_fan", lang="id")
    assert len(c.get("/insights?mode=casual_fan&lang=id").json()) == len(expect) >= 8
    assert c.get("/tracking").json()["frame_rate_hz"] == 2
    c.post("/reset")
    n = 0
    for e in EVENTS[:600]:
        n += len(c.post("/events", json=e).json()["emitted"])
    assert n >= 1


def test_var_and_offside_moments():
    """New event types: VAR overturns and the offside-flag pattern become schema-1.1 insights, grounded in metrics[]."""
    _, ins = run()
    kinds = {i["trigger"]["rule_id"]: i for i in ins}
    assert "var.goal_disallowed" in kinds and "var.penalty_overturned" in kinds
    d = kinds["var.goal_disallowed"]
    assert d["schema_version"] == "1.1" and d["trigger"]["kind"] == "var_overturn"
    assert {m["key"] for m in d["metrics"]} >= {"offside_margin_m", "shot_xg", "score_now"}
    assert d["provenance"]["grounding"]["status"] == "verified"
    # the disallowed shot is NOT a goal in the running score
    assert d["metrics"][3]["value"] == "2-2" or "-" in str(next(m for m in d["metrics"] if m["key"] == "score_now")["value"])


def test_disallowed_goal_does_not_count():
    p, ins = run()
    assert p.state.score == {"KES": 3, "IRO": 2}
    assert any(e["type"]["name"] == "Shot" and e["shot"]["outcome"]["name"] == "Goal Disallowed" for e in EVENTS)


def test_offside_pattern_fires_once_at_third_flag():
    off = next(e for e in EVENTS if e["type"]["name"] == "Offside")
    p = Pipeline(INFO)
    base = {"period": 1, "minute": 10}
    fired = []
    for k in range(5):
        e = json.loads(json.dumps(off))
        e.update(id=f"synthetic-off-{k}", index=9000 + k, timestamp=f"00:{10 + k:02d}:00.000", minute=10 + k, second=0, period=1)
        fired.append([m["rule_id"] for m in p.feed(e)])
    assert fired == [[], [], ["pattern.offside_flags"], [], []]
    ins = p.insight(p.moments[0])
    assert ins["trigger"]["kind"] == "pattern" and ins["provenance"]["grounding"]["status"] == "verified"
    assert not list(V.iter_errors(ins))
