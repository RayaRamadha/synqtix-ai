"""Agent layer tests. A scripted fake chat client stands in for Foundry, so these run offline and deterministically."""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

pytest.importorskip("agent_framework")
from agent_framework import BaseChatClient, ChatResponse, Message  # noqa: E402

import app as appmod  # noqa: E402
from engine import Pipeline  # noqa: E402
from engine.agents import AgentTeam, match_entities  # noqa: E402
from engine.verify import verify_llm_text  # noqa: E402

B = os.path.join(os.path.dirname(__file__), "..")
INFO = json.load(open(f"{B}/data/match_info.json"))
EVENTS = json.load(open(f"{B}/data/synthetic_match_900001.json"))
V = Draft202012Validator(json.load(open(f"{B}/insight.schema.json")))


class FakeClient(BaseChatClient):
    """script(agent, payload, n) -> reply text, or raises. `agent` is Explain / Persona / Recap."""

    def __init__(self, script):
        super().__init__()
        self.script, self.calls = script, []

    def _inner_get_response(self, *, messages, stream, options, **kw):
        sysmsg = str(options.get("instructions") or "")   # Agent Framework passes the agent's instructions via options
        agent = next(a for a in ("Explain", "Persona", "Recap") if f"You are {a}Agent" in sysmsg)
        payload = json.loads(messages[-1].text)
        self.calls.append((agent, payload))

        async def go():
            text = self.script(agent, payload, sum(1 for a, _ in self.calls if a == agent))
            return ChatResponse(messages=[Message(role="assistant", contents=[text])])
        return go()


def good(agent, p, n):
    if agent == "Explain":
        return json.dumps({"analysis": f"A pivotal moment for {p['team']} at {p['clock']}."})
    if agent == "Persona":
        return json.dumps({"headline": p["template_headline"][:85], "why_it_matters": f"{p['analysis']} The match turns here."})
    return json.dumps({"title": "A match that turned", "recap": "Kestrel came from behind to win the match with a late goal."})


@pytest.fixture(scope="module")
def pipe():
    p = Pipeline(INFO)
    for e in EVENTS:
        p.feed(e)
    return p


def pairs_for(p, mode="analyst", lang="en"):
    return [(m, i) for m in p.moments if (i := p.insight(m, mode=mode, lang=lang))]


def enrich(client, p, **kw):
    team = AgentTeam(client, "fake:model")
    return team, asyncio.run(team.enrich_many(pairs_for(p, **kw), p.names, match_entities(p), {}))


def test_llm_path_is_verified_attributed_and_schema_valid(pipe):
    client = FakeClient(good)
    _, out = enrich(client, pipe)
    assert len(out) == 10
    for i in out:
        assert not list(V.iter_errors(i)), i["headline"]
        pv = i["provenance"]
        assert pv["model"] == "fake:model" and pv["degraded"] is False and pv["grounding"]["status"] == "verified"
        assert pv["produced_by"] == ["InterpretAgent", "ExplainAgent", "PersonaAgent", "VerifierAgent"]
        assert {t["stage"] for t in pv["trace"]} >= {"interpret", "explain", "personalize", "verify"}
        assert "The match turns here." in i["why_it_matters"]
        assert i["overlay"]["payload"]["subtitle"] == i["why_it_matters"][:120]
    assert {a for a, _ in client.calls} == {"Explain", "Persona"}


def test_explain_agent_only_sees_grounded_material(pipe):
    client = FakeClient(good)
    enrich(client, pipe)
    ex = next(p for a, p in client.calls if a == "Explain")
    assert set(ex) >= {"factors", "metrics", "allowed_numbers", "allowed_names", "clock"}
    pe = next(p for a, p in client.calls if a == "Persona")
    assert pe["audience"]["language"] == "English" and pe["previous_problems"] == []


def test_verifier_feedback_loop_recovers(pipe):
    def script(agent, p, n):
        if agent == "Persona" and not p["previous_problems"]:
            return json.dumps({"headline": "Big moment", "why_it_matters": "A 47 percent swing, according to Zorbak."})
        return good(agent, p, n)
    _, out = enrich(FakeClient(script), pipe)
    for i in out:
        assert i["provenance"]["degraded"] is False
        assert [t["stage"] for t in i["provenance"]["trace"]].count("personalize") == 2   # retried once, then verified
        assert "47" not in i["why_it_matters"]


def test_hallucination_that_never_recovers_falls_back_to_template(pipe):
    bad = lambda a, p, n: good(a, p, n) if a != "Persona" else json.dumps({"headline": "Seven goals!", "why_it_matters": "He scored 7 in a row."})
    _, out = enrich(FakeClient(bad), pipe)
    _, template = None, [i for _, i in pairs_for(pipe)]
    for got, tmpl in zip(out, template):
        assert got["headline"] == tmpl["headline"] and got["why_it_matters"] == tmpl["why_it_matters"]
        assert got["provenance"]["degraded"] is True and got["provenance"]["degraded_reason"].startswith("llm_fallback")
        assert not list(V.iter_errors(got))


def test_model_outage_never_breaks_the_stream(pipe):
    def boom(a, p, n):
        raise RuntimeError("429 quota exceeded")
    _, out = enrich(FakeClient(boom), pipe)
    assert len(out) == 10 and all(i["provenance"]["degraded"] for i in out)
    assert all("RuntimeError" in i["provenance"]["degraded_reason"] or "ExplainAgent" in i["provenance"]["degraded_reason"] for i in out)


def test_garbage_output_falls_back(pipe):
    _, out = enrich(FakeClient(lambda a, p, n: "I am sorry, I cannot help."), pipe)
    assert all(i["provenance"]["degraded"] for i in out)


def test_budget_keeps_template_text_for_unfinished_items(pipe):
    class Slow(FakeClient):
        def _inner_get_response(self, **kw):
            inner = super()._inner_get_response(**kw)

            async def go():
                await asyncio.sleep(5)
                return await inner
            return go()
    team = AgentTeam(Slow(good), "fake:model")
    out = asyncio.run(team.enrich_many(pairs_for(pipe), pipe.names, match_entities(pipe), {}, budget_s=0.2))
    assert [i["headline"] for i in out] == [i["headline"] for _, i in pairs_for(pipe)]


def test_results_are_cached(pipe):
    client = FakeClient(good)
    team = AgentTeam(client, "fake:model")
    asyncio.run(team.enrich_many(pairs_for(pipe), pipe.names, match_entities(pipe), {}))
    n = len(client.calls)
    asyncio.run(team.enrich_many(pairs_for(pipe), pipe.names, match_entities(pipe), {}))
    assert len(client.calls) == n


def test_strict_verifier_rules(pipe):
    mo = next(m for m in pipe.moments if m["kind"] == "goal")
    ents = match_entities(pipe)
    ok = verify_llm_text("Goal for Kestrel", "A decisive moment.", mo["metrics"], ents)
    assert ok["status"] == "verified"
    assert verify_llm_text("Goal", "He scored 999 times.", mo["metrics"], ents)["status"] == "failed"
    assert any("seven" in x for x in verify_llm_text("Goal", "Seven chances in a row.", mo["metrics"], ents)["unbacked_numbers"])
    assert any("Zorbak" in p for p in verify_llm_text("Goal", "Then Zorbak appeared.", mo["metrics"], ents)["problems"])
    assert verify_llm_text("x" * 91, "fine", mo["metrics"], ents)["status"] == "failed"
    # a number that IS backed passes, in either decimal style
    xg = next(m["value"] for m in mo["metrics"] if m["key"] == "shot_xg")
    assert verify_llm_text("Goal", f"A {xg} xG chance.".replace(".", ","), mo["metrics"], ents)["status"] == "verified"


def test_recap_verified_and_fallback(pipe):
    meta, items = pipe.meta(), [i for _, i in pairs_for(pipe)]
    team = AgentTeam(FakeClient(good), "fake:model")
    r = asyncio.run(team.recap(items, meta, "casual_fan", "en", match_entities(pipe)))
    assert r["source"] == "llm" and r["grounding"]["status"] == "verified"
    bad = AgentTeam(FakeClient(lambda a, p, n: json.dumps({"title": "t", "recap": "Won 9-9 after 12 goals."})), "fake:model")
    r = asyncio.run(bad.recap(items, meta, "casual_fan", "id", match_entities(pipe)))
    assert r["source"] == "template" and r["degraded_reason"].startswith("llm_fallback") and "3-2" in r["recap"]


def test_api_serves_agent_text_and_exposes_topology(pipe, monkeypatch):
    monkeypatch.setattr(appmod, "TEAM", AgentTeam(FakeClient(good), "fake:model"))
    monkeypatch.setattr(appmod, "match_entities", match_entities)
    c = TestClient(appmod.app)
    h = c.get("/health").json()
    assert h["llm"] == {"enabled": True, "model": "fake:model", "status": appmod.LLM_STATUS}
    items = c.get("/insights?mode=casual_fan&lang=id").json()
    assert len(items) >= 8 and all(i["provenance"]["model"] == "fake:model" for i in items)
    assert all(i["provenance"]["model"] == "deterministic-templates-v0" for i in c.get("/insights?llm=false").json())
    assert c.get("/recap?lang=en").json()["source"] == "llm"
    topo = c.get("/agents").json()
    assert [a["name"] for a in topo["agents"]][:4] == ["InterpretAgent", "ExplainAgent", "PersonaAgent", "VerifierAgent"]
