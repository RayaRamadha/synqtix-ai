#!/usr/bin/env python3
"""Headless-browser smoke test (Playwright). Loads static/index.html from file://, plays across a goal (with its real VAR check), an offside call,
the disallowed goal and the red card, asserts: no console errors / page errors, 22 tracked players (21 after the red card... 10 for Ironvale),
real VAR panels shown, canvas not blank. Screenshots -> reports/screens/.
  CHROMIUM=/path/to/chromium python tests/browser_smoke.py
Without CHROMIUM it uses Playwright's own browser (python -m playwright install chromium)."""
import os, sys, json, time, pathlib
from playwright.sync_api import sync_playwright

B = pathlib.Path(__file__).resolve().parent.parent
OUT = B / "reports" / "screens"
OUT.mkdir(parents=True, exist_ok=True)
URL = (B / "static" / "index.html").as_uri()
errors, seen = [], {}


def blank(png):
    from PIL import Image
    im = Image.open(png).convert("RGB").resize((160, 90))
    px = list(im.get_flattened_data()) if hasattr(im, "get_flattened_data") else list(im.getdata())
    return len(set(px)) < 12


with sync_playwright() as pw:
    kw = {"headless": True, "args": ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]}
    if os.environ.get("CHROMIUM"):
        kw["executable_path"] = os.environ["CHROMIUM"]
    br = pw.chromium.launch(**kw)
    page = br.new_page(viewport={"width": 1440, "height": 900})
    page.on("console", lambda m: errors.append(("console." + m.type, m.text)) if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: errors.append(("pageerror", str(e))))
    page.on("requestfailed", lambda r: errors.append(("requestfailed", r.url)) if not r.url.startswith("data:") else None)
    page.goto(URL)
    try:
        page.wait_for_function("typeof FM !== 'undefined' && FM.ready === true", timeout=30000)
    except Exception:
        print("page errors so far:", errors)
        raise
    info = page.evaluate("""() => ({trk: !!FM.trk, realVar: FM.realVar, frames: FM.trk && FM.trk.n, src: S.src,
        goals: TR.events.filter(e => e.k === 'Shot' && e.o === 'Goal').map(e => e.s),
        dis: TR.events.filter(e => e.k === 'Shot' && e.o === 'Goal Disallowed').map(e => e.s),
        red: TR.events.filter(e => e.k === 'Foul Committed' && e.o === 'Red Card').map(e => e.s),
        off: TR.events.filter(e => e.k === 'Offside' && !e.vo).map(e => e.s), vars: TR.events.filter(e => e.k === 'VAR').length})""")
    print("loaded:", json.dumps(info)[:300])
    assert info["trk"] and info["realVar"] and info["frames"] > 10000, info
    assert info["src"] == "snapshot", "file:// should run from snapshots.js"

    def players_at(t):
        return page.evaluate("t => { const p = trkPlayers(t); return [p.filter(q => q.c === S.meta.order[0]).length, p.filter(q => q.c === S.meta.order[1]).length]; }", t)

    assert players_at(300) == [11, 11], players_at(300)
    assert players_at(info["red"][0] + 120) == [11, 10], players_at(info["red"][0] + 120)     # Ironvale down to ten

    def watch(t_start, until_fn, shot_prefix, speed=6, timeout=90):
        """jump to t_start, play, take a screenshot at each new hold kind/phase, until until_fn(S.t) or timeout"""
        page.evaluate("([t, sp]) => { S.speed = sp; S.hl = false; jump(t); setPlaying(true); }", [t_start, speed])
        t0 = time.time()
        local = set()
        while time.time() - t0 < timeout:
            st = page.evaluate("() => ({t: S.t, hold: FM.hold ? FM.hold.kind + ':' + holdPhase(FM.hold).n : null, var: $('var').innerText.slice(0, 220), playing: S.playing})")
            key = st["hold"]
            if key and key not in local:
                local.add(key)
                seen[shot_prefix + ":" + key] = st["var"]
                f = OUT / f"{shot_prefix}_{key.replace(':', '_')}.png"
                page.screenshot(path=str(f))
                assert not blank(f), f"blank screenshot {f}"
            if until_fn(st["t"]):
                break
            page.wait_for_timeout(150)
        page.evaluate("() => setPlaying(false)")

    g0 = info["goals"][0]
    watch(g0 - 5, lambda t: t > g0 + 130, "goal1", speed=6)
    assert any(":goal:" in k for k in seen), seen
    assert any(":var:" in k for k in seen) and "VAR" in " ".join(seen.values()) and "GOAL CONFIRMED" in " ".join(seen.values()).upper(), seen
    if info["off"]:
        watch(info["off"][0] - 8, lambda t: t > info["off"][0] + 6, "offside", speed=3)
        assert any(":off:" in k for k in seen), seen
    if info["dis"]:
        d0 = info["dis"][0]
        watch(d0 - 5, lambda t: t > d0 + 110, "disallowed", speed=6)
        assert "DISALLOWED" in " ".join(v for k, v in seen.items() if k.startswith("disallowed")).upper(), seen
    r0 = info["red"][0]
    watch(r0 - 8, lambda t: t > r0 + 90, "red", speed=6)
    assert "RED CARD" in " ".join(v for k, v in seen.items() if k.startswith("red")).upper(), seen
    page.evaluate("() => { S.lang = 'id'; }")
    page.screenshot(path=str(OUT / "final.png"))
    # canvas really painted (not just the pitch): players exist at the end
    final = page.evaluate("() => trkPlayers(S.t).length")
    assert final in (21, 22, 20), final
    br.close()

bad = [e for e in errors if "favicon" not in e[1]]
print("holds seen:", sorted(seen))
print("console/page errors:", bad)
assert not bad, bad
print("BROWSER SMOKE OK")
