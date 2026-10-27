"""Stage 4-5 (Render + Personalize): neutral Moment -> schema-valid insight for an audience/language.

Wording lives here (templates). It is the deterministic fallback for the LLM ExplainAgent: the
LLM can later rewrite `why_it_matters`, but only from the same factors/metrics, and must pass verify.py.
"""
import hashlib

LANGS = ("en", "id")
MODES = ("analyst", "casual_fan", "broadcaster")
PRIORITY = {"var_review": 0, "offside_pattern": 1, "transition_vulnerability": 0, "tactical_pressure_shift": 1, "milestone": 2, "game_state": 3,
            "finishing_efficiency": 4, "control_vs_chaos": 5}


def sur(name: str) -> str:
    return name.split()[-1]


def short(name: str) -> str:
    return name.split()[0]


# ---------- localized phrases ----------
GOAL_PHRASE = {
    "en": {"hat_trick": "completes the hat-trick", "penalty": "converts the penalty", "equalizer": "equalises",
           "go_ahead": "puts {team} ahead", "pull_back": "pulls one back", "opener": "opens the scoring",
           "extend": "extends the lead"},
    "id": {"hat_trick": "melengkapi hat-trick", "penalty": "mengeksekusi penalti", "equalizer": "menyamakan kedudukan",
           "go_ahead": "membawa {team} unggul", "pull_back": "memperkecil ketertinggalan", "opener": "membuka skor",
           "extend": "menambah keunggulan"},
}
TYPE_WORD = {"en": {"penalty": "penalty", "header": "header", "open play": "open play"},
             "id": {"penalty": "penalti", "header": "sundulan", "open play": "permainan terbuka"}}


def fnum(x, lang):
    s = f"{x}"
    return s.replace(".", ",") if lang == "id" else s


def factor_text(f: dict, lang: str, mode: str, names: dict) -> str | None:
    """Return one sentence for a factor, or None to skip it for this audience."""
    t, p = f["type"], f["params"]
    casual = mode == "casual_fan"
    n = lambda c: names[c]
    N = lambda x: fnum(x, lang)
    if t == "transition_vulnerability":
        if lang == "id":
            return (f"{n(p['opp'])} lebih banyak menguasai bola, tetapi justru kebobolan lewat serangan balik." if casual else
                    f"{n(p['opp'])} menguasai {N(p['share'])}% umpan dalam 10 menit terakhir, tetapi kebobolan lewat serangan balik.")
        return (f"{n(p['opp'])} had the ball but were caught out on the break." if casual else
                f"{n(p['opp'])} had {N(p['share'])}% of passes in the last 10 minutes but were caught on the counter.")
    if t == "tactical_pressure_shift":
        d = p["dir"]
        if casual and (d != "rose" or p["mins"] > 20):
            return None
        if lang == "id":
            if casual:
                return f"Kartu merah mulai mengubah arah pertandingan ke {n(p['team'])}."
            verb = {"rose": f"naik menjadi {N(p['since'])}% (sebelumnya {N(p['before'])}%)",
                    "held": f"bertahan di {N(p['since'])}% (sebelumnya {N(p['before'])}%)",
                    "fell": f"turun menjadi {N(p['since'])}% (sebelumnya {N(p['before'])}%)"}[d]
            return f"{p['mins']} menit setelah kartu merah, porsi umpan {n(p['team'])} {verb}, dengan {N(p['xg'])} xG tercipta sejak itu."
        if casual:
            return f"The red card is starting to tilt the game towards {n(p['team'])}."
        verb = {"rose": f"rose to {N(p['since'])}% (from {N(p['before'])}%)",
                "held": f"held at {N(p['since'])}% (was {N(p['before'])}%)",
                "fell": f"fell to {N(p['since'])}% (from {N(p['before'])}%)"}[d]
        return f"{p['mins']} minutes after the red card, {n(p['team'])}'s pass share {verb}, with {N(p['xg'])} xG created since."
    if t == "milestone":
        tw = ", ".join(TYPE_WORD[lang][x] for x in p["types"])
        if lang == "id":
            return (f"{sur(p['player'])} mencetak hat-trick: {p['goals']} gol hari ini!" if casual else
                    f"{sur(p['player'])} kini mengoleksi {p['goals']} gol dengan tiga cara berbeda: {tw}.")
        return (f"{sur(p['player'])} has a hat-trick: {p['goals']} goals today!" if casual else
                f"{sur(p['player'])} now has {p['goals']} goals, scored three different ways: {tw}.")
    if t == "game_state":
        s = p["situation"]
        team = n(p["team"]) if "team" in p else ""
        if s == "go_ahead":
            return (f"Dari tertinggal {p['deficit']} gol menjadi unggul: comeback lengkap!" if (lang == "id" and casual) else
                    f"Comeback lengkap: {team} sempat tertinggal {p['deficit']} gol dan kini memimpin." if lang == "id" else
                    f"Incredible comeback: from {p['deficit']} goals down to ahead!" if casual else
                    f"Comeback complete: {team} trailed by {p['deficit']} at one point and now lead.")
        if s == "equalizer":
            return (f"{team} kembali sejajar setelah sempat tertinggal {p['deficit']} gol." if lang == "id" else
                    f"{team} are level again after trailing by {p['deficit']}.")
        if s == "pull_back":
            return (f"{team} memangkas selisih menjadi {p['gap']} gol." if lang == "id" else
                    f"{team} cut the deficit to {p['gap']}.")
        if s == "extend":
            return (f"{team} kini unggul {p['gap']} gol." if lang == "id" else f"{team} now lead by {p['gap']}.")
        if s == "counterfactual":
            if lang == "id":
                rel = "unggul" if p["relation"] == "ahead" else "menyamakan kedudukan"
                return f"Jika gol, {team} akan {rel} ({p['score_if']}); skor saat ini {p['score_now']}."
            rel = "ahead" if p["relation"] == "ahead" else "level"
            return f"A goal here would have put {team} {rel} ({p['score_if']}); the score is {p['score_now']}."
        if s == "red":
            st = {"leading": ("memimpin", "leading"), "trailing": ("tertinggal", "trailing"), "level": ("imbang", "level")}[p["state"]]
            if lang == "id":
                return (f"{team} tinggal {p['n']} pemain." if casual else
                        f"{team} bermain dengan {p['n']} pemain sambil {st[0]} ({p['score']}).")
            return (f"{team} are down to {p['n']} men." if casual else
                    f"{team} are down to {p['n']} men while {st[1]} ({p['score']}).")
    if t == "var_review":
        tm = n(p["team"])
        if p["sub"] == "goal_disallowed":
            if lang == "id":
                return (f"VAR menganulir gol {tm}; skor tetap {p['score']}." if casual else
                        f"Gol {tm} dianulir VAR: pencetak gol offside {N(p['margin'])} m saat umpan dilepaskan, setelah pengecekan {p['secs']} detik. Skor tetap {p['score']}.")
            return (f"The goal for {tm} does not count; it stays {p['score']}." if casual else
                    f"{tm}'s goal was ruled out after a {p['secs']}-second VAR check: the scorer was {N(p['margin'])} m offside when the pass was played. The score stays {p['score']}.")
        if lang == "id":
            return (f"Penalti dibatalkan; skor tetap {p['score']}." if casual else
                    f"Pelanggaran terjadi {N(p['contact'])} m di luar kotak penalti; setelah pengecekan {p['secs']} detik, penalti dibatalkan menjadi tendangan bebas. Skor tetap {p['score']}.")
        return (f"No penalty after the check; it stays {p['score']}." if casual else
                f"The contact was {N(p['contact'])} m outside the area: after a {p['secs']}-second VAR check the penalty became a free kick. The score stays {p['score']}.")
    if t == "offside_pattern":
        if lang == "id":
            return (f"{n(p['team'])} berulang kali terjebak offside." if casual else
                    f"{n(p['team'])} sudah {p['n']} kali ditandai offside dalam 10 menit terakhir.")
        return (f"{n(p['team'])} keep getting caught offside." if casual else
                f"{n(p['team'])} have been flagged offside {p['n']} times in the last 10 minutes.")
    if t == "finishing_efficiency":
        if lang == "id":
            return (f"{n(p['team'])} memaksimalkan peluang yang sedikit." if casual else
                    f"{n(p['team'])} mencetak {p['goals']} gol dari {p['shots']} tembakan ({N(p['xg'])} xG): sangat klinis.")
        return (f"{n(p['team'])} are making their few chances count." if casual else
                f"{n(p['team'])} have scored {p['goals']} from {p['shots']} shots ({N(p['xg'])} xG): clinical.")
    if t == "control_vs_chaos":
        if p.get("context") == "red":
            if lang == "id":
                return None if casual else f"{n(p['team'])} sudah menguasai {N(p['share'])}% umpan dalam 10 menit terakhir."
            return None if casual else f"{n(p['team'])} already held {N(p['share'])}% of passes in the last 10 minutes."
        if lang == "id":
            return (f"{n(p['team'])} terus mendominasi bola tanpa hasil." if casual else
                    f"{n(p['team'])} menguasai {N(p['share'])}% umpan terbaru dan {N(p['xg'])} xG, tetapi baru {p['goals']} gol: menguasai bola belum tentu berbahaya.")
        return (f"{n(p['team'])} keep dominating the ball without scoring." if casual else
                f"{n(p['team'])} hold {N(p['share'])}% of recent passes and {N(p['xg'])} xG but have {p['goals']} goals: control is not the same as danger.")
    return None


def fallback_text(mo: dict, lang: str, mode: str, names: dict) -> str:
    m = {x["key"]: x["value"] for x in mo["metrics"]}
    f, N = mo["facts"], (lambda x: fnum(x, lang))
    tm = names[mo["team"]]
    if mode == "casual_fan" or "shot_xg" not in m:
        return headline(mo, lang, "casual_fan", names)
    if lang == "id":
        return f"Peluang dengan nilai {N(m['shot_xg'])} xG; {tm} sejauh ini mengumpulkan {N(m['team_xg_so_far'])} xG."
    return f"A {N(m['shot_xg'])} xG chance; {tm} have {N(m['team_xg_so_far'])} xG so far."


def headline(mo: dict, lang: str, mode: str, names: dict) -> str:
    f, k = mo["facts"], mo["kind"]
    casual = mode == "casual_fan"
    N = lambda x: fnum(x, lang)
    clk = "" if casual else f" ({mo['clock']})"
    if k == "goal":
        ph = GOAL_PHRASE[lang][f["sub"]].format(team=names[mo["team"]])
        base = f"{sur(f['shooter'])} {ph}"
        if lang == "id":
            return (f"GOL! {base}, skor {f['score']}" if casual else f"{base}: {f['score']}{clk}")
        return (f"GOAL! {base}, {f['score']}" if casual else f"{base}: {f['score']}{clk}")
    if k == "red_card":
        tm = names[f["team_down"]]
        if lang == "id":
            return (f"KARTU MERAH! {sur(f['player'])} keluar, {tm} tinggal {f['n']} pemain" if casual
                    else f"Kartu merah: {sur(f['player'])} diusir, {tm} bermain dengan {f['n']} pemain{clk}")
        return (f"RED CARD! {sur(f['player'])} is off, {tm} down to {f['n']}" if casual
                else f"Red card: {sur(f['player'])} sent off, {tm} down to {f['n']}{clk}")
    if k == "var_overturn":
        mm = {x["key"]: x["value"] for x in mo["metrics"]}
        if f["sub"] == "goal_disallowed":
            if lang == "id":
                return (f"GOL DIANULIR! VAR: {sur(f['shooter'])} offside" if casual else
                        f"VAR menganulir gol {sur(f['shooter'])} karena offside ({N(mm['offside_margin_m'])} m){clk}")
            return (f"NO GOAL! VAR rules {sur(f['shooter'])}'s goal out for offside" if casual else
                    f"VAR rules out {sur(f['shooter'])}'s goal for offside ({N(mm['offside_margin_m'])} m){clk}")
        if lang == "id":
            return "PENALTI DIBATALKAN oleh VAR" if casual else f"VAR membatalkan penalti: kontak {N(mm['contact_outside_box_m'])} m di luar kotak{clk}"
        return "NO PENALTY! VAR overturns it" if casual else f"VAR overturns the penalty: contact {N(mm['contact_outside_box_m'])} m outside the area{clk}"
    if k == "pattern":
        tm = names[f["team"]]
        if lang == "id":
            return f"{tm} terus terjebak offside" if casual else f"{tm}: {f['n']} offside dalam 10 menit{clk}"
        return f"{tm} caught offside again" if casual else f"{tm}: {f['n']} offside flags in 10 minutes{clk}"
    xg = next(m["value"] for m in mo["metrics"] if m["key"] == "shot_xg")
    if k == "big_save":
        kp = sur(f.get("keeper", "the keeper"))
        if lang == "id":
            return f"Penyelamatan besar {kp}!" if casual else f"{kp} menepis peluang {sur(f['shooter'])} ({N(xg)} xG){clk}"
        return f"What a save by {kp}!" if casual else f"{kp} denies {sur(f['shooter'])} ({N(xg)} xG chance){clk}"
    out = {"Post": ("mengenai tiang", "hits the post"), "Off T": ("melebar", "misses the target"),
           "Wayward": ("melenceng", "goes wide"), "Blocked": ("diblok", "is blocked")}.get(f["outcome"], ("gagal", "misses"))
    if lang == "id":
        return f"{sur(f['shooter'])} hampir mencetak gol!" if casual else f"Peluang besar: {sur(f['shooter'])} {out[0]} ({N(xg)} xG){clk}"
    return f"{sur(f['shooter'])} so close!" if casual else f"Big chance: {sur(f['shooter'])} {out[1]} ({N(xg)} xG){clk}"


def importance(mo: dict) -> dict:
    k, f = mo["kind"], mo["facts"]
    xg = next((m["value"] for m in mo["metrics"] if m["key"] == "shot_xg"), 0.5)
    if k == "goal":
        impact = {"hat_trick": 1.0, "go_ahead": 1.0, "equalizer": 0.9, "penalty": 0.9, "pull_back": 0.75,
                  "opener": 0.7, "extend": 0.7}[f["sub"]]
        if f["sub"] in ("penalty",) and f.get("situation") == "go_ahead":
            impact = 1.0
    elif k == "red_card":
        impact = 0.85
    elif k == "var_overturn":
        impact = 0.8
    elif k == "pattern":
        impact = 0.45
    elif k == "big_save":
        impact = 0.8 if f.get("relation") == "ahead" else 0.6
    else:
        impact = 0.5
    chance = min(1.0, xg / 0.5) if k not in ("red_card", "var_overturn", "pattern") else 0.5
    time = min(1.0, mo["stream_s"] / 5400)
    score = round(0.5 * impact + 0.2 * chance + 0.3 * time, 2)
    tier = "critical" if score >= 0.85 else "high" if score >= 0.7 else "medium" if score >= 0.5 else "low"
    return {"score": score, "tier": tier,
            "factors": [{"name": "impact", "weight": 0.5, "value": round(impact, 2)},
                        {"name": "chance_quality", "weight": 0.2, "value": round(chance, 2)},
                        {"name": "time_pressure", "weight": 0.3, "value": round(time, 2)}]}


def relevance(mo: dict, prefs: dict) -> float:
    involved = {p["player_id"] for p in mo["players"] if p["role"] in ("actor", "keeper")}
    fp = prefs.get("favorite_player_id")
    if prefs.get("player_focus") and fp:
        return 1.0 if fp in involved else 0.2
    r = 0.4
    if prefs.get("favorite_team") in (mo["team"], mo["facts"].get("team_down")):
        r += 0.3
    if fp and fp in involved:
        r += 0.3
    return round(min(r, 1.0), 2)


def render(mo: dict, names: dict, match_id: int, mode="analyst", lang="en", prefs: dict | None = None):
    """Returns an insight dict (without provenance trace/grounding) or None if filtered by personalization."""
    prefs = prefs or {}
    lang = lang if lang in LANGS else "en"
    mode = mode if mode in MODES else "analyst"
    imp = importance(mo)
    rel = relevance(mo, prefs)
    if prefs.get("player_focus") and prefs.get("favorite_player_id") and rel < 0.5 and imp["tier"] != "critical":
        return None
    factors = sorted(mo["factors"], key=lambda f: PRIORITY.get(f["type"], 9))
    limit = 3 if mode != "casual_fan" else 2
    used, sentences = [], []
    for f in factors:
        s = factor_text(f, lang, mode, names)
        if s:
            used.append(f)
            sentences.append(s)
        if len(sentences) >= limit:
            break
    if not sentences:  # always say something
        sentences = [fallback_text(mo, lang, mode, names)]
    why = " ".join(sentences)
    used_keys = {k for f in used for k in f["metric_keys"]}
    keep = [m for m in mo["metrics"] if m["key"] in used_keys or m["key"] in ("shot_xg", "shot_speed", "players_on_pitch", "score_after", "scorer_goals")]
    # headline numbers must be backed too: keep every metric (cheap, and gives analysts the full evidence)
    metrics = mo["metrics"]
    expl = [{"type": f["type"], "statement": s, "metric_keys": f["metric_keys"]} for f, s in zip(used, sentences)] or \
           [{"type": "game_state", "statement": why, "metric_keys": [k for k in ("shot_xg", "team_xg_so_far") if any(m["key"] == k for m in metrics)] or [metrics[0]["key"]]}]
    hl = headline(mo, lang, mode, names)
    focus = prefs.get("focus_metric")
    hi = ([focus] if focus and any(m["key"] == focus for m in metrics) else []) + \
         [k for k in ("shot_xg", "shot_speed", "offside_margin_m", "contact_outside_box_m", "offsides_10m", "score_after", "players_on_pitch") if any(m["key"] == k for m in metrics)]
    hi = list(dict.fromkeys(hi))[:3]
    analyst = mode != "casual_fan"
    otype = ("top_banner" if mo["kind"] in ("red_card", "var_overturn") else "pitch_map" if analyst and mo["annotations"] else "lower_third")
    insight_id = hashlib.sha1(f"{match_id}|{mo['event_id']}|{mo['rule_id']}|{mode}|{lang}".encode()).hexdigest()[:16]
    out = {
        "schema_version": "1.1", "insight_id": insight_id, "match_id": match_id, "synthetic": True,
        "trigger": {"kind": mo["kind"] if mo["kind"] != "big_chance" else "big_chance", "event_ids": [mo["event_id"]],
                    "event_index": mo["event_index"], "rule_id": mo["rule_id"]},
        "timing": {"period": mo["period"], "minute": mo["minute"], "second": mo["second"], "display_clock": mo["clock"],
                   "stream_offset_s": mo["stream_s"], "show_at_stream_offset_s": round(mo["stream_s"] + 2.0, 3),
                   "ttl_ms": 9000 if imp["tier"] in ("critical", "high") else 7000},
        "headline": hl, "why_it_matters": why, "explanation": {"factors": expl}, "metrics": metrics,
        "importance": imp, "audience_mode": mode, "language": lang,
        "players": mo["players"],
        "overlay": {"type": otype, "position": "top" if otype == "top_banner" else "right" if otype == "pitch_map" else "bottom",
                    "priority": {"critical": 9, "high": 7, "medium": 5, "low": 3}[imp["tier"]] + (1 if rel >= 0.7 else 0),
                    "payload": {"title": hl[:60], "subtitle": why[:120], "accent_team": mo["team"] if mo["kind"] != "red_card" else mo["facts"]["team_down"],
                                "highlight_metric_keys": hi,
                                **({"coord_space": "pitch_120x80_actor_attacks_x120", "annotations": mo["annotations"]} if otype == "pitch_map" else {})}},
    }
    pz = {k: v for k, v in prefs.items() if k in ("favorite_team", "favorite_player_id", "focus_metric", "player_focus")}
    if pz:
        out["personalization"] = {**pz, "relevance": rel}
    return out
