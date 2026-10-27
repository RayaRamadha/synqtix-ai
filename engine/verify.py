"""VerifierAgent (deterministic): every number in headline/why_it_matters must be backed by metrics[]."""
import re

NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str):
    text = re.sub(r"\d+(?:\+\d+)?'", " ", text)       # clock tokens: 90+3', 55'
    return [x.replace(",", ".") for x in NUM.findall(text)]


def _allowed(metrics):
    ok = set()
    for m in metrics:
        w = m.get("window")
        if w:                                   # a window's length (minutes) is backed by its own bounds
            ok.add(f"{(w['to_stream_s'] - w['from_stream_s']) / 60:g}")
        v = m["value"]
        if isinstance(v, (int, float)):
            ok.add(f"{float(v):g}")
        else:
            ok.update(f"{float(x):g}" for x in NUM.findall(str(v).replace(",", ".")))
    return ok


def verify(insight: dict) -> dict:
    allowed = _allowed(insight["metrics"])
    nums = _numbers(insight["headline"]) + _numbers(insight["why_it_matters"])
    bad = [n for n in nums if f"{float(n):g}" not in allowed]
    keys = {m["key"] for m in insight["metrics"]}
    missing = [k for f in insight["explanation"]["factors"] for k in f["metric_keys"] if k not in keys]
    status = "verified" if not bad and not missing else "failed"
    return {"status": status, "numbers_checked": len(nums), "unbacked_numbers": bad + [f"missing:{k}" for k in missing]}


# ---------------------------------------------------------------------------------------------
# Strict mode for LLM-written text (headline + why_it_matters).
# On top of the digit check above it also rejects: spelled-out numbers that no metric backs,
# capitalised names that are not on the match's roster/team list, and over-long text.
# Template text never goes through this path, so its wording ("three different ways") is unaffected.
# ---------------------------------------------------------------------------------------------
SPELLED = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "twice": 2, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6, "tujuh": 7, "delapan": 8,
           "sembilan": 9, "sepuluh": 10}  # "one"/"satu" are excluded on purpose: too ambiguous as ordinary words
CAP = re.compile(r"(?<![\w'])([A-ZÀ-Þ][a-zà-ÿ]{2,})")
WORD = re.compile(r"[A-Za-zÀ-ÿ]+")
LIMITS = {"headline": 90, "why_it_matters": 320}


def _unknown_entities(text: str, entities: set[str]) -> list[str]:
    bad = []
    for m in CAP.finditer(text):
        before = text[:m.start()].rstrip()
        if not before or before[-1] in ".!?:;—-–(\"“":      # sentence start: capitalised for grammar, not a name
            continue
        if m.group(1).lower() not in entities:
            bad.append(m.group(1))
    return bad


def verify_llm_text(headline: str, why: str, metrics: list, entities: set[str] | None = None, limits: dict | None = None) -> dict:
    """Same contract as verify() plus `problems` (human-readable, used as retry feedback)."""
    allowed = _allowed(metrics)
    lim = limits or LIMITS
    problems, bad = [], []
    for field, text in (("headline", headline), ("why_it_matters", why)):
        if not isinstance(text, str) or not text.strip():
            problems.append(f"{field} is empty")
            continue
        if len(text) > lim[field]:
            problems.append(f"{field} is {len(text)} characters; the limit is {lim[field]}")
        for n in _numbers(text):
            if f"{float(n):g}" not in allowed:
                bad.append(n)
        for w in WORD.findall(text):
            v = SPELLED.get(w.lower())
            if v is not None and f"{float(v):g}" not in allowed:
                bad.append(w.lower())
        if entities is not None:
            for name in _unknown_entities(text, {e.lower() for e in entities}):
                problems.append(f"unknown name '{name}' in {field}")
    if bad:
        problems.append("numbers not backed by the metrics: " + ", ".join(dict.fromkeys(bad)) +
                        "; allowed numbers are: " + ", ".join(sorted(allowed, key=float)))
    nums = _numbers(headline or "") + _numbers(why or "")
    return {"status": "verified" if not problems else "failed", "numbers_checked": len(nums),
            "unbacked_numbers": list(dict.fromkeys(bad)), "problems": problems}
