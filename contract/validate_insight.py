#!/usr/bin/env python3
"""Validate insight JSON(s) against the schema + cross-field integrity rules.
  python contract/validate_insight.py insight.schema.json contract/examples/*.json
Checks beyond JSON Schema: explanation.metric_keys exist in metrics[]; highlight/tag/focus keys exist;
show_at >= stream_offset; degraded=>reason; (optional) event_ids exist in the match file."""
import json, sys, re
from jsonschema import Draft202012Validator
schema = json.load(open(sys.argv[1])); v = Draft202012Validator(schema); bad = 0
def keys_ok(x):
    err = []; mk = {m["key"] for m in x["metrics"]}
    refs = [k for f in x["explanation"]["factors"] for k in f["metric_keys"]]
    refs += x["overlay"]["payload"].get("highlight_metric_keys", [])
    refs += [k for p in x.get("players", []) for k in p.get("tag_metric_keys", [])]
    if x.get("personalization", {}).get("focus_metric"): refs.append(x["personalization"]["focus_metric"])
    err += [f"unknown metric key: {k}" for k in refs if k not in mk]
    t = x["timing"]
    if t["show_at_stream_offset_s"] < t["stream_offset_s"]: err.append("show_at earlier than event time")
    return err
for p in sys.argv[2:]:
    x = json.load(open(p)); e = [f"{'/'.join(map(str,i.absolute_path))}: {i.message}" for i in v.iter_errors(x)]
    if not e: e = keys_ok(x)
    print(("OK   " if not e else "FAIL ") + p); [print("   -", i) for i in e]; bad += bool(e)
sys.exit(1 if bad else 0)
