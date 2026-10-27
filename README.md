# Match Insight Engine (deterministic core)

Synthetic football events -> explainable, personalized, overlay-ready insights. **Synthetic data only.**

```
replay.py / POST /events -> ingest (state.py) -> interpret+explain (detect.py) -> render+personalize (render.py) -> verify (verify.py) -> insight JSON (insight.schema.json)
```

| Stage | Module | Notes |
|---|---|---|
| Ingest | `engine/state.py` | dedup by event id, drops empty/invalid events, monotonic `stream_offset_s`, running score/xG/passes/red+yellow cards, offsides, VAR log |
| Interpret + Explain | `engine/detect.py` | language-neutral *Moments*: metrics with provenance + explanation factors (control vs chaos, pressure shift, transition vulnerability, game state, milestone, finishing, **VAR overturns**, **offside-flag pattern**) |
| Render + Personalize | `engine/render.py` | analyst / casual_fan, en / id, favorite team/player, player-focus, focus metric |
| Verify | `engine/verify.py` | every number in text must be backed by `metrics[]` (else `degraded`) |
| Explain + Persona (AI) | `engine/agents.py` | **Microsoft Agent Framework** workflow: ExplainAgent -> PersonaAgent -> VerifierAgent (retry loop), RecapAgent; model via **Microsoft Foundry** |

## Run
```bash
python -m sim.generate --decisions                 # (re)generate the synthetic match, tracking, match_info (+ decision log)
python export_tracks.py && python export_snapshots.py   # rebuild static/tracks.js + static/snapshots.js
pip install -r requirements.txt
python eval_answer_key.py                          # answer-key report -> reports/
python eval_answer_key.py --dup 0.1 --empty 0.05   # failure test
python -m pytest -q tests
CHROMIUM=/path/to/chromium python tests/browser_smoke.py   # headless-browser smoke test from file:// (Playwright)
uvicorn app:app --reload                           # /insights  /stream (SSE)  /events  /reset  /docs
```

## Design decisions
- Thresholds are conventional (big chance = xG >= 0.30; 10-min control window), not fitted to the answer key. Extra candidates are reported separately for manual review.
- Wording has two layers. The deterministic templates (`engine/render.py`) are the source of truth and the fallback. When a model is configured, the agent team (below) rewrites `headline` and `why_it_matters` **only from the same factors/metrics**, and the text is shown only if it passes the strict verifier; otherwise the template text is emitted and `provenance.degraded` explains why.
- The keeper for a save comes from the starting lineup, because the keeper's own event arrives after the shot in a live stream.

## Known limits
- The template path's verifier checks digits only. The strict verifier used for model text (`verify_llm_text`) also rejects unbacked spelled-out numbers ("seven", "tujuh"), names that are not on the roster, and over-long text. "one"/"satu" are not checked (too ambiguous as ordinary words).
- Pass share is computed from pass counts, not possession time, so it differs from `match_info.stats.possession_*`.
- Tracking is SIMULATED (see below): the engine itself still consumes events only; per-frame positions are used by the frontend and by `export_tracks.py`.
- Schema 1.1 (additive): `trigger.kind` `var_overturn`, factor types `var_review` / `offside_pattern`; 1.0 documents stay valid. Disallowed goals (shot outcome `Goal Disallowed`) never change the score.

## Frontend (static/index.html)
Single file, no build step. Stage with scorebug and pitch, four overlay slots (top banner, right pitch card, bottom lower-third, left player tag),
transport with clickable moment markers, mode / language / favourite club / player focus / focus metric, and a feed whose items expand into
"why it matters" + evidence (every metric with its source) + grounding badge.

```bash
uvicorn app:app --reload          # open http://127.0.0.1:8000/   (live API; badge shows "live API")
python export_snapshots.py        # re-generate static/snapshots.js after changing the engine
# no backend needed: open static/index.html directly or host static/ anywhere; badge shows "snapshot"
# point the page at another backend:  index.html?api=https://your-api.example.com
```
The page tries the API first and falls back to `snapshots.js` (pre-rendered for 3 modes x 2 languages), so the demo and judge access never depend on LLM quota.

## Decision model & simulator (`sim/`)
The old generator was the root cause of the unnatural play, so the match is now produced by a small, seeded, explainable **decision model** driving a 2 Hz **match simulator**.
`python -m sim.generate` writes `data/synthetic_match_900001.json` (events), `data/tracking_900001.json` (SIMULATED positions of 22 players + ball), `data/match_info.json` (stats, answer key, VAR/cards/subs, acceptance numbers) and, with `--decisions`, `data/decisions_900001.json` (every decision with its top-3 alternatives).

| File | Role |
|---|---|
| `sim/model.py` | pure maths: xG(distance, angle, body part, pressure, first-time, lane blockers), pass completion (distance, height, pressure on passer/receiver, lane block, skill), positional threat, **offside law** |
| `sim/decision.py` | utility model for the player in possession: shoot / pass (each team-mate) / carry / dribble / clear / hold, argmax(value + small seeded Gumbel noise), top-3 logged |
| `sim/world.py` | 0.5 s simulator: formations + compact lines + presser/marking, runs timed on the offside line, restarts, fouls/cards, VAR, substitutions, tracking frames, events |
| `sim/beats.py` | scripted beats (answer-key moments) |
| `sim/metrics.py`, `sim/probe.py` | acceptance metrics from a match / from randomised close-range situations through the same `decide()` |

**Design decisions**
- *Shoot vs pass.* Every option has a value in xG-like units. `shoot = xG x role - lost-possession cost`; `pass = p_complete x (CF x threat(receiver) + progression + keep) - (1-p) x loss`. `CF` (0.42) is the share of positional threat that becomes a shot after the next action, so shooting yourself beats passing from the same spot and a team-mate only wins if clearly better placed (cut-back). Shots below xG 0.05 happen only as speculative efforts after long possession.
- *Offside is a hard constraint.* A receiver is offside when, at the instant the ball is played, he is in the opponent half and nearer the goal line than both the ball and the second-last defender (keeper counts); throw-ins, goal kicks and corners are exempt. The passer perceives the margin with Gaussian noise (sigma 0.8 m) and drops receivers who look offside; real offsides come from perception errors and from the back line stepping up as the pass is played (the step-up is applied to the tracked positions, so the frame at the pass shows it). Result: `Pass Offside` + `Offside` + `Free Kick Awarded`, marginal ones flagged `x_var_candidate`.
- *Game state.* Ironvale: low block (line 28 m), direct passing, counter mode for 14 s after winning the ball in transition (better completion, easier box entries, more shots). Kestrel: possession-heavy, patience grows with time in the final third (more speculative shots), higher line and pressing in H2, much higher after the red card.
- *Scripted beats* (`sim/beats.py`) only fix **when** the answer-key moments happen: at `arm` seconds the beat team wins the ball through a plain bad pass the nearest opponent reads; the model builds the move; after `build` seconds the possession is protected (no tackles / failed passes) so the moment resolves in its minute; the designated shooter only shoots once the model's own xG is >= `min_xg`; the outcome (goal / post / save / disallowed) is fixed. Passes, carries and off-ball movement in between are model output. **Unscripted shots never score** (they resolve as saved / blocked / off target), so the scoreline stays 3-2; this is a deliberate simplification.
- *Extra decisions (not in the answer key):* 33' penalty given on the pitch, **overturned** by VAR (contact 1-2 m outside the area -> free kick); 66' Valcourt goal **disallowed** for offside (margin from the pass-time frame, < 1 m), recorded as a shot with outcome `Goal Disallowed`.
- *New event types / fields* (documented in `match_info.x_extensions`): `Offside`, `Free Kick Awarded`, `VAR` (review, outcome, checks, duration, subject, freeze-frame), yellow cards, `Foul Won`, restart `pass.type` (Kick Off / Throw-in / Goal Kick / Corner / Free Kick), `pass.x_offside_check`, shot `Goal Disallowed`.
- *Tracking file.* `frame_layout` is documented in the file; 2 Hz, global frame (Kestrel attack +x), every pass/shot is played on the frame grid so the freeze-frame of a pass is the frame at its timestamp. Back-line spread, speed limits, no teleporting and offside margins re-derived from frames are covered by tests.
- Pure rule/utility model, no external data: **no licensing question**, nothing is fitted to open football data (the xG curve is hand-set to plausible reference points, not calibrated).

**Everything produced here is simulated.** Positions, VAR checks/durations, ball speed and pressure are model output; labels in the UI say so.

## Full-match view (FM-style 2D)
The stage is a canvas that replays the **whole match** from `static/tracks.js` (`python export_tracks.py`): events plus delta-encoded **simulated tracking frames**.
22 players with shirt numbers and the ball come from the tracking frames (linear interpolation between 2 Hz frames); recent pass lines, shot lines with xG, pressure rings, substitutions and the red card (Ironvale play with 10).
If `tracks.js` has no tracking frames the old fallback is used (event locations + formation-slot drift). Side panel: **Commentary** (EN + ID, click to jump), **Key moments**, **Stats** (now with offsides).
Speeds 1x-60x, **Highlights** fast-forward, next-moment button. Works from `file://` (snapshot mode, no failed requests) or with `uvicorn app:app` (`/tracking` serves the tracking file).

### Offside line, VAR and officials
- The dashed **offside line** is computed from the tracked positions (2nd-last defender incl. keeper, never past halfway/ball); attackers beyond it get a red dot.
- **Offside calls** and **VAR panels are driven by the real events**: the offside panel freezes the frame of the pass and shows the margin from the data; the VAR panel lists the real checks, the real outcome (goal confirmed / disallowed, red card upheld, penalty confirmed / overturned), the margin, and the freeze-frame + line at the key pass computed from tracking. The VAR clock is skipped (dead time) after the verdict.
- Honest limits: VAR checks, durations and verdicts are generated by the simulator, not real decisions; assistant referees and the referee are presentation only.

## Agent layer (Microsoft Agent Framework + Microsoft Foundry)
```
Moment -> ExplainAgent (LLM) -> PersonaAgent (LLM) -> VerifierAgent (deterministic) --ok--> insight
                                      ^                        |
                                      +---- concrete feedback -+ (max INSIGHT_LLM_ATTEMPTS) --exhausted--> template text + provenance.degraded
```
Built as an Agent Framework `Workflow` (`WorkflowBuilder`, `Executor`, conditional edge back to PersonaAgent); each LLM step is an `Agent` on a `FoundryChatClient` (or `OpenAIChatCompletionClient` for an Azure OpenAI resource).

| Agent | Kind | Role |
|---|---|---|
| InterpretAgent | deterministic | events -> moments with metrics and provenance (`engine/detect.py`) |
| ExplainAgent | LLM | *why* the moment matters, in neutral English, from factors + metrics only |
| PersonaAgent | LLM | voice for analyst / casual fan / broadcaster, English or Indonesian, favourite club and player |
| VerifierAgent | deterministic | numbers (digits and spelled out), names, length; failures go back to PersonaAgent as concrete feedback |
| RecapAgent | LLM | full-time recap from the verified insights, through the same verifier |

Failure behaviour is part of the design: a model error, timeout, quota limit, malformed JSON or a hallucinated number never reaches the screen. The template text is served instead and `provenance.degraded_reason` says what happened. Results are cached per insight (`data/llm_cache.json`), requests wait at most `INSIGHT_LLM_BUDGET_S`, and the cache is warmed in the background at start-up.

```bash
cp .env.example .env     # FOUNDRY_PROJECT_ENDPOINT + FOUNDRY_MODEL (az login locally; managed identity in Azure)
uvicorn app:app          # /health shows llm.enabled, /agents shows the topology, /recap the full-time recap
python export_snapshots.py --llm    # pre-render model-written, verified text into static/snapshots.js (+ data/llm_cache.json)
```

## Deploy (Azure Container Apps)
```bash
./infra/deploy.sh        # builds the image in Azure, creates the app, assigns a managed identity and the Foundry roles (no API keys)
```
No credentials, no problem: the same image serves the template text, and `static/snapshots.js` keeps the page working with no backend at all.

## Testing the agent layer
`tests/test_agents.py` runs the real Agent Framework workflow against a scripted chat client (success, retry-after-feedback, hallucination -> fallback, outage, garbage output, time budget, cache, recap, API). The Azure OpenAI client path was also exercised end-to-end against a local OpenAI-compatible stub. Live Foundry calls need your own project; none are made by the test suite.
