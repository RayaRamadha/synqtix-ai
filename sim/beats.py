"""Scripted beats: the *moments* (answer key) are scheduled, the *play around them* comes from the decision model.

A beat only does five things: (1) at `arm` seconds (period-local) it makes `team` win the ball (a plain bad pass by the other team that
the nearest opponent reads - never a teleport), (2) it lets the decision model build the move for `build` seconds, with a bonus for finding
the designated shooter, (3) after that the possession is "protected" (no tackles / failed passes) so the moment resolves inside its minute,
(4) the designated shooter only shoots once the model's own xG for his position is >= `min_xg` (so a "big chance" really is one),
(5) the outcome (goal / post / save ...) is fixed by the beat. Every pass, carry and off-ball movement in between is model output.
`min` / `max` = the minute window (period-local seconds) the key event must land in; checked by the generator.
"""
BEATS = [
    dict(key="counter1", kind="counter", per=1, arm=712, build=9, team="IRO", shooter=92010, outcome="Goal", min_xg=0.12, window=(720, 780)),
    dict(key="foul33", kind="box_edge_foul", per=1, arm=1940, build=14, team="KES", shooter=91009, victim=91009, fouler=92003, window=(1950, 2040)),
    dict(key="post", kind="chance", per=1, arm=1492, build=16, team="KES", shooter=91011, outcome="Post", min_xg=0.32, window=(1500, 1560)),
    dict(key="counter2", kind="counter", per=1, arm=2276, build=9, team="IRO", shooter=92011, outcome="Goal", min_xg=0.12, window=(2280, 2340)),
    dict(key="red", kind="foul_red", per=2, arm=588, build=10, team="KES", shooter=91011, fouler=92004, window=(600, 660)),
    dict(key="goal61", kind="chance", per=2, arm=950, build=16, team="KES", shooter=91011, outcome="Goal", min_xg=0.2, window=(960, 1020)),
    dict(key="disallowed", kind="disallowed", per=2, arm=1252, build=12, team="KES", shooter=91012, outcome="Goal Disallowed", min_xg=0.22, window=(1260, 1320)),
    dict(key="penalty", kind="foul_pen", per=2, arm=1684, build=12, team="KES", shooter=91011, fouler=92003, window=(1700, 1800)),
    dict(key="breakaway", kind="chance", per=2, arm=2262, build=8, team="IRO", shooter=92010, outcome="Saved", min_xg=0.32, window=(2280, 2340)),
    dict(key="winner", kind="corner_goal", per=2, arm=2876, build=0, team="KES", shooter=91011, outcome="Goal", window=(2880, 2940)),
]
SUBS = [dict(per=2, t=0, team="KES", out=91008, inn=91012), dict(per=2, t=900, team="IRO", out=92006, inn=92012),
        dict(per=2, t=1500, team="KES", out=91007, inn=91013), dict(per=2, t=1860, team="IRO", out=92008, inn=92013)]
