"""Pure football maths: geometry, xG, pass completion, threat, offside law. No randomness, no state.

Coordinates: 120 x 80 pitch. Functions ending in `_att` take the *attacking frame* of the team in
possession (own goal at x=0, opponent goal at (120, 40)). `att()` converts global -> attacking frame.
Global frame: KES attacks towards x=120, IRO towards x=0 (the frontend mirrors IRO the same way).
"""
import math

L, W = 120.0, 80.0
GOAL = (120.0, 40.0)
POST_LO, POST_HI = 36.34, 43.66
BOX_X, BOX_Y0, BOX_Y1 = 102.0, 18.0, 62.0       # penalty area (attacking frame)

# tunable parameters (documented in README)
P = dict(
    xg_b0=-1.277, xg_bd=-0.098, xg_bt=0.839, head_mult=0.55, press_xg=0.35, block_xg=0.15, pen_xg=0.78,
    conv=0.42,            # share of a pass/carry "threat" that turns into a realised shot (delay factor)
    pass_b0=3.4, pass_bd=0.05, press_p=1.0, press_r=1.0, lane=1.8, high=0.4,
)


def hyp(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def att(team_dir: int, x: float, y: float):
    """global -> attacking frame (team_dir=+1 attacks x=120, -1 attacks x=0). Self-inverse."""
    return (x, y) if team_dir > 0 else (L - x, W - y)


def goal_angle(x, y):
    dx = GOAL[0] - x
    a, b = math.hypot(dx, POST_LO - y), math.hypot(dx, POST_HI - y)
    c = POST_HI - POST_LO
    v = (a * a + b * b - c * c) / (2 * a * b + 1e-9)
    return math.acos(max(-1.0, min(1.0, v)))


def xg_shot(x, y, part="foot", pressure=0.0, first_time=False, blockers=0, penalty=False):
    """xG of a shot from attacking-frame (x, y): distance + angle + body part + pressure + lane blockers."""
    if penalty:
        return P["pen_xg"]
    d = hyp(x, y, *GOAL)
    z = P["xg_b0"] + P["xg_bd"] * d + P["xg_bt"] * goal_angle(x, y)
    p = 1 / (1 + math.exp(-z))
    if part == "head":
        p *= P["head_mult"]
    if first_time and part != "head":
        p *= 0.93
    p *= 1 - P["press_xg"] * pressure
    p *= max(0.25, 1 - P["block_xg"] * blockers)
    return max(0.003, min(0.9, p))


def threat(x, y):
    """Positional value of holding the ball at attacking-frame (x, y): open-shot xG + a progression term."""
    return 0.9 * xg_shot(x, y) + 0.035 * (max(0.0, x) / L) ** 2.5


def pressure_at(x, y, opps):
    """0..1 pressure from opponents' (attacking-frame) positions."""
    best = min((hyp(x, y, ox, oy) for ox, oy in opps), default=99.0)
    n5 = sum(1 for ox, oy in opps if hyp(x, y, ox, oy) < 5.0)
    return max(0.0, min(1.0, (1 - best / 7.0) + 0.12 * max(0, n5 - 1)))


def seg_dist(px, py, ax, ay, bx, by):
    """distance from point P to segment AB, and the projection parameter u in [0,1]"""
    dx, dy = bx - ax, by - ay
    n = dx * dx + dy * dy
    u = 0.0 if n < 1e-9 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / n))
    return hyp(px, py, ax + u * dx, ay + u * dy), u


def lane_block(ax, ay, bx, by, opps, radius=2.2):
    """0..1 how blocked the straight lane A->B is (opponents near the segment, not at its ends)."""
    s = 0.0
    for ox, oy in opps:
        d, u = seg_dist(ox, oy, ax, ay, bx, by)
        if 0.12 < u < 0.97 and d < radius:
            s += (1 - d / radius)
    return min(1.0, s)


def lane_blockers(ax, ay, bx, by, opps, radius=1.6):
    return sum(1 for ox, oy in opps if 0.1 < seg_dist(ox, oy, ax, ay, bx, by)[1] < 0.97 and seg_dist(ox, oy, ax, ay, bx, by)[0] < radius)


def pass_completion(dist, high, press_passer, press_recv, lane, skill=0.0, through=False):
    z = (P["pass_b0"] - P["pass_bd"] * dist - P["press_p"] * press_passer - P["press_r"] * press_recv
         - P["lane"] * lane + skill - (P["high"] if high else 0.0) - (0.45 if through else 0.0))
    return 1 / (1 + math.exp(-z))


def offside_margin_att(recv_x, ball_x, def_xs):
    """Attacking frame. Positive margin (metres) => receiver is in an offside position when the ball is played:
    he is in the opponent half and nearer the goal line than BOTH the ball and the second-last defender (keeper counts).
    `def_xs` = x of all outfield+keeper defenders. Also returns the offside line used."""
    xs = sorted(def_xs, reverse=True)                     # deepest (nearest own goal line) first
    second_last = xs[1] if len(xs) > 1 else xs[0]
    line = max(second_last, ball_x, 60.0)
    return recv_x - line, line
