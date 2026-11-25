"""Experimental pressure/length/time-aware ink policy, shared by plan and run.

Equivalent millimetres are a heuristic consumption index, NOT measured ink
volume. Timings and capacity require paper validation; no calibration changes.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import numpy as np
from config import ArmConfig, ShuimoConfig, JobConfig, PALETTE
from shuimo import press_to_width_mm


@dataclass(frozen=True)
class InkPolicy:
    capacity_eq_mm: float = 170.
    reserve_eq_mm: float = 25.
    max_segment_eq_mm: float = 120.
    max_age_s: float = 75.
    max_idle_s: float = 15.
    initial_dip_s: float = .6
    refresh_dip_s: float = .3
    drip_s: float = 2.5
    draw_mm_s: float = 3.5  # conservative planning estimate, not controller v%
    paper_travel_mm_s: float = 35.
    return_allowance_s: float = 8.
    fresh_distance_mm: float = 12.
    fresh_press_factor: float = .8
    rest_every: int = 0  # 0 disables scheduled rests
    rest_s: float = 20.
    max_strokes_per_load: int = 0  # 0 keeps the legacy length/time policy.

    def __post_init__(self):
        values = asdict(self)
        values.pop("rest_every")
        values.pop("max_strokes_per_load")
        if type(self.max_strokes_per_load) is not int or self.max_strokes_per_load < 0:
            raise ValueError("max_strokes_per_load must be a nonnegative integer")
        if not isinstance(self.rest_every, int) or isinstance(self.rest_every, bool) or self.rest_every < 0:
            raise ValueError("rest_every must be a nonnegative integer")
        if any(not math.isfinite(float(v)) or v <= 0 for v in values.values()):
            raise ValueError("Ink policy requires finite positive values")
        if self.max_segment_eq_mm + self.reserve_eq_mm > self.capacity_eq_mm:
            raise ValueError("Segment exceeds usable ink budget")
        if self.fresh_press_factor > 1:
            raise ValueError("Fresh-ink compensation must not increase pressure")


def distances(stroke, scale):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(stroke["points"], axis=0), axis=1))] * scale


def cumulative_cost(stroke, scale):
    arc = distances(stroke, scale)
    width = np.array([press_to_width_mm(p) for p in stroke["press"]])
    return np.r_[0., np.cumsum(np.diff(arc) * (width[:-1] + width[1:]) / (2 * .9))]


def cost(stroke, scale):
    return float(cumulative_cost(stroke, scale)[-1])


def refresh_stats(s, scale):
    s["length_mm"] = float(distances(s, scale)[-1])
    s["press_mean"] = float(np.mean(s["press"]))
    s["press_max"] = float(max(s["press"]))
    s["width_mm"] = float(np.mean([press_to_width_mm(p) for p in s["press"]]))
    return s


def split_long_strokes(strokes, scale, policy):
    """Split only over-budget strokes at a valley when available, else a smooth
    point. Shared endpoints preserve geometry; both sides taper at the seam.
    No mid-stroke station excursion is ever generated.
    """
    out = []
    for parent, stroke in enumerate(strokes):
        p = np.asarray(stroke["points"], float)
        cumulative = cumulative_cost(stroke, scale)
        cuts, start = [0], 0
        while cumulative[-1] - cumulative[start] > policy.max_segment_eq_mm:
            last = int(np.searchsorted(cumulative, cumulative[start] + policy.max_segment_eq_mm, side="right") - 1)
            first = max(start + 2, int(np.searchsorted(cumulative, cumulative[start] + .6 * policy.max_segment_eq_mm)))
            candidates = list(range(first, min(last, len(p)-2) + 1))
            if not candidates: raise ValueError("Trajectory samples too coarse for safe ink split")
            valleys = [i for i in candidates if p[i, 1] >= max(p[max(start, i-3), 1], p[min(len(p)-1, i+3), 1])]
            if valleys:
                cut = max(valleys, key=lambda i: (p[i, 1], i))
            else:
                cut = candidates[-1]
            cuts.append(cut)
            start = cut
        cuts.append(len(p)-1)
        for segment, (a, b) in enumerate(zip(cuts, cuts[1:])):
            s = deepcopy(stroke)
            for key in ("points", "press", "speed", "source_width_mm"):
                if key in s: s[key] = s[key][a:b+1]
            if len(cuts) > 2:
                arc = distances(s, scale)
                pressure = np.asarray(s["press"])
                if a > 0:
                    pressure *= .65 + .35 * np.clip(arc / 1.2, 0, 1)
                if b < len(p)-1:
                    pressure *= np.clip((arc[-1]-arc) / 1.2, 0, 1)
                s["press"] = pressure.tolist()
            s.update(parent_stroke_id=parent, segment=segment, segments=len(cuts)-1)
            out.append(refresh_stats(s, scale))
    return out


def order_for_ink(strokes, scale, policy):
    """Fill a load with a nearby fitting stroke of the SAME layer. Parent
    segments remain in order, and no extra mark is drawn merely to use ink.
    This is a planning heuristic; runtime freshness checks still take priority.
    """
    out, current, used = [], np.array([0., 0.]), 0.
    # Cache invariant consumption once; find each parent's first remaining
    # segment in a single scan instead of a nested scan for every candidate.
    costs = [cost(s, scale) for s in strokes]
    for layer in sorted({s["layer"] for s in strokes}):
        pending = [i for i, s in enumerate(strokes) if s["layer"] == layer]
        while pending:
            first = {}
            for i in pending:
                s = strokes[i]
                parent = s["parent_stroke_id"]
                first[parent] = min(first.get(parent, s["segment"]), s["segment"])
            available = [i for i in pending if strokes[i]["segment"] == first[strokes[i]["parent_stroke_id"]]]
            fitting = [i for i in available if used + costs[i] + policy.reserve_eq_mm <= policy.capacity_eq_mm]
            if not fitting:
                used = 0.
                fitting = available
            chosen = min(fitting, key=lambda i: float(np.linalg.norm(np.array(strokes[i]["points"][0])-current)))
            out.append(strokes[chosen])
            pending.remove(chosen)
            used += costs[chosen]
            current = np.array(strokes[chosen]["points"][-1])
    return out


class InkState:
    def __init__(self, policy):
        self.policy = policy
        self.dip_at = None
        self.last_activity = None
        self.used = 0.
        self.drawn_mm = 0.
        self.completed_since_load = 0

    def load(self, now):
        self.dip_at = self.last_activity = float(now)
        self.used = self.drawn_mm = 0.
        self.completed_since_load = 0

    def reasons(self, now, demand, duration):
        if self.dip_at is None: return ["initial"]
        reasons = []
        if (self.policy.max_strokes_per_load and
                self.completed_since_load >= self.policy.max_strokes_per_load):
            reasons.append("stroke_count_budget")
        if self.used + demand + self.policy.reserve_eq_mm > self.policy.capacity_eq_mm + 1e-6:
            reasons.append("pressure_length_budget")
        if now - self.dip_at + duration > self.policy.max_age_s:
            reasons.append("elapsed_time")
        if now - self.last_activity >= self.policy.max_idle_s:
            reasons.append("idle_or_rest")
        return reasons

    def consume(self, stroke, scale, now):
        self.used += cost(stroke, scale)
        self.drawn_mm += stroke["length_mm"]
        self.completed_since_load += 1
        self.last_activity = float(now)


def condition_fresh(stroke, state, scale):
    s = deepcopy(stroke)
    p = state.policy
    arc = distances(s, scale)
    factor = p.fresh_press_factor + (1-p.fresh_press_factor) * np.clip(
        (state.drawn_mm + arc) / p.fresh_distance_mm, 0, 1)
    s["press"] = (np.asarray(s["press"]) * factor).tolist()
    return refresh_stats(s, scale)


def duration(stroke, policy):
    return float(stroke["length_mm"]) / policy.draw_mm_s + 1.2


def dip_step(pal_index, initial, reasons, policy):
    return dict(kind="dip", pal_index=pal_index, reason=", ".join(reasons),
                dwell_s=policy.initial_dip_s if initial else policy.refresh_dip_s,
                drip_s=policy.drip_s)


def build_plan(strokes, pal_index, scale, policy=None):
    policy = policy or InkPolicy()
    if policy.max_strokes_per_load and JobConfig.SKIP_PIGMENT_ACTIONS:
        raise ValueError("Stroke-count loading requires pigment actions")
    state = InkState(policy)
    steps, rows = [], []
    now, current = 0., None
    dips = 0
    for i, original in enumerate(strokes):
        demand = cost(original, scale)
        draw_s = duration(original, policy)
        if demand + policy.reserve_eq_mm > policy.capacity_eq_mm:
            raise ValueError("Split over-budget strokes before ink planning")
        if draw_s + policy.return_allowance_s > policy.max_age_s:
            raise ValueError("One stroke is too slow for the configured freshness interval")
        if policy.rest_every and i and i % policy.rest_every == 0:
            steps.append(dict(kind="rest", seconds=policy.rest_s))
            now += policy.rest_s
        travel = 1. if current is None else float(np.linalg.norm(np.array(original["points"][0])-current)) * scale / policy.paper_travel_mm_s + 1.
        reasons = state.reasons(now, demand, draw_s + travel)
        if reasons and not JobConfig.SKIP_PIGMENT_ACTIONS:
            step = dip_step(pal_index, state.dip_at is None, reasons, policy)
            steps.append(step)
            # Outbound transfer, dwell and drain; brush age starts after drain.
            now += policy.return_allowance_s + step["dwell_s"] + step["drip_s"]
            state.load(now)
            dips += 1
            travel = policy.return_allowance_s
        elif state.dip_at is None:
            state.load(now)
        s = condition_fresh(original, state, scale)
        now += travel
        rows.append(dict(stroke=i+1, dip_cycle=dips, reason=reasons,
                         start_age_s=round(now-state.dip_at, 2),
                         end_age_s=round(now+draw_s-state.dip_at, 2),
                         used_before_eq_mm=round(state.used, 2),
                         demand_eq_mm=round(cost(s, scale), 2),
                         length_mm=round(s["length_mm"], 2),
                         layer=s["layer"]))
        steps.append(dict(kind="stroke", stroke=s, pal_index=pal_index, layer=s["layer"]))
        now += draw_s
        state.consume(s, scale, now)
        current = np.array(s["points"][-1])
    total = sum(s["length_mm"] for s in strokes)
    stats = dict(strokes=len(strokes), length_mm=total, dips=dips, washes=0, wipes=0, tests=0,
                 lookahead_dips=max(0, dips-1), capacity_mm=policy.capacity_eq_mm,
                 overlong_count=0, overlong_max_mm=0,
                 policy="balanced-experimental", ink_policy=asdict(policy), ink_cycles=rows,
                 estimated_total_s=round(now, 1),
                 total_dip_dwell_s=sum(s.get("dwell_s", 0) for s in steps),
                 groups=[dict(layer=1, name=PALETTE[pal_index]["name"], pal_index=pal_index,
                              strokes=len(strokes), length_mm=round(total, 1))])
    return steps, stats
