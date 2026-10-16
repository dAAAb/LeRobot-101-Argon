"""Empirical empty-jaw comparison, NOT a grasp/force detector.

Inputs use measured monotonic capture timestamps and raw servo units. `speed` is
the commanded speed setting, `pwm_limit` its exact cap setting, `servo_status=0`
is a decoded clean status, and `phase='cruise'` means the caller has already
excluded acceleration. `load_signed_pwm` is signed CONTROL OUTPUT, not force.
Optional current is only reported as raw changing/flat/unavailable evidence.

Construct with explicit calibrated position bounds, closing direction sign,
and angle-bin width. Build the baseline from >=3 physically separate empty-jaw
cycles. Every supported context/bin requires >=3 cycles and >=12 samples, with
>=3 samples from each contributing cycle. No interpolation between unsupported
bins, directions, speed settings or PWM caps. Baseline fitting excludes faults,
endpoints, acceleration, reversal guards and incomplete context transitions.

Classification: unknown / free / stalled / contact_candidate. `free` means
consistent with the measured empty-jaw baseline; `stalled` means anomalous slow
progress/tracking error without sustained excess PWM. Neither establishes the
presence/absence of an object. Even contact_candidate is never true grasp.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
import math
import time
import numpy as np
from .gripper_context import measured_body, require_matched_body


@dataclass(frozen=True)
class Sample:
    timestamp: float
    direction: str
    goal: float
    position: float
    load_signed_pwm: float
    pwm_limit: float
    speed: float
    servo_status: int = 0
    phase: str = 'cruise'
    valid: bool = True
    current_raw: float | None = None
    current_valid: bool = False
    body_position: dict | None = None


@dataclass(frozen=True)
class Config:
    position_min: float
    position_max: float
    closing_sign: int
    angle_bin_width: float
    endpoint_margin: float
    position_resolution: float = 1.
    pwm_resolution: float = 1.
    min_cycles: int = 3
    min_samples: int = 12
    min_samples_per_cycle: int = 3
    min_fresh_samples: int = 3
    sustain_seconds: float = .15
    max_gap_seconds: float = .30
    max_age_seconds: float = .30
    transition_guard_seconds: float = .20
    slow_progress_fraction: float = .20

    def __post_init__(self):
        if (self.closing_sign not in (-1, 1) or isinstance(self.closing_sign, bool) or
                not all(math.isfinite(x) for x in [self.position_min, self.position_max,
                    self.angle_bin_width, self.endpoint_margin, self.position_resolution,
                    self.pwm_resolution, self.sustain_seconds, self.max_gap_seconds,
                    self.max_age_seconds, self.transition_guard_seconds, self.slow_progress_fraction]) or
                not self.position_min < self.position_max or self.angle_bin_width <= 0 or
                not 0 < self.endpoint_margin < (self.position_max-self.position_min)/2 or
                self.position_resolution <= 0 or self.pwm_resolution <= 0 or
                self.min_cycles < 3 or self.min_samples < 12 or self.min_samples_per_cycle < 3 or
                self.min_fresh_samples < 3 or not 0 < self.sustain_seconds <= 2 or
                not 0 < self.max_gap_seconds <= 1 or not 0 < self.max_age_seconds <= 1 or
                self.transition_guard_seconds <= 0 or not 0 < self.slow_progress_fraction <= .25):
            raise ValueError('Invalid calibration/evidence configuration')


def _context(s):
    return (s.direction, float(s.speed), float(s.pwm_limit))


def _reason(s, c):
    try:measured_body(s.body_position)
    except ValueError as exc:return str(exc)
    values = [s.timestamp, s.goal, s.position, s.load_signed_pwm, s.pwm_limit, s.speed]
    if (s.valid is not True or any(isinstance(x, bool) or not isinstance(x, (int, float)) or
            not math.isfinite(x) for x in values)):
        return 'invalid sample'
    if not isinstance(s.servo_status, int) or isinstance(s.servo_status, bool) or s.servo_status != 0:
        return 'servo fault or unknown status'
    if s.direction not in ('closing', 'opening') or s.pwm_limit <= 0 or s.speed <= 0:
        return 'invalid direction, cap, or speed'
    if s.phase != 'cruise':
        return 'acceleration/reversal/non-cruise phase'
    if not c.position_min <= s.goal <= c.position_max:
        return 'goal outside calibrated travel'
    if not c.position_min+c.endpoint_margin < s.position < c.position_max-c.endpoint_margin:
        return 'endpoint exclusion'
    sign = c.closing_sign if s.direction == 'closing' else -c.closing_sign
    if sign*(s.goal-s.position) <= c.position_resolution:
        return 'goal reached or contradicts requested direction'
    if s.current_valid is True and (s.current_raw is None or isinstance(s.current_raw, bool) or
            not isinstance(s.current_raw, (int, float)) or not math.isfinite(s.current_raw)):
        return 'invalid reported current'
    return None


def _key(s, c):
    return (math.floor((s.position-c.position_min)/c.angle_bin_width), *_context(s))


def _feature(previous, s, c):
    sign = c.closing_sign if s.direction == 'closing' else -c.closing_sign
    dt = s.timestamp-previous.timestamp
    return {'progress': sign*(s.position-previous.position)/dt,
            'tracking_error': sign*(s.goal-s.position),
            'pwm_effort': max(0., sign*s.load_signed_pwm), 'dt': dt}


def _upper(values, resolution):
    a = np.asarray(values, dtype=float)
    mad = float(np.median(np.abs(a-np.median(a))))
    # Never call an already observed empty-jaw friction spike novel merely
    # because it occupied <5% of a bin. The empirical maximum is a conservative
    # floor under the quantile/MAD envelope, still expressed in raw units.
    return float(max(np.quantile(a, .95)+3*1.4826*mad, np.max(a))+resolution)


@dataclass(frozen=True)
class EmptyJawBaseline:
    config: Config
    bins: dict
    body_reference: dict


def fit_empty_baseline(cycles, config, *, body_reference=None):
    """Returns matched empirical bins; bad/undersampled bins remain unsupported.

    Cycle identity is the list position. Callers must ensure these are independent
    physical empty-jaw cycles, not copies/slices of the same cycle. Temporal
    intervals may not overlap between cycles; historical capture age is fine.
    """
    if len(cycles) < config.min_cycles:
        raise ValueError('At least three independent empty-jaw cycles are required')
    reference=measured_body(body_reference if body_reference is not None else
                            next((s.body_position for cycle in cycles for s in cycle),None))
    bins = defaultdict(lambda: defaultdict(list)); prior_end = None
    for cycle_id, cycle in enumerate(cycles):
        if not cycle:
            continue
        times = [s.timestamp for s in cycle]
        if (not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in times) or
                any(b <= a for a, b in zip(times, times[1:])) or
                (prior_end is not None and times[0] <= prior_end)):
            raise ValueError('Cycles need unique, increasing, non-overlapping capture timestamps')
        prior_end = times[-1]
        previous = None; context = None; transition_at = times[0]
        for s in cycle:
            # Changed gravity context invalidates the acquisition, not merely
            # an isolated angle bin. Do not pool trajectories from other poses.
            require_matched_body(s.body_position,reference)
            if _reason(s, config):
                previous = None; context = None; transition_at = s.timestamp; continue
            ctx = _context(s)
            if previous is None or ctx != context or s.timestamp-previous.timestamp > config.max_gap_seconds:
                previous = s; context = ctx; transition_at = s.timestamp; continue
            if s.timestamp-transition_at < config.transition_guard_seconds:
                previous = s; continue
            f = _feature(previous, s, config)
            if f['progress'] < 0:
                previous = s; transition_at = s.timestamp; continue
            bins[_key(s, config)][cycle_id].append(f)
            previous = s
    baseline = {}
    for key, by_cycle in bins.items():
        supported = {i: rows for i, rows in by_cycle.items() if len(rows) >= config.min_samples_per_cycle}
        rows = [r for rs in supported.values() for r in rs]
        if len(supported) < config.min_cycles or len(rows) < config.min_samples:
            continue
        progress = np.array([r['progress'] for r in rows]); median = float(np.median(progress))
        # A flat/noisy empty baseline cannot establish that low motion is unusual.
        if median <= config.position_resolution/float(np.median([r['dt'] for r in rows])):
            continue
        baseline[key] = {'cycles': len(supported), 'samples': len(rows),
                         'tracking_error_upper': _upper([r['tracking_error'] for r in rows], config.position_resolution),
                         'pwm_effort_upper': _upper([r['pwm_effort'] for r in rows], config.pwm_resolution),
                         'envelope': 'max(q95 + 3 scaled MAD, observed maximum) + raw resolution',
                         'progress_median': median,
                         'progress_low_limit': max(0., min(float(np.quantile(progress, .10)),
                                                          median*config.slow_progress_fraction))}
    return EmptyJawBaseline(config, baseline, reference)


class ContactCandidateDetector:
    def __init__(self, config, baseline):
        if not isinstance(baseline, EmptyJawBaseline) or baseline.config != config:
            raise ValueError('Baseline configuration must match the active detector exactly')
        self.config = config; self.baseline = baseline.bins
        self.body_reference=measured_body(baseline.body_reference)
        self.reset()

    @classmethod
    def from_empty_cycles(cls, config, cycles):
        return cls(config, fit_empty_baseline(cycles, config))

    def reset(self):
        self.previous = None; self.context = None; self.transition_at = None
        self.last_timestamp = None
        self.pending = deque(maxlen=max(512, self.config.min_fresh_samples))

    def update(self, sample, *, now=None, cancelled=False):
        c = self.config; s = sample
        out = {'state': 'unknown', 'true_grasp': False, 'force_estimate': None,
               'evidence': 'position tracking and raw PWM only', 'reason': '',
               'fresh_samples': 0}
        def reject(reason, *, reset_clock=False):
            self.previous = None; self.context = None; self.pending.clear(); self.transition_at = None
            if reset_clock: self.last_timestamp = None
            out['reason'] = reason
            return out
        if cancelled:
            return reject('cancelled', reset_clock=True)
        try:require_matched_body(s.body_position,self.body_reference)
        except ValueError as exc:return reject(str(exc))
        reason = _reason(s, c)
        if reason:
            return reject(reason)
        now = time.monotonic() if now is None else now
        if not isinstance(now, (int, float)) or not math.isfinite(now) or not 0 <= now-s.timestamp <= c.max_age_seconds:
            return reject('stale or future capture timestamp')
        if self.last_timestamp is not None and s.timestamp <= self.last_timestamp:
            return reject('duplicate or non-increasing capture timestamp')
        self.last_timestamp = s.timestamp
        ctx = _context(s)
        if self.previous is None or ctx != self.context or s.timestamp-self.previous.timestamp > c.max_gap_seconds:
            self.previous = s; self.context = ctx; self.transition_at = s.timestamp; self.pending.clear()
            out['reason'] = 'new context/reversal/gap guard'; return out
        previous = self.previous; self.previous = s
        if s.timestamp-self.transition_at < c.transition_guard_seconds:
            self.pending.clear(); out['reason'] = 'acceleration/reversal settling guard'; return out
        baseline = self.baseline.get(_key(s, c))
        if baseline is None:
            self.pending.clear(); out['reason'] = 'no matched independent empty-jaw baseline'; return out
        f = _feature(previous, s, c)
        if f['progress'] < -c.position_resolution/f['dt']:
            self.pending.clear(); self.transition_at = s.timestamp
            out['reason'] = 'observed reverse motion'; return out
        out.update(context={'angle_bin': _key(s, c)[0], 'direction': s.direction,
                            'speed': s.speed, 'pwm_limit': s.pwm_limit},
                   measurements=f, matched_baseline=dict(baseline))
        slow = abs(f['progress']) <= baseline['progress_low_limit']
        error_high = f['tracking_error'] > baseline['tracking_error_upper']
        effort_high = f['pwm_effort'] > baseline['pwm_effort_upper']
        out['exceeds_baseline'] = {'slow_progress': slow, 'tracking_error': error_high, 'pwm_effort': effort_high}
        if not (slow and error_high):
            self.pending.clear()
            if f['progress'] >= baseline['progress_low_limit'] and not error_high and not effort_high:
                out.update(state='free', reason='motion consistent with matched empty-jaw baseline')
            else:
                out['reason'] = 'partial or inconsistent anomaly; insufficient contact evidence'
            return out
        self.pending.append((s.timestamp, effort_high, s.current_raw if s.current_valid else None))
        while len(self.pending)>1 and self.pending[1][0] < s.timestamp-max(c.sustain_seconds*2,c.max_gap_seconds*2):
            self.pending.popleft()
        out['fresh_samples'] = len(self.pending)
        out['sustained_seconds'] = self.pending[-1][0]-self.pending[0][0]
        if len(self.pending) < c.min_fresh_samples or self.pending[-1][0]-self.pending[0][0] < c.sustain_seconds:
            out['reason'] = 'waiting for sustained fresh evidence'; return out
        # All accumulated sustained samples must support effort increase; a spike
        # does not become contact merely because position was already stalled.
        excess_effort = all(row[1] for row in self.pending)
        current = [row[2] for row in self.pending if row[2] is not None]
        out['current_evidence'] = ('unavailable_or_flat_not_force_evidence' if len(current)<3 or max(current)==min(current)
                                   else 'varying_raw_current_not_calibrated_force')
        if s.direction == 'closing' and excess_effort:
            out.update(state='contact_candidate', reason='sustained closing stall with excess tracking error and PWM')
        else:
            out.update(state='stalled', reason='sustained tracking stall; no supported closing-contact conclusion')
        return out
