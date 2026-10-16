"""Pure bounded body endpoint control and read-only hold-anchor bookkeeping.
No motor, camera, file, sleep, network, or I/O calls. Caller owns every write.
"""
from dataclasses import dataclass
from math import isfinite
from .feedback import BODY_JOINTS, EndpointFeedback

ALL_JOINTS = BODY_JOINTS | {'gripper'}


def tick(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or int(value) != value:
        raise ValueError('Expected a finite integer raw tick')
    return int(value)


def check_bounds(pose, bounds, names=None):
    """Both measured coordinates and every proposed raw goal use this same box."""
    names = set(bounds) if names is None else set(names)
    if not names or not names <= ALL_JOINTS or not names <= set(bounds) or not names <= set(pose):
        raise ValueError('Missing explicit local raw bounds or joint state')
    for name in names:
        lo, hi = map(tick, bounds[name])
        if not 0 <= lo < hi <= 4095 or not lo <= tick(pose[name]) <= hi:
            raise ValueError('Outside explicitly screened raw interval: ' + name)


def adopt_goal_anchor(actual, physical_goals, *, previous_goals=None, previous_bias=None):
    """Caller has already verified live torque/status and stable measured hold.

    Cold adoption establishes a NEW measured logical anchor, not a reconstruction
    of the old intended target. It leaves the existing physical goals untouched.
    Warm adoption preserves known bias exactly, even if actual drifts slightly.
    Gripper is never given position-error compensation.
    """
    if set(actual) != ALL_JOINTS or set(physical_goals) != ALL_JOINTS:
        raise ValueError('All six actual positions and physical goals are required')
    actual = {n: tick(v) for n, v in actual.items()}
    physical = {n: tick(v) for n, v in physical_goals.items()}
    if any(not 0 <= v <= 4095 for v in [*actual.values(), *physical.values()]):
        raise ValueError('Invalid raw position')
    if previous_goals is not None:
        if set(previous_goals) != ALL_JOINTS or previous_bias is None or set(previous_bias) != ALL_JOINTS:
            raise ValueError('Incomplete warm hold state')
        if any(tick(previous_goals[n]) != physical[n] for n in ALL_JOINTS):
            raise ValueError('Physical hold goals changed outside the tracked command ledger')
        bias = {n: tick(previous_bias[n]) for n in ALL_JOINTS}
        mode = 'warm-preserved'
    else:
        if (any(abs(physical[n] - actual[n]) > EndpointFeedback.MAX_BIAS for n in BODY_JOINTS) or
                abs(physical['gripper'] - actual['gripper']) > 16):
            raise ValueError('Cold adoption exceeds body bias budget or the 16-tick gripper hold window')
        bias = {n: physical[n] - actual[n] if n in BODY_JOINTS else 0 for n in ALL_JOINTS}
        mode = 'cold-measured-anchor'
    if bias['gripper'] != 0 or any(abs(v) > EndpointFeedback.MAX_BIAS for v in bias.values()):
        raise ValueError('Invalid retained compensation')
    logical = {n: physical[n] - bias[n] for n in ALL_JOINTS}
    if any(not 0 <= v <= 4095 for v in logical.values()):
        raise ValueError('Invalid logical hold anchor')
    return {'physical_goals': physical, 'logical_goals': logical, 'bias': bias, 'mode': mode}


@dataclass(frozen=True)
class LocalDecision:
    status: str
    correction: object = None
    reason: str = ''


class LocalEndpoint:
    """Correct the worst body-axis error first, with one global 3-second budget.

    Reuses EndpointFeedback's 6-tick/0.12s, +/-60 bias, reversal, and four-write
    no-progress limits. Final completion needs three simultaneous in-tolerance
    samples. Previously settled axes may be corrected again if coupling moves
    them; no controller/timer is restarted and no nominal pose replaces actual.
    """
    def __init__(self, targets, bias, physical_goals, bounds, precision=3, started_at=None, *, stall_response_wait=False):
        if not targets or not set(targets) <= BODY_JOINTS:
            raise ValueError('Endpoint feedback only supports explicitly commanded body axes')
        check_bounds(targets, bounds, targets)
        check_bounds(physical_goals, bounds, targets)
        self.targets = {n: tick(v) for n, v in targets.items()}
        self.bounds = {n: tuple(bounds[n]) for n in targets}
        self.precision = precision
        if type(stall_response_wait) is not bool or (stall_response_wait and set(targets)!={'elbow_flex'}):
            raise ValueError('Stalled-response observation requires a single elbow endpoint')
        self.controllers = {n: EndpointFeedback(n, v, *self.bounds[n], initial_bias=tick(bias[n]), precision=precision,
                                                stall_response_wait=stall_response_wait)
                            for n, v in self.targets.items()}
        if any(c.motor_goal != tick(physical_goals[n]) for n, c in self.controllers.items()):
            raise ValueError('Logical target plus bias does not match the acknowledged physical endpoint')
        if started_at is not None and (isinstance(started_at, bool) or not isinstance(started_at, (int, float)) or not isfinite(started_at)):
            raise ValueError("Invalid start timestamp")
        self.started = started_at; self.last_seen = started_at; self.last_sample = None
        self.first_error = {}; self.previous_error = {}
        self.stable = 0; self.pending = None; self.terminal = None
        self.waiting_response=False

    def poll_delay(self,now):
        """Do not sleep past an enabled response window or its global deadline."""
        if not self.waiting_response:return EndpointFeedback.PERIOD
        deadlines=[c.stall_response_deadline for c in self.controllers.values() if c.stall_response_deadline is not None]
        if not deadlines:return EndpointFeedback.PERIOD
        if self.started is not None:deadlines.append(self.started+EndpointFeedback.MAX_SECONDS)
        return max(0.,min(EndpointFeedback.PERIOD,min(deadlines)-now))

    def observe(self, actual, now):
        if self.terminal is not None:
            return self.terminal
        self.waiting_response=False
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not isfinite(now):
            return self.abort('Invalid monotonic timestamp')
        if self.last_seen is not None and now < self.last_seen:
            return self.abort('Monotonic clock moved backwards')
        self.last_seen = now
        if self.started is None: self.started = now
        if now - self.started >= EndpointFeedback.MAX_SECONDS:
            return self.abort('Local endpoint exceeded 3 seconds')
        if self.pending is not None:
            return self.abort('Previous proposal was not committed after its successful write')
        check_bounds(actual, self.bounds)
        errors = {n: self.targets[n] - actual[n] for n in self.targets}
        for name, error in errors.items():
            if abs(error) > EndpointFeedback.REVERSAL_TICKS and (
                error * self.first_error.get(name, 0) < 0 or error * self.previous_error.get(name, 0) < 0):
                return self.abort(name + ': position error reversed by more than 8 ticks')
            if error: self.previous_error[name] = error
        for name,controller in self.controllers.items():
            response=controller.stalled_response(actual[name],now)
            if response is not None:
                self.stable=0
                if response.status=='abort':return self.abort(name+': '+response.reason)
                self.waiting_response=True
                return LocalDecision('wait',reason=response.reason)
        if any(abs(v) > self.precision for v in errors.values()): self.stable = 0
        if self.last_sample is not None and now - self.last_sample < EndpointFeedback.PERIOD - 1e-9:
            return LocalDecision('wait')
        self.last_sample = now
        if all(abs(v) <= self.precision for v in errors.values()):
            self.stable += 1
            if self.stable >= EndpointFeedback.REQUIRED_STABLE_SAMPLES:
                self.terminal = LocalDecision('done', reason='Three simultaneous measured endpoint samples')
                return self.terminal
            return LocalDecision('settling')
        name = max(errors, key=lambda n: abs(errors[n]))
        decision = self.controllers[name].observe(actual[name], now)
        if decision.status == 'abort': return self.abort(name + ': ' + decision.reason)
        if decision.status == 'propose':
            self.first_error.setdefault(name, errors[name])
            self.pending = LocalDecision('propose', correction=decision)
            return self.pending
        return LocalDecision(decision.status, reason=decision.reason)

    def validate_before_write(self, decision, now):
        if decision is not self.pending or self.pending is None:
            raise ValueError('Stale or foreign correction')
        if not isinstance(now, (int, float)) or isinstance(now, bool) or not isfinite(now) or now < self.last_seen:
            raise ValueError('Invalid pre-write time')
        if now - self.started >= EndpointFeedback.MAX_SECONDS:
            raise RuntimeError('Local endpoint expired before write')

    def commit(self, decision, now):
        if decision is not self.pending or self.pending is None:
            raise ValueError('Stale or foreign correction')
        self.controllers[decision.correction.joint].commit(decision.correction, now)
        self.last_seen = now; self.pending = None

    def abort(self, reason):
        self.pending = None
        self.terminal = LocalDecision('abort', reason=reason)
        return self.terminal
