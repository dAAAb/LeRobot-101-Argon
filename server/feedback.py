"""Bounded endpoint feedback; standard library only, with NO hardware access.

Integration contract:
  1. Construct one EndpointFeedback per commanded BODY joint. ``initial_bias``
     must describe the motor goal already in force: logical_target + bias.
  2. Feed fresh measured position and a monotonic timestamp to observe().
  3. For a 'propose' decision, the caller rechecks cancellation / camera / load /
     temperature / joint limits, writes decision.goal, then calls commit()
     with the monotonic time AFTER that successful write.
     If that write fails, abort the containing robot operation; do not commit.
  4. 'abort' means stop the operation and perform the caller's best-effort hold.
     'done' means the existing compensated goal should remain in force.

Keep logical trajectory targets separate from physical goals across segments.
Do not seed the next segment from measured position and discard this bias.
This compensates observed position error, not force: the caller must supervise
contact/current and must not treat convergence as collision or grasp safety.
"""

from dataclasses import dataclass
import math


BODY_JOINTS = frozenset((
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
))


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _integer(value, name):
    if not _finite(value) or int(value) != value:
        raise ValueError(f"{name} must be an integer tick value")
    return int(value)


@dataclass(frozen=True)
class FeedbackDecision:
    status: str  # wait | settling | propose | done | abort
    joint: str
    logical_target: int
    bias: int
    error: float
    goal: int | None = None
    reason: str = ""


class EndpointFeedback:
    """One endpoint, one body joint. All time is supplied by the caller.

    Limits must be the caller's already validated safe interval (including its
    desired margin), not a request to expand onboard travel limits.
    Defaults intentionally implement the commissioned bounds without tunable
    overrides: 6 ticks / 0.12 s, gain .35, bias +/-60, duration 3 s.
    """

    PERIOD = .12
    GAIN = .35
    MAX_INCREMENT = 6
    MAX_BIAS = 60
    MAX_SECONDS = 3.0
    REVERSAL_TICKS = 8
    REQUIRED_STABLE_SAMPLES = 3
    PROGRESS_WINDOW = 4
    MIN_MOVEMENT = 3.0
    STALL_RESPONSE_SECONDS = .5

    def __init__(self, joint, logical_target, lower, upper, *, initial_bias=0, precision=2, stall_response_wait=False):
        if joint not in BODY_JOINTS:
            raise ValueError("Only body joints are supported; gripper compensation is excluded")
        self.joint = joint
        if type(stall_response_wait) is not bool or (stall_response_wait and joint!='elbow_flex'):
            raise ValueError('Stalled-response observation is only supported for an explicit elbow endpoint')
        self.stall_response_wait=stall_response_wait
        self.logical_target = _integer(logical_target, "logical_target")
        self.lower = _integer(lower, "lower")
        self.upper = _integer(upper, "upper")
        self.bias = _integer(initial_bias, "initial_bias")
        # Local Cartesian completion can need a further one-tick correction
        # after the ordinary 3-tick joint criterion would have settled. This
        # opt-in tolerance changes no rate, bias, deadline, or stall limit.
        if precision not in (1, 2, 3) or isinstance(precision, bool):
            raise ValueError("precision must be 1, 2, or 3 ticks")
        self.precision = precision
        if not 0 <= self.lower < self.upper <= 4095:
            raise ValueError("Invalid safe interval")
        if not self.lower <= self.logical_target <= self.upper:
            raise ValueError("Logical target is outside safe interval")
        if abs(self.bias) > self.MAX_BIAS:
            raise ValueError("Initial bias exceeds 60 ticks")
        if not self.lower <= self.motor_goal <= self.upper:
            raise ValueError("Initial compensated motor goal is outside safe interval")
        self.started_at = None
        self.last_seen_at = None
        self.last_sample_at = None
        self.last_correction_at = None
        self.stable_samples = 0
        self.corrections = 0
        self._initial_control_error = None
        self._last_nonzero_error = None
        self._anchor_actual = None
        self._anchor_error = None
        self._window_corrections = 0
        self._pending = None
        self._pending_at = None
        self._terminal = None

    @property
    def motor_goal(self):
        return self.logical_target + self.bias

    def _decision(self, status, error, *, goal=None, bias=None, reason=""):
        return FeedbackDecision(status, self.joint, self.logical_target,
                                self.bias if bias is None else bias, error, goal, reason)

    def _abort(self, error, reason):
        self._pending = None
        self._pending_at = None
        self._terminal = self._decision("abort", error, reason=reason)
        return self._terminal

    @property
    def stall_response_deadline(self):
        if self.stall_response_wait and self._window_corrections>=self.PROGRESS_WINDOW:
            return self.last_correction_at+self.STALL_RESPONSE_SECONDS
        return None

    def stalled_response(self, actual, now):
        """Write-free response gate after four commits; caller validates the sample.

        LocalEndpoint also applies this before its simultaneous stability check.
        Neither this gate nor its clock restarts the correction/write budget.
        """
        deadline=self.stall_response_deadline
        if deadline is None:return None
        error=self.logical_target-actual
        movement=abs(actual-self._anchor_actual)
        reduction=self._anchor_error-abs(error)
        if movement>=self.MIN_MOVEMENT and reduction>0:return None
        if now>=deadline:
            return self._abort(error,'no progress after four corrections: '
                               f'movement={movement:.2f}, error reduction={reduction:.2f} ticks; response observation expired')
        return self._decision('wait',error,reason='waiting for stalled motor response without another write')

    def observe(self, actual, now):
        """Return a decision without I/O. Proposals are uncommitted until commit()."""
        if self._terminal is not None:
            return self._terminal
        if not _finite(actual) or not _finite(now):
            return self._abort(float("nan"), "nonfinite measurement or timestamp")
        error = self.logical_target - actual
        if self.last_seen_at is not None and now < self.last_seen_at:
            return self._abort(error, "monotonic clock moved backwards")
        self.last_seen_at = now
        if self.started_at is None:
            self.started_at = now
        if now - self.started_at >= self.MAX_SECONDS:
            return self._abort(error, "endpoint feedback exceeded 3 seconds")
        if not self.lower <= actual <= self.upper:
            return self._abort(error, "measured position outside safe interval")
        if self._pending is not None:
            return self._abort(error, "previous proposal was not committed after a successful write")

        # Check large reversal even on an otherwise rate-limited observation.
        reversed_from_initial = (self._initial_control_error is not None
                                 and error * self._initial_control_error < 0)
        reversed_from_previous = (self._last_nonzero_error is not None
                                  and error * self._last_nonzero_error < 0)
        if (reversed_from_initial or reversed_from_previous) and abs(error) > self.REVERSAL_TICKS:
            return self._abort(error, "position error reversed by more than 8 ticks")
        if error != 0:
            self._last_nonzero_error = error

        response=self.stalled_response(actual,now)
        if response is not None:return response

        if self.last_sample_at is not None and now - self.last_sample_at < self.PERIOD - 1e-9:
            # An out-of-tolerance observation breaks a purported stable streak,
            # even though rapid calls cannot count toward the three samples.
            if abs(error) > self.precision:
                self.stable_samples = 0
            return self._decision("wait", error, reason="sample period has not elapsed")
        self.last_sample_at = now

        if abs(error) <= self.precision:
            self.stable_samples += 1
            if self.stable_samples >= self.REQUIRED_STABLE_SAMPLES:
                self._terminal = self._decision("done", error, reason="three sustained endpoint samples")
                return self._terminal
            return self._decision("settling", error, reason="endpoint requires three sustained samples")
        self.stable_samples = 0

        # Test the measured response AFTER four successfully written corrections.
        # Motion alone is insufficient: moving the wrong way must also abort.
        if self._window_corrections >= self.PROGRESS_WINDOW:
            movement = abs(actual - self._anchor_actual)
            reduction = self._anchor_error - abs(error)
            if movement < self.MIN_MOVEMENT or reduction <= 0:
                return self._abort(error, "no progress after four corrections: "
                                   f"movement={movement:.2f}, error reduction={reduction:.2f} ticks")
            self._window_corrections = 0
            self._anchor_actual = None
            self._anchor_error = None

        if self.last_correction_at is not None and now - self.last_correction_at < self.PERIOD - 1e-9:
            return self._decision("wait", error, reason="correction period has not elapsed")

        increment = int(round(self.GAIN * error))
        increment = max(-self.MAX_INCREMENT, min(self.MAX_INCREMENT, increment))
        if increment == 0:
            increment = 1 if error > 0 else -1
        proposed_bias = self.bias + increment
        proposed_goal = self.logical_target + proposed_bias

        # Fail instead of integrating against a saturated bound. Do not silently
        # clip a proposal or accumulate an unapplied internal bias.
        if abs(proposed_bias) > self.MAX_BIAS:
            return self._abort(error, "60-tick compensation budget exhausted")
        if not self.lower <= proposed_goal <= self.upper:
            return self._abort(error, "correction would cross safe joint limit")
        if self._window_corrections == 0:
            self._anchor_actual = actual
            self._anchor_error = abs(error)
        if self._initial_control_error is None:
            self._initial_control_error = error
        self._pending = self._decision("propose", error, goal=proposed_goal, bias=proposed_bias)
        self._pending_at = now
        return self._pending

    def commit(self, decision, now):
        """Acknowledge that the caller successfully wrote exactly decision.goal.

        Call immediately after the write, before another observe(), with its
        completion timestamp. Rate limiting then includes slow serial writes.
        The caller must not reuse old proposals or acknowledge a failed write.
        """
        if self._pending is None or decision is not self._pending:
            raise ValueError("Cannot commit a stale or foreign proposal")
        if not _finite(now) or now < self.last_seen_at:
            raise ValueError("Commit requires a valid monotonic completion timestamp")
        self.bias = decision.bias
        self.last_correction_at = now
        self.last_seen_at = now
        self.corrections += 1
        self._window_corrections += 1
        self._pending = None
        self._pending_at = None
