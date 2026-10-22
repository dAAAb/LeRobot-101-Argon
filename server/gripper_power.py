"""Temporary STS3215 gripper RAM PWM cap with a durable recovery barrier.

Pure I/O abstraction. No imports of Robot, SDK, cameras, or application state.
read(servo_id,address,width)->int; write(servo_id,address,value,width)->None.
The adapter must raise on ANY transport/servo error, and the shared lock must
exclude all other writers. The caller also owns the motion job for the entire
session; it must reject trajectories without the active owner token. An I/O
lock alone does not reserve the intervals between enter() and finish(). Existing Robot._read/_write adapters may translate
id 6 to 'gripper'. stop_latch MUST only latch stop; it must not issue hold/rearm.

Only id 6 RAM addresses 48 (PWM limit) and 42 (goal) can be written. No torque,
EEPROM, or body-joint writes. The device model, position mode and status are read
again at safety boundaries. All cap increases require a newly verified neutral
goal under cap 100, regardless of the journal's historical phase.

Startup: construct then call assert_motion_allowed() in begin/resume. This only
reads the journal. An unfinished/corrupt journal blocks; recovery never runs
automatically. An active session can authorize its caller with its exact token.
Explicit recover() requires the same verified device identity and calibrated
position bounds; it rechecks hardware and re-neutralizes before restoration.
If contact_possible was durably marked, generic finish/recover refuse all I/O.
Such sessions stay at low cap until an explicit opening-only release, measured
stationarity, and fresh trusted wrist-empty evidence. New instances cannot resume
holding: only recover_for_release can adopt a pending contact session.

Use `with session:` for normal completion and explicit cancellation cleanup.
KeyboardInterrupt, asyncio.CancelledError, and SessionCancelled are cancellation.
Other body exceptions leave the pending barrier and latch stop without retries.
For a manual try/finally, call leave_pending() on body I/O/fault exceptions before
finally calls finish(); finish then refuses automatic recovery from that fault. A failure
inside enter/finish is not retried automatically: stop stays latched and explicit
recovery is required. Repeated successful finish does not emit hardware writes.
"""
from __future__ import annotations
import json, math, os, tempfile, time, uuid
from asyncio import CancelledError
from contextlib import contextmanager
from pathlib import Path


class RecoveryRequired(RuntimeError): pass
class SessionFailure(RuntimeError): pass
class SessionCancelled(RuntimeError): pass


class AtomicJournal:
    """Atomic same-directory replacement, fsync file AND directory before return.

    Caller holds the same exclusive controller lock for get/put sequences.
    Completed records are kept as a receipt and may be replaced by a new session.
    A malformed record remains a barrier instead of being silently discarded.
    """
    def __init__(self, path): self.path = Path(path)
    def get(self):
        if not os.path.lexists(self.path): return None
        try:
            flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
            fd = os.open(self.path, flags)
            with os.fdopen(fd, 'r') as stream:
                raw = stream.read(32769)
            if len(raw) > 32768: raise ValueError('oversized journal')
            data = json.loads(raw)
            if not isinstance(data, dict): raise ValueError('journal is not an object')
            return data
        except (OSError, ValueError) as exc:
            raise RecoveryRequired(f'Unreadable pending PWM journal: {exc}') from exc
    def put(self, record):
        data = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=self.path.name+'.', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try: os.fsync(directory)
            finally: os.close(directory)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)


class GripperPWMSession:
    SERVO_ID = 6
    MODEL = 777
    ORIGINAL = 500
    LOW = 100
    PHASES = {'prepared', 'lowered', 'neutralizing', 'neutralized', 'restoring', 'completed',
              'contact_possible', 'holding', 'contact_stopped', 'releasing', 'release_waiting_vision'}
    CONTACT_PHASES = {'contact_possible', 'holding', 'contact_stopped', 'releasing', 'release_waiting_vision'}

    def __init__(self, *, read, write, lock, stop_latch, journal, device_key,
                 position_bounds, clock=time.monotonic, sleep=time.sleep,
                 position_tolerance=0, settle_samples=3, settle_interval=.05,
                 settle_timeout=1., verified_open_position=None, opening_direction=None,
                 empty_evidence_max_age=2., release_tolerance=None, low_cap=100):
        lo, hi = position_bounds
        if (not device_key or not isinstance(device_key, str) or
                not all(isinstance(v, int) and not isinstance(v, bool) for v in (lo, hi)) or
                not 0 <= lo < hi <= 4095 or not isinstance(position_tolerance, int) or
                not 0 <= position_tolerance <= 2 or settle_samples < 3 or
                not .01 <= settle_interval <= .2 or not .1 <= settle_timeout <= 3):
            raise ValueError('Invalid verified device or neutralization bounds')
        if ((verified_open_position is None) != (opening_direction is None) or
                (verified_open_position is not None and
                 (type(verified_open_position) is not int or not lo <= verified_open_position <= hi or
                  type(opening_direction) is not int or opening_direction not in (-1, 1))) or
                isinstance(empty_evidence_max_age, bool) or not isinstance(empty_evidence_max_age, (int, float)) or
                not math.isfinite(empty_evidence_max_age) or not .1 <= empty_evidence_max_age <= 3):
            raise ValueError('Contact lifecycle requires a verified open position and opening direction')
        release_tolerance = position_tolerance if release_tolerance is None else release_tolerance
        if (type(release_tolerance) is not int or not position_tolerance <= release_tolerance <= 6 or
                type(low_cap) is not int or not 100 <= low_cap <= 300):
            raise ValueError('Release tolerance must be 0..6 ticks and the low cap 100..300')
        # Under the low cap the jaw can stall a few ticks short of the verified open goal (servo
        # deadband). The open check accepts that shortfall in the OPENING direction only.
        self.release_tolerance = release_tolerance
        self.LOW = low_cap
        self.read = read; self.write = write; self.lock = lock
        self.stop_latch = stop_latch; self.journal = journal
        self.device_key = device_key; self.bounds = (lo, hi)
        self.clock = clock; self.sleep = sleep; self.tolerance = position_tolerance
        self.settle_samples = settle_samples; self.settle_interval = settle_interval
        self.settle_timeout = settle_timeout
        self.open_position = verified_open_position; self.opening_direction = opening_direction
        self.empty_evidence_max_age = float(empty_evidence_max_age)
        self.token = None; self.state = 'new'
        self.open_retreat_token = None

    def _read(self, address, width=2):
        value = self.read(self.SERVO_ID, address, width)
        if not isinstance(value, int) or isinstance(value, bool):
            raise SessionFailure(f'Noninteger servo value at {address}')
        return value

    def _write(self, address, value):
        if address not in (42, 48): raise AssertionError('RAM-write allowlist violation')
        self.write(self.SERVO_ID, address, value, 2)

    def _guard(self):
        if self._read(3) != self.MODEL: raise SessionFailure('Servo model identity changed')
        if self._read(33, 1) != 0: raise SessionFailure('Servo is not in verified position mode')
        if self._read(65, 1) != 0: raise SessionFailure('Servo protection active; no goal/rearm permitted')

    def _position(self):
        position = self._read(56)
        if not self.bounds[0] <= position <= self.bounds[1]:
            raise SessionFailure('Present position outside verified calibration')
        return position

    def _record(self):
        data = self.journal.get()
        if data is None: return None
        if (type(data.get('version')) is not int or data['version'] not in (1, 2) or not isinstance(data.get('phase'),str) or data['phase'] not in self.PHASES or
                data.get('servo_id') != self.SERVO_ID or data.get('model') != self.MODEL or
                data.get('device_key') != self.device_key or
                data.get('position_bounds') != list(self.bounds) or
                data.get('original_cap') != self.ORIGINAL or
                # A completed record keeps the low cap it actually used; any pending one must match now.
                (data.get('low_cap') != self.LOW and not (data['phase'] == 'completed' and
                    type(data.get('low_cap')) is int and 100 <= data['low_cap'] <= 200)) or
                not isinstance(data.get('session_id'), str) or len(data['session_id']) != 32):
            raise RecoveryRequired('Pending journal identity/schema does not match this verified gripper')
        # v1 is only the pre-contact empty-jaw protocol. All new records are v2;
        # a contact record can never be downgraded to a legacy missing-flag state.
        if data['version'] == 1:
            if data['phase'] in self.CONTACT_PHASES or any(k in data for k in ('contact_possible', 'contact_id', 'release')):
                raise RecoveryRequired('Legacy journal cannot describe a contact session')
        else:
            if type(data.get('contact_possible')) is not bool:
                raise RecoveryRequired('Contact state missing or invalid')
            if data['contact_possible']:
                if (data['phase'] not in self.CONTACT_PHASES or
                        not self._identifier(data.get('contact_id')) or
                        type(data.get('verified_open_position')) is not int or
                        data['verified_open_position'] != self.open_position or
                        type(data.get('opening_direction')) is not int or
                        data['opening_direction'] != self.opening_direction):
                    raise RecoveryRequired('Contact barrier identity/configuration is invalid')
            elif data['phase'] in self.CONTACT_PHASES:
                raise RecoveryRequired('Contact phase has no contact barrier')
            elif 'contact_id' in data and not self._release_receipt_valid(data):
                raise RecoveryRequired('A cleared contact barrier requires a verified release receipt')
        if data['phase'] == 'completed':
            goal, position = data.get('neutral_goal'), data.get('neutral_position')
            if (data.get('restored_cap') != self.ORIGINAL or
                    not all(isinstance(v, int) and not isinstance(v, bool) for v in (goal, position)) or
                    not self.bounds[0] <= goal <= self.bounds[1] or
                    abs(goal-position) > self.tolerance or
                    type(data.get('neutral_samples')) is not int or data['neutral_samples'] < 3):
                raise RecoveryRequired('Completed journal lacks a valid neutralization receipt')
        return data

    @staticmethod
    def _identifier(value):
        return isinstance(value, str) and len(value) == 32 and all(c in '0123456789abcdef' for c in value)

    @staticmethod
    def _finite_stamp(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    def _release_receipt_valid(self, record):
        r = record.get('release_receipt')
        if not isinstance(r, dict): return False
        return (r.get('contact_id') == record.get('contact_id') and
                self._identifier(r.get('release_id')) and r.get('session_id') == record['session_id'] and
                r.get('source') == 'wrist' and r.get('empty_verified') is True and
                type(r.get('open_goal')) is int and r['open_goal'] == record.get('verified_open_position') and
                type(r.get('open_position')) is int and self._open_enough(r['open_position'], r['open_goal']) and
                type(r.get('stationary_samples')) is int and r['stationary_samples'] >= self.settle_samples and
                self._finite_stamp(r.get('stationary_monotonic')) and
                self._finite_stamp(r.get('capture_monotonic')) and r['capture_monotonic'] > r['stationary_monotonic'] and
                isinstance(r.get('camera_generation'), str) and bool(r['camera_generation']) and
                type(r.get('sequence')) is int and r['sequence'] > 0 and
                isinstance(r.get('frame_sha256'), str) and len(r['frame_sha256']) == 64 and
                all(c in '0123456789abcdef' for c in r['frame_sha256']))

    def _phase(self, record, phase, **details):
        updated = {**record, **details, 'phase': phase}
        self.journal.put(updated)
        return updated

    def _failed(self, exc):
        self.state = 'blocked'
        self.stop_latch()
        raise SessionFailure(f'PWM session stopped; pending journal requires explicit recovery: {exc}') from exc

    def assert_motion_allowed(self, owner_token=None):
        """Owner reservation check, not a jaw-command or collision authorization.

        Call from begin/resume. No hardware I/O or auto-recovery. Contact/holding
        ownership permits separately checked body motion; caller must prohibit
        general gripper writes/rearm. Release phase rejects general body begin.
        """
        with self.lock:
            try:
                record = self._record()
            except RecoveryRequired:
                self.stop_latch(); raise
            if record is None or record['phase'] == 'completed': return
            if (self.state in ('active', 'holding') and owner_token is not None and
                    owner_token == self.token == record['session_id'] and
                    record['phase'] in ('lowered', 'contact_possible', 'holding')):
                return
            # Narrow exception: after verify_release_open(allow_retreat=True) measured the jaw stationary
            # at the open position, its owner may move the BODY away before taking the empty-jaw image.
            if (self.state == 'releasing' and owner_token is not None and
                    owner_token == self.token == record['session_id'] == self.open_retreat_token and
                    record['phase'] == 'release_waiting_vision'):
                return
            self.stop_latch()
            raise RecoveryRequired('Unfinished gripper PWM session blocks begin/resume')

    def enter(self):
        with self.lock:
            try:
                record = self._record()
                if record is not None and record['phase'] != 'completed':
                    raise RecoveryRequired('Pending journal must be explicitly recovered before another session')
                if self.state not in ('new', 'completed'):
                    raise RecoveryRequired('This PWM session is not ready to enter')
                self._guard()
                original = self._read(48)
                if original != self.ORIGINAL:
                    raise SessionFailure('Expected verified original RAM PWM cap 500')
                self._position()
                self.token = uuid.uuid4().hex
                record = {'version': 2, 'session_id': self.token, 'phase': 'prepared', 'contact_possible': False,
                          'servo_id': self.SERVO_ID, 'model': self.MODEL,
                          'device_key': self.device_key, 'position_bounds': list(self.bounds),
                          'original_cap': original, 'low_cap': self.LOW}
                # No servo write can precede successful durable publication.
                self.journal.put(record)
                self.state = 'prepared'
                self._guard()
                self._write(48, self.LOW)
                if self._read(48) != self.LOW: raise SessionFailure('Low cap readback mismatch')
                self._guard()
                self._phase(record, 'lowered')
                self.state = 'active'
                return self.token
            except RecoveryRequired:
                self.state = 'blocked'; self.stop_latch(); raise
            except BaseException as exc:
                self._failed(exc)

    def _open_enough(self, position, goal):
        short = self.opening_direction * (goal - position)
        return -self.tolerance <= short <= self.release_tolerance

    def _quiet_check(self, goal, required_cap, *, release=False):
        self._guard()
        if self._read(48) != required_cap: raise SessionFailure('PWM cap changed unexpectedly')
        if self._read(42) != goal: raise SessionFailure('Neutral goal changed unexpectedly')
        position = self._position(); moving = self._read(66, 1)
        if moving not in (0, 1): raise SessionFailure('Invalid moving flag')
        near = self._open_enough(position, goal) if release else abs(position-goal) <= self.tolerance
        return position, moving == 0 and near

    def _neutralize(self, record):
        record = self._phase(record, 'neutralizing')
        self._guard()
        if self._read(48) != self.LOW: raise SessionFailure('Neutralization requires read-back low cap 100')
        goal = self._position()
        # This status read is intentionally immediately before the only goal write.
        if self._read(65, 1) != 0: raise SessionFailure('Fault prohibits neutralizing goal write')
        self._write(42, goal)
        if self._read(42) != goal: raise SessionFailure('Neutral goal write/readback mismatch')
        deadline = self.clock()+self.settle_timeout
        previous = None; stable = 0
        # Deadline plus an iteration bound keeps a faulty injected clock bounded.
        for _ in range(int(self.settle_timeout/self.settle_interval)+2):
            position, quiet = self._quiet_check(goal, self.LOW)
            stable = stable+1 if quiet and (previous is None or abs(position-previous) <= 1) else 0
            previous = position
            if stable >= self.settle_samples:
                return self._phase(record, 'neutralized', neutral_goal=goal,
                                   neutral_position=position, neutral_samples=stable), goal
            if self.clock() >= deadline: break
            self.sleep(self.settle_interval)
        raise SessionFailure('Goal was not verified stationary under the low cap')

    def _complete(self, record, *, recovery=False, cancelled=False):
        # This must precede ALL hardware reads/writes, including lowering a cap
        # after an uncertain acknowledgement. Contact recovery uses release only.
        if record.get('contact_possible'):
            raise RecoveryRequired('Possible held object: explicit measured release and fresh wrist-empty evidence required')
        self._guard()
        cap = self._read(48)
        if cap != self.LOW:
            if not recovery or cap != self.ORIGINAL:
                raise SessionFailure('Unexpected cap; no neutralization or cap increase attempted')
            # A crash/failed acknowledgement may have left 500. Explicit recovery
            # first lowers it, never sends a goal under the higher cap.
            self._guard(); self._write(48, self.LOW)
            if self._read(48) != self.LOW: raise SessionFailure('Recovery low cap readback mismatch')
        record, goal = self._neutralize(record)
        record = self._phase(record, 'restoring', cancelled=bool(cancelled))
        # Journal persistence may take time; repeat the live guard afterward.
        _, quiet = self._quiet_check(goal, self.LOW)
        if not quiet: raise SessionFailure('Gripper no longer neutral before restoration')
        if self._read(65, 1) != 0: raise SessionFailure('Fault prohibits restoring cap')
        self._write(48, self.ORIGINAL)
        if self._read(48) != self.ORIGINAL: raise SessionFailure('Original cap restoration readback mismatch')
        _, quiet = self._quiet_check(goal, self.ORIGINAL)
        if not quiet: raise SessionFailure('Post-restoration neutral goal verification failed')
        self._phase(record, 'completed', restored_cap=self.ORIGINAL)
        self.state = 'completed'
        return {'completed': True, 'restored_cap': self.ORIGINAL, 'neutral_goal': goal,
                'cancelled': bool(cancelled)}

    @contextmanager
    def _contact_operation(self):
        with self.lock:
            try: yield
            except RecoveryRequired:
                self.state = 'blocked'; self.stop_latch(); raise
            except BaseException as exc:
                self._failed(exc)

    def _owned_contact(self, token, *, states=('active', 'holding', 'releasing')):
        record = self._record()
        if (record is None or not record.get('contact_possible') or self.state not in states or
                token is None or token != self.token or token != record['session_id']):
            raise RecoveryRequired('Contact operation requires the live local session owner')
        return record

    def _low_guard(self):
        self._guard()
        if self._read(48) != self.LOW:
            raise SessionFailure('Contact operations require read-back PWM cap 100')

    def mark_contact_possible(self, owner_token):
        """Durably call BEFORE the first potentially contacting closing goal.

        No goal is sent here. Robot integration must gate its own gripper writes;
        assert_motion_allowed reserves the owner and is not a jaw-motion permit.
        Once marked, ordinary finish/recover can never increase power.
        """
        with self._contact_operation():
            record = self._record()
            if (record is None or self.state != 'active' or owner_token is None or
                    owner_token != self.token or owner_token != record['session_id'] or
                    record['phase'] not in ('lowered', 'contact_possible')):
                raise RecoveryRequired('Contact marking requires the active session owner')
            if self.open_position is None:
                raise RecoveryRequired('Verified release position and opening direction have not been configured')
            self._low_guard()
            if not record.get('contact_possible'):
                record = self._phase(record, 'contact_possible', contact_possible=True,
                    contact_id=uuid.uuid4().hex, verified_open_position=self.open_position,
                    opening_direction=self.opening_direction)
            # Persistence can block; do not hand control back with stale cap/fault evidence.
            self._low_guard()
            return {'contact_possible': True, 'contact_id': record['contact_id'], 'pwm_limit': self.LOW}

    def defer_holding(self, owner_token):
        """Keep live ownership and low power across body motion/job-result delivery.

        'holding' means a possible held object, not a verified grasp. Does not
        command the jaw, clear stop, prove a safe body path, or finish the session.
        """
        with self._contact_operation():
            record = self._owned_contact(owner_token, states=('active', 'holding'))
            if record['phase'] not in ('contact_possible', 'holding'):
                raise RecoveryRequired('Cannot defer holding during release/stop')
            samples = self._wait_contact_stationary()
            goal = samples[-1]['goal']; position = samples[-1]['position']
            self._phase(record, 'holding', holding_goal=goal, holding_position=position,
                        holding_stationary_samples=samples,
                        holding_stationary_duration=samples[-1]['monotonic']-samples[0]['monotonic'])
            self._low_guard(); self.state = 'holding'
            return {'completed': False, 'holding': True, 'contact_possible': True,
                    'grasp_confirmed': False, 'pwm_limit': self.LOW}

    def _wait_contact_stationary(self):
        """Live encoder stationarity; NOT quiet-goal, contact or force proof.

        A contact-loaded position servo may keep Moving66 set because its goal
        is beyond the measured jaw position. Require a constant physical goal,
        >=0.35s of fresh readings with <=2 ticks position spread instead. Preserve
        raw moving values as diagnostics, accepting only defined 0/1 values.
        """
        deadline = self.clock()+self.settle_timeout; samples = []; goal = None; previous_time = None
        for _ in range(int(self.settle_timeout/self.settle_interval)+2):
            self._low_guard(); observed_goal = self._read(42); position = self._position()
            moving = self._read(66, 1); stamp = self.clock()
            if (not self.bounds[0] <= observed_goal <= self.bounds[1] or moving not in (0, 1) or
                    not self._finite_stamp(stamp) or previous_time is not None and stamp <= previous_time):
                raise SessionFailure('Invalid or nonfresh holding telemetry')
            previous_time = stamp
            if goal is not None and observed_goal != goal:
                raise SessionFailure('Physical gripper goal changed during holding verification')
            goal = observed_goal
            samples.append({'monotonic': stamp, 'goal': goal, 'position': position, 'moving': moving})
            if max(s['position'] for s in samples)-min(s['position'] for s in samples) > 2:
                samples = samples[-1:]
            if len(samples) >= self.settle_samples and stamp-samples[0]['monotonic'] >= .35:
                return samples
            if stamp >= deadline: break
            self.sleep(self.settle_interval)
        raise SessionFailure('Gripper position was not measured stationary for holding')

    def stop_contact(self, owner_token):
        """Explicit healthy stop: remove remaining closing demand, NEVER restore.

        If fault/transport validity is unknown, use leave_pending() with no I/O.
        The new goal can only move toward opening relative to both the previous
        command and measured position. Failed ACKs leave a durable contact barrier.
        """
        with self._contact_operation():
            self.stop_latch()
            record = self._record()
            if (record is not None and record.get('contact_possible') and
                    record['phase'] == 'contact_stopped' and record.get('stop_goal_verified') is True and
                    self.state == 'blocked' and owner_token is not None and
                    owner_token == self.token == record['session_id']):
                return {'stop_requested': True, 'already_stopped': True, 'contact_possible': True,
                        'closing_demand_removed': True, 'stationary_verified': False}
            record = self._owned_contact(owner_token)
            self._low_guard()
            record = self._phase(record, 'contact_stopped', stop_goal_verified=False)  # durable BEFORE any goal
            self._low_guard()
            old = self._read(42); actual = self._position()
            if not self.bounds[0] <= old <= self.bounds[1]:
                raise SessionFailure('Invalid outstanding gripper goal')
            goal = max(old, actual) if self.opening_direction == 1 else min(old, actual)
            if self._read(65, 1) != 0: raise SessionFailure('Fault prohibits stopping goal')
            if goal != old:
                self._write(42, goal)
                if self._read(42) != goal: raise SessionFailure('Contact stop goal readback mismatch')
            self._low_guard()
            self._phase(record, 'contact_stopped', stop_goal=goal, stop_goal_verified=True)
            self.state = 'blocked'
            return {'stop_requested': True, 'contact_possible': True, 'pwm_limit': self.LOW,
                    'stop_goal': goal, 'closing_demand_removed': True, 'stationary_verified': False}

    def _begin_release(self, record, *, camera_generation, frame_sequence):
        if (not isinstance(camera_generation, str) or not camera_generation or len(camera_generation) > 256 or
                type(frame_sequence) is not int or frame_sequence < 0):
            raise RecoveryRequired('Release requires a current wrist camera generation/sequence barrier')
        self._low_guard()
        release = {'release_id': uuid.uuid4().hex, 'camera_generation': camera_generation,
                   'sequence_floor': frame_sequence, 'started_monotonic': self.clock()}
        self._phase(record, 'releasing', release=release)
        self._low_guard(); self.state = 'releasing'
        return {**release, 'session_id': record['session_id'], 'open_goal': self.open_position,
                'owner_token': self.token, 'contact_possible': True, 'pwm_limit': self.LOW}

    def begin_release(self, owner_token, *, camera_generation, frame_sequence):
        """Start explicit release; no jaw motion and no power increase.

        Frames/evidence must come from the trusted local wrist-camera pipeline.
        Body trajectories are blocked while releasing. Only release_goal below
        may send jaw goals; caller owns speed, collision and stop checks.
        """
        with self._contact_operation():
            record = self._owned_contact(owner_token, states=('active', 'holding'))
            return self._begin_release(record, camera_generation=camera_generation, frame_sequence=frame_sequence)

    def recover_for_release(self, *, camera_generation, frame_sequence):
        """Explicit recovery of a possible held object after crash/fault/stop.

        Does not clear stop, rearm, send a goal, or restore power. Caller must
        explicitly own an opening-only recovery job; ordinary begin/resume stay
        blocked. No old visual evidence survives this new release challenge.
        """
        with self._contact_operation():
            self.stop_latch()
            record = self._record()
            if record is None or not record.get('contact_possible'):
                raise RecoveryRequired('No pending contact session to release')
            self.token = record['session_id']
            return self._begin_release(record, camera_generation=camera_generation, frame_sequence=frame_sequence)

    def release_goal(self, owner_token, goal):
        """One caller-rate-limited opening-only step under 100; never closes.

        This is NOT a trajectory. Caller must check cancellation and validate
        geometry before each step under the same shared lock. Encoder jitter
        that would imply a closing step is rejected rather than compensated.
        """
        with self._contact_operation():
            record = self._owned_contact(owner_token, states=('releasing',))
            if record['phase'] != 'releasing' or type(goal) is not int or not self.bounds[0] <= goal <= self.bounds[1]:
                raise RecoveryRequired('Invalid release goal or release phase')
            self._low_guard(); actual = self._position(); old = self._read(42)
            if (not self.bounds[0] <= old <= self.bounds[1] or
                    self.opening_direction * (goal-actual) < 0 or
                    self.opening_direction * (goal-old) < 0 or
                    self.opening_direction * (self.open_position-goal) < 0):
                raise RecoveryRequired('Release goal would close or exceed the verified open position')
            if self._read(65, 1) != 0: raise SessionFailure('Fault prohibits release goal')
            self._write(42, goal)
            if self._read(42) != goal: raise SessionFailure('Release goal readback mismatch')
            self._low_guard()
            return {'goal': goal, 'contact_possible': True, 'pwm_limit': self.LOW}

    def verify_release_open(self, owner_token, *, allow_retreat=False):
        """Measure stable open jaw, then return challenge for a NEW wrist frame.

        This does not clear contact_possible. A current-looking image received
        before this measurement is not accepted as release evidence.
        """
        with self._contact_operation():
            record = self._owned_contact(owner_token, states=('releasing',))
            if record['phase'] != 'releasing' or not isinstance(record.get('release'), dict):
                raise RecoveryRequired('Release has not started')
            position, samples = self._wait_quiet(self.open_position, self.LOW, release=True)
            release = {**record['release'], 'stationary_monotonic': self.clock(),
                       'open_position': position, 'stationary_samples': samples}
            self._phase(record, 'release_waiting_vision', release=release)
            self.open_retreat_token = record['session_id'] if allow_retreat is True else None
            return {**release, 'session_id': record['session_id'], 'source': 'wrist',
                    'open_goal': self.open_position, 'contact_possible': True}

    def _wait_quiet(self, goal, cap, *, release=False):
        deadline = self.clock()+self.settle_timeout; previous = None; stable = 0
        for _ in range(int(self.settle_timeout/self.settle_interval)+2):
            position, quiet = self._quiet_check(goal, cap, release=release)
            stable = stable+1 if quiet and (previous is None or abs(position-previous) <= 1) else 0
            previous = position
            if stable >= self.settle_samples: return position, stable
            if self.clock() >= deadline: break
            self.sleep(self.settle_interval)
        raise SessionFailure('Open gripper was not verified stationary under the low cap')

    def confirm_released(self, owner_token, evidence):
        """Clear contact and restore ONLY after new trusted wrist-empty evidence.

        `evidence` is an internal server-produced result, never browser/model
        input passed through unchecked. Caller verifies image hash, provenance,
        content, visibility of BOTH jaws and space between them, and capture
        pipeline latency. capture_monotonic is JPEG arrival, not exact exposure:
        caller must discard buffered frames after verify_release_open(). This
        helper validates freshness/binding and live open state, not image truth.

        Required evidence: source='wrist', empty_verified=True, session_id,
        release_id, camera_generation, sequence, capture_monotonic, frame_sha256.
        No object dimension, observer camera, or force claim is used.
        """
        with self._contact_operation():
            record = self._owned_contact(owner_token, states=('releasing',))
            release = record.get('release'); now = self.clock()
            if (record['phase'] != 'release_waiting_vision' or not isinstance(release, dict) or
                    not isinstance(evidence, dict) or evidence.get('source') != 'wrist' or
                    evidence.get('empty_verified') is not True or evidence.get('session_id') != self.token or
                    evidence.get('release_id') != release.get('release_id') or
                    evidence.get('camera_generation') != release.get('camera_generation') or
                    type(evidence.get('sequence')) is not int or evidence['sequence'] <= release['sequence_floor'] or
                    not self._finite_stamp(evidence.get('capture_monotonic')) or
                    not release['stationary_monotonic'] < evidence['capture_monotonic'] <= now or
                    now-evidence['capture_monotonic'] > self.empty_evidence_max_age or
                    not isinstance(evidence.get('frame_sha256'), str) or len(evidence['frame_sha256']) != 64 or
                    any(c not in '0123456789abcdef' for c in evidence['frame_sha256'])):
                raise RecoveryRequired('Release needs fresh wrist-empty evidence bound to this measured release')
            # Re-measure after inference. No body/other jaw motion is authorized
            # during this phase. The caller may reacquire evidence if too old.
            position, samples = self._wait_quiet(self.open_position, self.LOW, release=True)
            if self.clock()-evidence['capture_monotonic'] > self.empty_evidence_max_age:
                raise RecoveryRequired('Wrist-empty evidence expired during live release verification')
            self.open_retreat_token = None
            receipt = {k: evidence[k] for k in ('source', 'empty_verified', 'session_id', 'release_id',
                       'camera_generation', 'sequence', 'capture_monotonic', 'frame_sha256')}
            receipt.update(contact_id=record['contact_id'], open_goal=self.open_position,
                           open_position=position, stationary_samples=samples,
                           stationary_monotonic=release['stationary_monotonic'])
            # A journal error here leaves contact_possible durable. Restoration
            # runs in the same exclusive lock, with no opportunity to close again.
            record = self._phase(record, 'lowered', contact_possible=False, release_receipt=receipt)
            result = self._complete(record)
            return {**result, 'released': True, 'contact_possible': False}

    def finish(self, *, cancelled=False):
        """Empty-jaw cleanup; contact_possible requires explicit release instead."""
        with self.lock:
            if cancelled: self.stop_latch()
            try:
                record = self._record()
                if record is not None and record['phase'] == 'completed':
                    self.state = 'completed'; return {'completed': True, 'already_completed': True}
                if record is None or record['session_id'] != self.token or self.state != 'active':
                    raise RecoveryRequired('finish cannot take over an interrupted session; explicit recovery required')
                return self._complete(record, cancelled=cancelled)
            except RecoveryRequired:
                self.state = 'blocked'; self.stop_latch(); raise
            except BaseException as exc:
                self._failed(exc)

    def recover(self):
        """Explicit operator-authorized recovery; never called by startup/resume."""
        with self.lock:
            self.stop_latch()  # Recovery completion does not automatically resume.
            try:
                record = self._record()
                if record is None: raise RecoveryRequired('No pending session to recover')
                if record['phase'] == 'completed': return {'completed': True, 'already_completed': True}
                self.token = record['session_id']
                return self._complete(record, recovery=True)
            except RecoveryRequired:
                self.state = 'blocked'; raise
            except BaseException as exc:
                self._failed(exc)

    def leave_pending(self):
        """Mark a body I/O/fault failure; no hardware retry or cleanup is attempted."""
        with self.lock:
            self.open_retreat_token = None
            self.state = 'blocked'
            self.stop_latch()

    def __enter__(self): self.enter(); return self
    def __exit__(self, exc_type, exc_value, traceback):
        # Contact sessions deliberately outlive the body job when holding. A
        # cancellation can remove closing demand at low cap; a generic fault
        # must not trigger additional I/O. Neither path clears contact evidence.
        with self.lock:
            try: record = self._record()
            except RecoveryRequired:
                self.leave_pending(); raise
            if record is not None and record.get('contact_possible'):
                if exc_type is not None:
                    if (issubclass(exc_type, (KeyboardInterrupt, CancelledError, SessionCancelled)) and
                            self.state in ('active', 'holding', 'releasing')):
                        self.stop_contact(self.token)
                    else: self.leave_pending()
                elif self.state != 'holding':
                    self.leave_pending()
                    raise RecoveryRequired('Contact session must explicitly defer holding or release')
                return False
        if exc_type is not None and not issubclass(exc_type, (KeyboardInterrupt, CancelledError, SessionCancelled)):
            self.leave_pending()
        else:
            self.finish(cancelled=exc_type is not None)
        return False
