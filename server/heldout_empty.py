"""One NEW empty cycle against frozen three-cycle baseline. Never fits data.

validate_heldout_empty(rows, saved_report, training_last_capture_monotonic=...)
returns a JSON-safe report. The cutoff must be retained from the original training
JSONL before any latest-file overwrite. Raw sign-magnitude PWM is decoded again,
and the frozen baseline's empirical polarity is applied, never re-estimated.
The replay verifies acquisition timestamps/ordering/camera age; it cannot prove
current wall-clock freshness or independently establish physical jaw emptiness.
"""
from collections import Counter, defaultdict
import dataclasses, hashlib, json, math
from . import contact as mod
from .gripper_context import read_body_context,require_matched_body

STATES = ('unknown', 'free', 'stalled', 'contact_candidate')
BODY = {'shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'}
EXPECTED_CONFIG = mod.Config(position_min=1456, position_max=2983, closing_sign=-1,
                             angle_bin_width=40, endpoint_margin=24)


def finite(v): return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
def integral(v): return finite(v) and float(v).is_integer()
def signed(raw, bit): return -(raw & ((1 << bit)-1)) if raw & (1 << bit) else raw


def _frozen_baseline(report):
    if report.get('fit_allowed') is not True or report.get('errors'):
        raise ValueError('Saved training report did not qualify')
    reference=read_body_context(report.get('body_context'))
    cfg = mod.Config(**report['config'])
    if cfg != EXPECTED_CONFIG:
        raise ValueError('Saved configuration does not match the verified protocol; no substitutions')
    p = report['pwm_polarity']; sign = p.get('multiply_signed_raw_by')
    if (p.get('valid') is not True or type(sign) is not int or sign not in (-1, 1) or
            not finite(p.get('agreement_fraction')) or p['agreement_fraction'] < .95):
        raise ValueError('Saved empirical PWM polarity is unqualified')
    bins = {}
    for b in report['bins']:
        if (not integral(b.get('angle_bin')) or b.get('direction') not in ('opening', 'closing') or
                b.get('speed_ticks_per_second') != 50 or b.get('pwm_limit') != 100 or
                not integral(b.get('cycles')) or b['cycles'] < 3 or
                not integral(b.get('samples')) or b['samples'] < 12):
            raise ValueError('Unsupported saved baseline bin')
        index = int(b['angle_bin'])
        if b.get('position_interval') != [1456+index*40, 1456+(index+1)*40]:
            raise ValueError('Saved bin interval differs from its configuration')
        for name in ('tracking_error_upper', 'pwm_effort_upper', 'progress_median', 'progress_low_limit'):
            if not finite(b.get(name)) or b[name] < 0: raise ValueError('Invalid saved empirical threshold')
        if not 0 <= b['progress_low_limit'] <= b['progress_median']*cfg.slow_progress_fraction+1e-8:
            raise ValueError('Saved low-progress threshold exceeds configured envelope')
        key = (index, b['direction'], float(b['speed_ticks_per_second']), float(b['pwm_limit']))
        if key in bins: raise ValueError('Duplicate saved bin')
        bins[key] = dict(b)
    if not bins: raise ValueError('No saved supported bins')
    return cfg, mod.EmptyJawBaseline(cfg, bins, reference), sign


def validate_heldout_empty(rows, saved_report, *, training_last_capture_monotonic,
                           min_evaluated_per_direction=30):
    """Pass requires protocol validity, bidirectional coverage, and zero candidates.

    A classifier-evaluated sample has a matched baseline and has cleared the
    detector's transition guards. Unknown samples can still be evaluated partial
    anomalies; their counts remain visible. This is an empty-jaw false-positive
    check, not validation of contact sensitivity, force, or successful grasp.
    """
    result = {'passed': False, 'refit_performed': False, 'true_grasp': False,
              'force_estimate': None, 'validation': 'one-held-out-empty-cycle',
              'errors': [], 'warnings': [], 'counts': {s: 0 for s in STATES},
              'counts_by_direction': {d: {s: 0 for s in STATES} for d in ('opening', 'closing')},
              'evaluated_by_direction': {'opening': 0, 'closing': 0},
              'false_positive_sample_indices': [], 'unknown_reasons': {},
              'min_evaluated_per_direction': min_evaluated_per_direction}
    errors = result['errors']
    try:
        if not finite(training_last_capture_monotonic): raise ValueError('Original training timestamp cutoff required')
        if type(min_evaluated_per_direction) is not int or min_evaluated_per_direction < 30:
            raise ValueError('At least 30 evaluated samples per direction required')
        cfg, baseline, pwm_sign = _frozen_baseline(saved_report)
        result['baseline_sha256'] = hashlib.sha256(json.dumps(saved_report, sort_keys=True,
                                                            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        result['frozen_pwm_polarity'] = pwm_sign
        result['body_context'] = dict(saved_report['body_context'])
        result['training_last_capture_monotonic'] = training_last_capture_monotonic
        meta = [r for r in rows if r.get('type') == 'metadata']; ends = [r for r in rows if r.get('type') == 'end']
        samples = [r for r in rows if r.get('type') == 'sample']
        result['sample_count'] = len(samples)
        if len(meta) != 1 or len(ends) != 1 or not samples or rows[0] is not meta[0] or rows[-1] is not ends[0]:
            raise ValueError('Exactly one leading metadata row, sample rows, and one final receipt required')
        if len(samples)+2 != len(rows): raise ValueError('Unexpected acquisition row type')
        expected = {'empty_verified': True, 'original_pwm_limit': 500, 'pwm_limit': 100,
                    'speed_ticks_per_second': 50, 'sweep_min': 1520, 'sweep_max': 1860,
                    'camera_source': 'wrist', 'feedback': False}
        for k, value in expected.items():
            if meta[0].get(k) != value: errors.append(f'metadata mismatch: {k}')
        require_matched_body(meta[0].get('body_reference'),baseline.body_reference)
        end = ends[0]
        if (end.get('complete') is not True or end.get('restore_verified') is not True or
                end.get('restored_pwm_limit') != 500 or end.get('outstanding_closing_goal') is not False):
            errors.append('Held-out acquisition or verified neutral-goal/PWM restoration incomplete')
        detector = mod.ContactCandidateDetector(cfg, baseline)
        previous = None; leg_order = []; leg_samples = defaultdict(list)
        polarity = defaultdict(list); evaluated_bins = defaultdict(Counter); unknown = Counter(); currents = set()
        brackets = []; camera_ages = []; excluded = Counter(); sampled_goal_rates = defaultdict(list)
        for index, raw in enumerate(samples):
            bad = []
            try:require_matched_body(raw.get('body_position'),baseline.body_reference)
            except ValueError as exc:bad.append(str(exc))
            fields = ['capture_start_monotonic', 'capture_end_monotonic', 'capture_monotonic', 'position', 'goal',
                      'load_raw', 'load_signed_pwm', 'current_raw', 'pwm_limit', 'speed_ticks_per_second',
                      'frame_sequence', 'frame_capture_monotonic', 'leg_elapsed_seconds', 'goal_distance_to_endpoint']
            if any(not finite(raw.get(k)) for k in fields):
                bad.append('missing/nonfinite numerical field')
            direction = raw.get('direction')
            if type(raw.get('cycle')) is not int or raw['cycle'] != 0: bad.append('expected one held-out cycle with id 0')
            if direction not in ('opening', 'closing'): bad.append('invalid direction')
            if bad:
                errors.append(f'sample {index}: '+', '.join(bad)); detector.reset()
                result['counts']['unknown'] += 1; unknown['invalid input'] += 1
                if direction in result['counts_by_direction']: result['counts_by_direction'][direction]['unknown'] += 1
                previous = None; continue
            t = raw['capture_monotonic']; t0 = raw['capture_start_monotonic']; t1 = raw['capture_end_monotonic']
            dt = t-previous['capture_monotonic'] if previous else None
            same_leg = previous is not None and direction == previous['direction']
            span = t1-t0; age = t-raw['frame_capture_monotonic']
            brackets.append(span); camera_ages.append(age)
            if t0 <= training_last_capture_monotonic: bad.append('sample overlaps/reuses original training time')
            if not (0 <= span <= .10 and t0 <= t <= t1 and abs(t-(t0+t1)/2) <= .002): bad.append('invalid snapshot bracket')
            if previous and (dt <= 0 or t0 < previous['capture_end_monotonic']): bad.append('non-increasing/overlapping telemetry')
            if same_leg and dt > cfg.max_gap_seconds: bad.append('interrupted telemetry within a leg')
            if raw.get('model_number') != 777 or type(raw.get('status')) is not int or raw['status'] != 0 or raw.get('communication_ok') is not True:
                bad.append('hardware/transport fault')
            if raw.get('camera_source') != 'wrist' or not integral(raw['frame_sequence']) or not 0 <= age <= .50:
                bad.append('missing/stale wrist frame')
            if raw['pwm_limit'] != 100 or raw['speed_ticks_per_second'] != 50: bad.append('unmatched cap/speed')
            if not all(integral(raw[k]) and 1456 <= raw[k] <= 2983 for k in ('position', 'goal')):
                bad.append('position/goal is not an in-range observed raw register')
            if not (integral(raw['load_raw']) and 0 <= raw['load_raw'] <= 2047 and
                    abs(signed(int(raw['load_raw']), 10)) <= 1000): bad.append('invalid load encoding')
            elif raw['load_signed_pwm'] != signed(int(raw['load_raw']), 10): bad.append('raw/signed PWM decode disagreement')
            if not integral(raw['current_raw']) or not 0 <= raw['current_raw'] <= 65535: bad.append('invalid current encoding')
            if raw.get('phase') not in ('cruise', 'excluded'): bad.append('missing/invalid acquisition phase')
            endpoint = 1860 if direction == 'opening' else 1520
            sign = 1 if direction == 'opening' else -1
            if abs(raw['goal_distance_to_endpoint']-abs(endpoint-raw['goal'])) > 1e-8:
                bad.append('endpoint distance does not match observed goal42')
            if same_leg:
                dq = sign*(raw['goal']-previous['goal'])
                if dq < -1: bad.append('command reversed within a directional leg')
                # Goal42 is sampled after command generation, potentially after
                # other reads/scheduling delays. Its sample-to-sample derivative
                # is diagnostic only and cannot prove wire-command velocity.
                if dt > 0: sampled_goal_rates[direction].append(dq/dt)
                if raw['leg_elapsed_seconds'] <= previous['leg_elapsed_seconds']: bad.append('leg clock not increasing')
            if bad:
                errors.append(f'sample {index}: '+', '.join(bad)); detector.reset()
                result['counts']['unknown'] += 1; result['counts_by_direction'][direction]['unknown'] += 1
                unknown['invalid protocol sample'] += 1; previous = raw; continue
            if not leg_order or leg_order[-1] != direction: leg_order.append(direction)
            leg_samples[direction].append(raw)
            if same_leg and dt <= .30:
                dp = raw['position']-previous['position']
                if abs(dp)/dt >= 10 and abs(raw['load_signed_pwm']) >= 3:
                    polarity[direction].append(1 if dp*raw['load_signed_pwm']*pwm_sign > 0 else 0)
            phase = raw['phase']
            if (raw['leg_elapsed_seconds'] < .4 or min(raw['position']-1520, 1860-raw['position']) <= 20 or
                    raw['goal_distance_to_endpoint'] <= 20):
                if phase == 'cruise': excluded['acceleration/endpoint override'] += 1
                phase = 'excluded'
            sample = mod.Sample(timestamp=t, direction=direction, goal=raw['goal'], position=raw['position'],
                load_signed_pwm=raw['load_signed_pwm']*pwm_sign, pwm_limit=raw['pwm_limit'],
                speed=raw['speed_ticks_per_second'], servo_status=raw['status'], phase=phase,
                current_raw=raw['current_raw'], current_valid=False,body_position=raw['body_position'])
            outcome = detector.update(sample, now=t)
            state = outcome['state']
            if state not in STATES or outcome.get('true_grasp') is not False or outcome.get('force_estimate') is not None:
                raise ValueError('Classifier violated its non-grasp/non-force contract')
            result['counts'][state] += 1; result['counts_by_direction'][direction][state] += 1
            if 'matched_baseline' in outcome:
                result['evaluated_by_direction'][direction] += 1
                evaluated_bins[direction][outcome['context']['angle_bin']] += 1
            if state == 'unknown': unknown[outcome['reason']] += 1
            if state == 'contact_candidate': result['false_positive_sample_indices'].append(index)
            currents.add(signed(int(raw['current_raw']), 15)); previous = raw
        if leg_order != ['opening', 'closing']: errors.append('Need one complete opening leg followed by one closing leg')
        for direction, start, end_position in [('opening', 1520, 1860), ('closing', 1860, 1520)]:
            leg = leg_samples[direction]
            if not leg or any(abs(leg[which][field]-expected_position)>20 for which, expected_position in [(0,start),(-1,end_position)] for field in ('position','goal')):
                errors.append(f'{direction}: observed travel does not cover both prescribed endpoints')
            if result['evaluated_by_direction'][direction] < min_evaluated_per_direction:
                errors.append(f'{direction}: fewer than {min_evaluated_per_direction} classifier-evaluated samples')
            agreement = sum(polarity[direction])/len(polarity[direction]) if polarity[direction] else 0
            if len(polarity[direction]) < 15 or agreement < .95:
                errors.append(f'{direction}: new motion does not support the frozen PWM polarity')
        result['polarity_check'] = {d: {'moving_samples': len(polarity[d]),
            'frozen_sign_agreement_fraction': sum(polarity[d])/len(polarity[d]) if polarity[d] else 0}
            for d in ('opening','closing')}
        if result['false_positive_sample_indices']: errors.append('Empty-jaw contact_candidate false positive observed')
        if result['counts']['stalled']: result['warnings'].append('Stall classifications occurred in the declared empty cycle; inspect them separately')
        result['unknown_reasons'] = dict(unknown); result['excluded_reasons'] = dict(excluded)
        result['evaluated_bin_coverage'] = {d: dict(v) for d,v in evaluated_bins.items()}
        result['current_evidence'] = {'raw_values': sorted(currents), 'force_validated': False}
        result['sampled_goal_rate_max_ticks_per_second'] = {d:max(v) for d,v in sampled_goal_rates.items()}
        result['speed_validation'] = 'exact nominal protocol setting; sampled goal derivatives are not command-write timing'
        result['max_snapshot_span_seconds'] = max(brackets) if brackets else None
        result['max_wrist_frame_age_seconds'] = max(camera_ages) if camera_ages else None
        result['passed'] = not errors
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        errors.append(str(exc))
    return result
