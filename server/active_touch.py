"""Bounded engineering acquisition. Empty-jaw evidence is supplied by inspection.
No object approach, contact action, force estimate, or jaw-scale calibration.
"""
from collections import deque
import json,time
from .gripper_power import SessionCancelled
from .empty_jaw_baseline import analyze
from .gripper_context import require_matched_body

class AcquisitionStopped(SessionCancelled):pass

def collect_empty_jaw(robot,report,cancel,empty_evidence,validation_only=False):
    saved=None;training_end=None
    if validation_only:
        from .heldout_empty import _frozen_baseline
        saved=json.loads((robot.runtime/'gripper-empty-baseline.json').read_text())
        _,baseline,_=_frozen_baseline(saved)
        training=[json.loads(s) for s in (robot.runtime/'gripper-empty-latest.jsonl').read_text().splitlines()]
        training_end=max(r['capture_end_monotonic'] for r in training if r.get('type')=='sample')
        if training_end>=time.monotonic():raise RuntimeError('原始量測時間無法核對，請重新建立基線')
    # Reject a different gravity context before enabling torque or sending goals.
    if validation_only:
        robot.connect()
        require_matched_body({k:v for k,v in robot.read().items() if k!='gripper'},baseline.body_reference)
    robot.begin();origin=robot.read();n='gripper';power=robot.gripper_power
    reference={k:v for k,v in origin.items() if k!=n}
    if validation_only:require_matched_body(reference,baseline.body_reference)
    minimum,maximum,speed=1520,1860,50
    if not minimum-20<=origin[n]<=minimum+20:
        raise RuntimeError('空夾爪基線需要從已檢查的收合姿勢開始')
    robot.validate({n:minimum});robot.validate({n:maximum})
    original=robot.gripper_telemetry()
    if original['pwm_limit']!=500:raise RuntimeError('夾爪原始 PWM 上限與已驗證值不同')
    path=robot.runtime/('gripper-empty-check-latest.jsonl' if validation_only else 'gripper-empty-latest.jsonl')
    rows=[];completed=False;receipt=None;failure=None
    with open(path,'w') as log:
        def record(row):
            rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
        record({'type':'metadata','empty_verified':True,'empty_evidence':empty_evidence,
                'body_reference':reference,
                'original_pwm_limit':original['pwm_limit'],'pwm_limit':100,
                'speed_ticks_per_second':speed,'sweep_min':minimum,'sweep_max':maximum,
                'camera_source':'wrist','feedback':False,'validation_only':validation_only})
        try:
            robot.pwm_owner=power.enter()
            robot.glide({n:minimum},speed=speed)
            cycles=1 if validation_only else 3
            for cycle in range(cycles):
                for direction,target in [('opening',maximum),('closing',minimum)]:
                    if cancel.is_set():raise AcquisitionStopped('空夾爪基線已停止')
                    report(('核對新的空夾爪循環' if validation_only else f'量測空夾爪回饋 · {cycle+1} / 3')+' · '+('張開' if direction=='opening' else '收合'))
                    window=deque()
                    def observe(progress):
                        stamp=robot.camera.latest_stamp()
                        s=robot.gripper_telemetry()
                        if s['pwm_limit']!=100:raise RuntimeError('量測期間夾爪 PWM 上限已改變')
                        if s['temperature_c']>60:raise RuntimeError('夾爪溫度過高')
                        elapsed=progress['elapsed'];remaining=progress['duration']-elapsed
                        phase='cruise' if elapsed>=.4 and remaining>=.4 and min(s['position']-minimum,maximum-s['position'])>20 and abs(target-s['goal'])>20 else 'excluded'
                        row={**s,**stamp,'type':'sample','cycle':cycle,'direction':direction,
                             'phase':phase,'capture_start_monotonic':s['read_start_monotonic'],
                             'capture_end_monotonic':s['read_end_monotonic'],
                             'speed_ticks_per_second':speed,'communication_ok':True,
                             'leg_elapsed_seconds':elapsed,'goal_distance_to_endpoint':abs(target-s['goal'])}
                        record(row);window.append(row)
                        try:
                            require_matched_body(s['body_position'],reference)
                            if validation_only:require_matched_body(s['body_position'],baseline.body_reference)
                        except ValueError as exc:raise AcquisitionStopped(str(exc)) from exc
                        while len(window)>1 and s['capture_monotonic']-window[1]['capture_monotonic']>=.4:window.popleft()
                        sign=1 if direction=='opening' else -1
                        if elapsed>=.8 and s['capture_monotonic']-window[0]['capture_monotonic']>=.4:
                            first=window[0]
                            if sign*(s['goal']-first['goal'])>=15 and sign*(s['position']-first['position'])<3:
                                raise AcquisitionStopped('低輸出下夾爪沒有持續前進；停止量測並清除待執行目標')
                    robot.glide({n:target},speed=speed,observe=observe)
                    if robot.stop.wait(.35):raise AcquisitionStopped('空夾爪基線已停止')
            robot.glide({n:origin[n]},speed=speed,return_origin=True)
            completed=True
        except BaseException as exc:
            failure=exc
            # A concurrent stop does not turn an I/O or servo fault into a
            # cancellation. Preserve the exception's origin before cleanup.
            if not isinstance(exc,(SessionCancelled,KeyboardInterrupt)):power.leave_pending()
        finally:
            try:
                if power.state=='active':receipt=power.finish(cancelled=failure is not None or cancel.is_set())
                if receipt and 'neutral_goal' in receipt:
                    robot.goals[n]=receipt['neutral_goal'];robot.bias[n]=0
            except BaseException as exc:
                completed=False
                failure=exc
            finally:robot.pwm_owner=None
            record({'type':'end','complete':completed and failure is None,
                    'restored_pwm_limit':(receipt or {}).get('restored_cap'),
                    'restore_verified':bool(receipt and receipt.get('completed')),
                    'outstanding_closing_goal':False if receipt and receipt.get('completed') else None,
                    'error':str(failure) if failure else None})
    if validation_only:
        from .heldout_empty import validate_heldout_empty
        result=validate_heldout_empty(rows,saved,training_last_capture_monotonic=training_end)
    else:result=analyze(rows)
    output='gripper-empty-check.json' if validation_only else 'gripper-empty-baseline.json'
    (robot.runtime/output).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    if failure:
        robot.error=str(failure);robot.stop.set();robot.hold();raise failure
    if validation_only:
        report('新的空夾爪循環通過檢查，沒有誤判接觸' if result['passed'] else '新的空夾爪循環尚未通過檢查')
        return result
    count=len(result.get('bins',[]))
    report(f'空夾爪量測完成 · {count} 組可比較的回饋區間' if count else '空夾爪量測完成；回饋基線仍不足以判斷接觸')
    return result
