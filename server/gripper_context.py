"""Measured arm pose context for an empty-jaw baseline; no force/scale model.

Four encoder ticks (~0.35 degrees per joint) is a conservative reuse boundary,
not a measured guarantee that gravity/friction remain invariant in that region.
Each telemetry sample must contain the five body encoders in its read bracket.
"""
import math

BODY = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')
TOLERANCE_TICKS = 4

def measured_body(value):
    if not isinstance(value,dict) or set(value)!=set(BODY):
        raise ValueError('Missing measured five-joint body pose')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
           or int(v)!=v or not 0<=v<=4095 for v in value.values()):
        raise ValueError('Invalid measured body encoder')
    return {n:int(value[n]) for n in BODY}

def require_matched_body(measured,reference):
    actual=measured_body(measured);reference=measured_body(reference)
    if any(abs(actual[n]-reference[n])>TOLERANCE_TICKS for n in BODY):
        raise ValueError('Body pose differs from the measured empty-jaw baseline')
    return actual

def body_context(reference):
    return {'version':1,'reference':measured_body(reference),
            'tolerance_ticks':TOLERANCE_TICKS,'evidence':'encoders-in-every-telemetry-bracket'}

def read_body_context(context):
    if not isinstance(context,dict):raise ValueError('Saved baseline has no measured body context')
    reference=measured_body(context.get('reference'))
    if context!=body_context(reference):raise ValueError('Saved body-context policy differs from the verified protocol')
    return reference
