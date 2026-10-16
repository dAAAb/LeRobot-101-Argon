"""Small executable skill graph and pure replay, inspired by GaP's checkpoints.

This is our own bounded graph format, not a GaP runtime or trained policy. It
compiles only the existing SkillPlan and never evaluates generated code. One
physical skill runs at a time; a failed checkpoint has no motion retry edge.
"""
from __future__ import annotations

from copy import deepcopy
import math
import unicodedata
from .skill_plan import SkillPlan


def target_key(value):
    return ' '.join(unicodedata.normalize('NFKC', value).split()).casefold()


def compile_graph(plan):
    plan=SkillPlan.model_validate(plan).model_dump()
    nodes=[];last_observation=None
    for index,step in enumerate(plan['steps']):
        dependency=None
        if step['skill']=='grasp' and last_observation and target_key(step['target'])==last_observation[1]:
            dependency=last_observation[0]
        if step['skill']=='observe':
            if step['target']:last_observation=(f'step_{index}',target_key(step['target']))
        else:last_observation=None
        nodes.append({'id':f'step_{index}','index':index,'skill':step['skill'],'label':step['label'],
                      'postcondition':{'observe':'observation_returned','move':'distance_reached',
                        'wave':'wave_verified','grasp':'held_object_verified',
                        'place':'placement_verified','stack':'stack_verified',
                        'open':'empty_jaw_transaction_finished','release':'release_verified'}[step['skill']],
                      'requires_visible_from':dependency,
                      'on_verified':f'step_{index+1}' if index+1<len(plan['steps']) else 'completed',
                      'on_unconfirmed':'blocked','on_error':'blocked'})
    return {'version':1,'kind':'bounded_skill_graph','entry':'step_0' if nodes else 'unsupported',
            'nodes':nodes,'terminals':['completed','blocked','cancelled','unsupported'],
            'maximum_node_visits':len(nodes),'automatic_motion_retries':0,
            'generated_code':False}


def _number(value):
    return type(value) in (int,float) and math.isfinite(value)


def checkpoint(step,receipt):
    """Independent postcondition checks; a completed flag alone is insufficient."""
    if not isinstance(receipt,dict) or receipt.get('skill')!=step['skill'] or receipt.get('completed') is not True:
        return {'passed':False,'reason':'這一步尚未確認完成'}
    kind=step['skill'];evidence={}
    if kind=='move':
        goal=receipt.get('requested_mm');actual=receipt.get('measured_model_progress_mm')
        tolerance=max(1.5,step['distance_mm']*.15) if receipt.get('execution_mode')=='planned_segment' else .75
        upper=tolerance if receipt.get('execution_mode')=='planned_segment' else 1.5
        minimum=max(.5,step['distance_mm']*.5,step['distance_mm']-tolerance) if receipt.get('execution_mode')=='planned_segment' else step['distance_mm']-tolerance
        passed=bool(_number(goal) and _number(actual) and abs(goal-step['distance_mm'])<1e-6
                    and minimum<=actual<=goal+upper and type(receipt.get('step_count')) is int
                    and 1<=receipt['step_count']<=16)
        evidence={'requested_mm':goal,'measured_model_progress_mm':actual,
                  'model_displacement_not_external_metrology':True}
    elif kind in ('wave','grasp','place','stack'):
        key={'wave':'wave_completed','grasp':'grasp_confirmed','place':'place_confirmed','stack':'stack_confirmed'}[kind]
        passed=isinstance(receipt.get('receipt'),dict) and receipt['receipt'].get(key) is True
        if kind in ('place','stack','grasp'):
            passed=passed and receipt['receipt'].get('completed') is True
        evidence={key:passed}
    elif kind=='observe':
        value=receipt.get('observation');passed=isinstance(value,dict) and type(value.get('visible')) is bool
        evidence={'visible':value.get('visible') if isinstance(value,dict) else None,
                  'target':step['target']}
    else:
        value=receipt.get('receipt') or {}
        if not isinstance(value,dict):return {'passed':False,'reason':'夾爪回報格式不完整'}
        neutral=value.get('neutral_goal')
        transaction=(value.get('completed') is True and value.get('restored_cap')==500
                     and type(neutral) is int and 0<=neutral<=4095
                     and value.get('cancelled',False) is False and value.get('contact_possible',False) is False)
        passed=(kind=='release' and receipt.get('already_empty') is True and not value) or transaction
        evidence={'already_empty':receipt.get('already_empty') is True,
                  'transaction_completed':value.get('completed') is True,
                  'released':value.get('released') is True}
    return {'passed':bool(passed),'reason':'完成條件已核對' if passed else '回報缺少這個動作的完成證據',
            'evidence':evidence}


class SkillGraphRun:
    def __init__(self,plan):
        self.plan=SkillPlan.model_validate(plan).model_dump();self.graph=compile_graph(self.plan)
        self.active=self.graph['entry'];self.events=[]

    def node(self,index):
        if type(index) is not int or not 0<=index<len(self.graph['nodes']):raise ValueError('Unknown graph node')
        node=self.graph['nodes'][index]
        if self.active!=node['id']:raise ValueError('Graph node is not the active node')
        return node

    def verify(self,index,receipt,*,dependency=None):
        node=self.node(index);result=checkpoint(self.plan['steps'][index],receipt)
        if node['requires_visible_from'] is not None:
            if not self.dependency_passed(node,dependency):result={'passed':False,'reason':'前一步沒有確認看見這個物件，先重新觀察','evidence':{'target_visible':False}}
        self.active=node['on_verified'] if result['passed'] else node['on_unconfirmed']
        event={'node':node['id'],'outcome':'verified' if result['passed'] else 'unconfirmed',
               'checkpoint':result,'next':self.active}
        self.events.append(event);return deepcopy(event)

    def dependency_passed(self,node,dependency):
        source=node['requires_visible_from']
        if source is None:return True
        event=next((e for e in self.events if e['node']==source),None)
        return bool(event and event['outcome']=='verified' and
                    event.get('checkpoint',{}).get('evidence',{}).get('visible') is True and
                    isinstance(dependency,dict) and dependency.get('source_node')==source and
                    dependency.get('current') is True and dependency.get('visible') is True)

    def fail(self,index,reason,*,cancelled=False):
        node=self.node(index)
        self.active='cancelled' if cancelled else 'blocked'
        event={'node':node['id'],'outcome':'cancelled' if cancelled else 'error',
               'reason':str(reason)[:500],'next':self.active}
        self.events.append(event);return deepcopy(event)

    def snapshot(self):
        return {'definition':deepcopy(self.graph),'active_node':self.active,'events':deepcopy(self.events),
                'completed':self.active=='completed','replay':False}


def replay(plan,records):
    """Pure offline branch evaluation; no imports of app, Robot or model clients.

    Each record contains one node's receipt and optional visibility binding, or
    an error. No scripts, paths, arbitrary graph or live API calls are accepted.
    Replay checks order and completeness, never asserts real-world success.
    """
    run=SkillGraphRun(plan)
    if not isinstance(records,list) or len(records)>8:raise ValueError('Invalid replay record count')
    for index,record in enumerate(records):
        if not isinstance(record,dict) or set(record)-{'node','receipt','dependency','error','cancelled'}:
            raise ValueError('Invalid replay record')
        if record.get('node')!=f'step_{index}':raise ValueError('Missing, duplicate or out-of-order replay node')
        run.node(index)
        if 'error' in record:
            if set(record)-{'node','error','cancelled'} or not isinstance(record['error'],str) or len(record['error'])>500 or type(record.get('cancelled',False)) is not bool:raise ValueError('Ambiguous replay outcome')
            run.fail(index,record['error'],cancelled=record.get('cancelled',False))
        else:
            if set(record)-{'node','receipt','dependency'} or 'receipt' not in record:raise ValueError('Ambiguous replay outcome')
            run.verify(index,record['receipt'],dependency=record.get('dependency'))
    if run.active.startswith('step_'):raise ValueError('Replay is incomplete')
    result=run.snapshot();result.update(replay=True,physical_success_verified=False)
    return result
