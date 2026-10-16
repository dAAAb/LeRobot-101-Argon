"""Replay recorded skill outcomes offline; never opens hardware or calls models.

Input JSON: {"plan": <SkillPlan>, "records": [{"node":"step_0",
"receipt": <skill receipt>}, ...]}. Records stop at the first terminal branch.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from server.skill_graph import replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:
        if args.input.stat().st_size>131072:raise ValueError('Replay input exceeds 128 KiB')
        value=json.loads(args.input.read_text(),parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
        if not isinstance(value,dict) or set(value)!={'plan','records'}:raise ValueError('Expected plan and records only')
        result=replay(value['plan'],value['records'])
        text=json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)
        if args.output:args.output.write_text(text+'\n')
        else:print(text)
    except (ValueError,OSError) as exc:parser.exit(1,str(exc)+'\n')


if __name__=='__main__':main()
