"""Leave-one-placement-out check. No hardware access.

For each placement: fit the model on the others, freeze it, fit only that board's (x, y, yaw)
on its train views, and report the error on its held-out views. This is the honest number for
a region the model has never seen.
  python kcal_loo.py                 # every committed session-2026* placement
  python kcal_loo.py SESSION ...     # explicit sessions (committed or under .runtime/kinematic-cal)
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.optimize import least_squares

import kcal_model as km
from kcal_common import ROOT


def resolve(s):
    for base in (ROOT, km.COMMITTED_SESSIONS):
        if (base / s / 'views.jsonl').exists():
            return base / s
    raise SystemExit(f'session not found: {s}')


def main():
    sessions = [resolve(s) for s in sys.argv[1:]] or sorted(
        p for p in km.COMMITTED_SESSIONS.glob('session-2026*') if not p.name.endswith('-test'))
    for out in sessions:
        model, _ = km.solve([s for s in sessions if s != out], save=False)
        views = km.load_views([out])
        train = [v for v in views if v['split'] == 'train']
        held = [v for v in views if v['split'] != 'train']
        K, D, X, offs = model['K'], model['D'], model['X'], model['offsets_deg']
        b0 = km.initial_boards(train, K, D, X, offs)
        pl = next(iter(b0))
        tilt, z = model['table_tilt'], model['table_z']
        corners = np.array([v['corners'] for v in train])

        def board(p):
            return tilt @ km.board_on_table(p[0], p[1], p[2], z)

        def objective(p):
            return (km.predict(train, K, D, X, {pl: board(p)}, offs)[0] - corners).reshape(-1)

        sol = least_squares(objective, np.array(b0[pl][:3]), loss='soft_l1')
        ev = km.evaluate(held, K, D, {**model, 'boards': {pl: board(sol.x)}})
        print(f"predict {out.name}: offsets_deg {{{', '.join(f'{k} {v:+.2f}' for k, v in offs.items())}}}  "
              f"held-out n={len(held)}  median {ev['board_corner_median_mm']} mm  max {ev['board_corner_max_mm']} mm  "
              f"rms {ev['rms_px']} px", flush=True)


if __name__ == '__main__':
    main()
