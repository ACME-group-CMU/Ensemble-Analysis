#!/usr/bin/env python
"""Recompute cached observables for every structure in data/all_sizes.

    python regenerate.py            # only what is missing or stale
    python regenerate.py --force    # everything
    python regenerate.py --size 24  # one cell size

Failures are collected and reported at the end rather than being swallowed.
"""
import argparse
import multiprocessing as mp
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import io, observables


def _worker(struct_id):
    try:
        observables.populate(struct_id)
        return struct_id, None
    except Exception:
        return struct_id, traceback.format_exc(limit=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--size', type=int, choices=io.SIZES)
    ap.add_argument('--jobs', type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()

    ok, bad = io.validate()
    if bad:
        print(f'{len(bad)} unreadable structure(s) — these are skipped:')
        for sid, why in sorted(bad.items()):
            print(f'  {sid}: {why}')
        print()

    ids = [s for s in ok if args.size is None or io.size_of(s) == args.size]
    todo = ids if args.force else [s for s in ids if not observables.is_current(s)]
    print(f'{len(ids)} structures, {len(todo)} to compute, {args.jobs} workers')
    if not todo:
        return

    failures = {}
    done = 0
    with mp.Pool(args.jobs) as pool:
        for struct_id, err in pool.imap_unordered(_worker, todo, chunksize=4):
            done += 1
            if err:
                failures[struct_id] = err
            if done % 50 == 0 or done == len(todo):
                print(f'  {done}/{len(todo)}  ({len(failures)} failed)', flush=True)

    if failures:
        print(f'\n{len(failures)} FAILED:')
        for sid, err in sorted(failures.items())[:10]:
            print(f'--- {sid} ---\n{err}')
        sys.exit(1)
    print('\nall structures computed')


if __name__ == '__main__':
    main()
