"""Phase-C comparison: SeqOpt's pure-Python NSGA-II selection core vs the DEAP reference.

Benchmarks the NSGA-II survival selection (chunked-vectorized non-dominated sort + crowding +
selNSGA2) across a grid of population size x objective count:

  * ours (tools.selNSGA2 re-implementation) — pure-Python, DEAP-free at runtime
  * DEAP (tools.selNSGA2)                    — the reference oracle (dev/test-only dependency)

Reports correctness (survivor rank/crowding profile identical to DEAP) plus wall-clock and
peak memory, so the maintainer can make the ship-ours-vs-depend-on-DEAP call from data. The
shipped runtime never imports DEAP; it is needed only to run this script.

Usage:  python .github/scripts/seqopt_deap_comparison.py
"""
import time
import tracemalloc
import numpy as np

from aaanalysis.protein_engineering._backend.seqopt.nsga2 import (
    fast_non_dominated_sort, crowding_distance, select_nsga2)

try:
    from deap import base, creator, tools
except ImportError:                                     # pragma: no cover
    raise SystemExit("This comparison needs the dev-only 'deap' package: pip install deap")

if not hasattr(creator, "_CmpFitMax"):
    creator.create("_CmpFitMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "_CmpInd"):
    creator.create("_CmpInd", list, fitness=creator._CmpFitMax)

GRID = [(50, 2), (100, 2), (200, 2), (200, 3), (500, 3)]
REPEATS = 5


def deap_select(W, mu):
    creator._CmpFitMax.weights = (1.0,) * W.shape[1]
    inds = []
    for i, row in enumerate(W):
        ind = creator._CmpInd(row.tolist())
        ind.fitness.values = tuple(float(v) for v in row)
        ind.idx = i
        inds.append(ind)
    chosen = tools.selNSGA2(inds, mu)
    return {ind.idx for ind in chosen}


def profile(W, idxs):
    _, rank = fast_non_dominated_sort(W)
    crowd = np.zeros(len(W))
    for front in fast_non_dominated_sort(W)[0]:
        d = crowding_distance(W, front)
        for j, member in enumerate(front):
            crowd[member] = d[j]
    return sorted((int(rank[i]), round(float(crowd[i]), 9)) for i in idxs)


def timed(fn, repeats):
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    dt = (time.perf_counter() - t0) / repeats
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return dt * 1e3, peak / 1024.0          # ms, KiB


def main():
    print(f"{'n x m':>10} | {'impl':<12} | {'ms/call':>9} | {'peak KiB':>9} | correct")
    print("-" * 60)
    for n, m in GRID:
        rng = np.random.default_rng(n * 10 + m)
        W = rng.random((n, m))
        mu = n // 2
        ref = profile(W, deap_select(W, mu))
        runs = {
            "ours": lambda: select_nsga2(W, mu),
            "deap": lambda: deap_select(W, mu),
        }
        sels = {
            "ours": set(select_nsga2(W, mu)[0]),
            "deap": deap_select(W, mu),
        }
        for name, fn in runs.items():
            ms, kib = timed(fn, REPEATS)
            ok = "OK" if profile(W, sels[name]) == ref else "DIFF"
            print(f"{f'{n}x{m}':>10} | {name:<12} | {ms:9.3f} | {kib:9.1f} | {ok}")
        print("-" * 60)


if __name__ == "__main__":
    main()
