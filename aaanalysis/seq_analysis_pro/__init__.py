"""
Pro sequence analysis: similarity, filtering, and motif scanning (``pro`` extra).

Public objects: comp_seq_sim, filter_seq, scan_motif.
Gated behind the ``pro`` extra — ``comp_seq_sim`` / ``filter_seq`` need biopython and
``scan_motif`` wraps the MEME suite (FIMO). Complements core ``seq_analysis``:
``scan_motif`` selects motif-matched windows by FIMO p-value where
``AAWindowSampler.sample_motif_matched`` uses a pure-Python PWM-sum threshold (a
genuinely different selection, not a re-scored mimic). Imported lazily and replaced by
install-hint stubs when the extra is absent.

See ``.claude/rules/pro-core-boundary.md`` for the pro/core boundary, ``CONTEXT.md``
for domain terms.
"""
from ._comp_seq_sim import comp_seq_sim
from ._filter_seq import filter_seq
from ._scan_motif import scan_motif

__all__ = [
    "comp_seq_sim",
    "filter_seq",
    "scan_motif",
]
