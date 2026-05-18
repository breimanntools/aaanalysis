"""
This is a script for the backend of get_dssp; picks the chain that best
matches ``df_seq[sequence]``, aligns DSSP output to it, and applies the
ss_mode / gap_handling policies.
"""
from typing import List, Tuple, Optional

from Bio.Align import PairwiseAligner

import aaanalysis.utils as ut


# I Helper Functions
def _make_aligner() -> PairwiseAligner:
    """Construct a global-alignment PairwiseAligner suitable for protein matching.

    Uses identity scoring with mild gap penalties so the alignment prefers
    matching identical residues over introducing gaps; this is sufficient for
    matching a PDB ATOM sequence to its parent UniProt sequence.
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5
    return aligner


def _identity_fraction(target_seq: str, atom_seq: str,
                       aligner: PairwiseAligner) -> float:
    """Fraction of target positions whose aligned partner is the same residue."""
    if not target_seq or not atom_seq:
        return 0.0
    alignment = aligner.align(target_seq, atom_seq)[0]
    a, b = alignment[0], alignment[1]
    matches = sum(1 for x, y in zip(a, b) if x == y and x != "-")
    return matches / len(target_seq)


# II Main Functions
def pick_best_chain_(target_seq: str,
                     chains: List[Tuple[str, str, List[str]]]
                     ) -> Optional[Tuple[str, str, List[str], float]]:
    """Choose the chain whose ATOM sequence best matches ``target_seq``.

    Parameters
    ----------
    target_seq : str
        The sequence from ``df_seq[sequence]`` for this entry.
    chains : list of (chain_id, atom_seq, atom_ss)
        Per-chain records from :func:`run_dssp_for_entry_`.

    Returns
    -------
    (chain_id, atom_seq, atom_ss, identity) or None
        ``None`` when no chains are present. ``identity`` is the fraction
        of ``target_seq`` positions that match the chain's atom sequence
        under a global alignment (in ``[0.0, 1.0]``).
    """
    if not chains:
        return None
    aligner = _make_aligner()
    best = None
    best_score = -1.0
    for chain_id, atom_seq, atom_ss in chains:
        score = _identity_fraction(target_seq, atom_seq, aligner)
        if score > best_score:
            best_score = score
            best = (chain_id, atom_seq, atom_ss)
    return (*best, best_score)


def align_chain_to_sequence_(target_seq: str,
                             atom_seq: str,
                             atom_ss: List[str]) -> List[str]:
    """Map each ``target_seq`` position to its DSSP code or ``'-'``.

    Runs a global pairwise alignment and walks the aligned columns. For every
    target position that aligns to an ATOM residue, the corresponding DSSP code
    is emitted; gaps and aligned-to-gap positions emit ``ut.STR_SS_GAP``.

    Returns
    -------
    list of str
        Length equals ``len(target_seq)``; each element is a single character.
    """
    aligner = _make_aligner()
    alignment = aligner.align(target_seq, atom_seq)[0]
    a_aln, b_aln = str(alignment[0]), str(alignment[1])
    out: List[str] = []
    atom_idx = 0
    for ta, ab in zip(a_aln, b_aln):
        if ta == "-":
            # Target has a gap here — nothing to emit (target shorter than alignment).
            if ab != "-":
                atom_idx += 1
            continue
        # Target position present.
        if ab == "-":
            out.append(ut.STR_SS_GAP)
        else:
            out.append(atom_ss[atom_idx])
            atom_idx += 1
    return out


def apply_ss_mode_(ss_list: List[str], ss_mode: str) -> List[str]:
    """Convert raw DSSP codes to ``ss3`` or ``ss8`` representation.

    For ``ss_mode='ss8'`` the raw DSSP space (' ') is rendered as ``'-'`` so a
    user never sees a literal blank in the output; the alignment gap symbol is
    preserved.
    """
    if ss_mode == ut.SS_MODE_3:
        return [ut.DICT_DSSP_3STATE.get(c, "C") if c != ut.STR_SS_GAP else ut.STR_SS_GAP
                for c in ss_list]
    # ss8: normalize DSSP's literal space to STR_SS_GAP for readability
    return [ut.STR_SS_GAP if c == " " else c for c in ss_list]


def apply_gap_handling_(ss_list: List[str], gap_handling: str) -> List[str]:
    """Either keep gap-pad characters (``'pad'``) or drop them (``'omit'``)."""
    if gap_handling == ut.GAP_OMIT:
        return [c for c in ss_list if c != ut.STR_SS_GAP]
    return ss_list


def count_mismatches_(target_seq: str, atom_seq: str) -> int:
    """Number of target positions whose aligned partner is a different residue.

    Gaps don't count as mismatches; only aligned residue pairs are compared.
    """
    aligner = _make_aligner()
    alignment = aligner.align(target_seq, atom_seq)[0]
    a_aln, b_aln = str(alignment[0]), str(alignment[1])
    return sum(1 for x, y in zip(a_aln, b_aln)
               if x != "-" and y != "-" and x != y)
