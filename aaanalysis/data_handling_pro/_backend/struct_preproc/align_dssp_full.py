"""
This is a script for the backend of the StructurePreprocessor: picks the
chain whose ATOM sequence best matches ``df_seq[sequence]``, then aligns
the full per-residue feature streams (``ss, asa, phi, psi`` + the four
H-bond fields added in v1.2: donor/acceptor partner offset + energy) from
:func:`run_dssp_full_for_entry_` onto target positions. Identity-fraction
scoring uses :class:`Bio.Align.PairwiseAligner` with a global identity
score and mild gap penalties.
"""
from typing import List, Tuple, Optional

from Bio.Align import PairwiseAligner

import aaanalysis.utils as ut


# I Helper Functions
def _make_aligner() -> PairwiseAligner:
    """Construct a global identity-scored protein aligner (same as get_dssp)."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5
    return aligner


def _identity_fraction(target_seq: str, atom_seq: str,
                       aligner: PairwiseAligner) -> float:
    if not target_seq or not atom_seq:
        return 0.0
    alignment = aligner.align(target_seq, atom_seq)[0]
    a, b = str(alignment[0]), str(alignment[1])
    matches = sum(1 for x, y in zip(a, b) if x == y and x != "-")
    return matches / len(target_seq)


# II Main Functions
ChainFull = Tuple[
    str, str, List[str], List[float], List[float], List[float],
    List[float], List[float], List[float], List[float],
]


def pick_best_chain_full_(
    target_seq: str, chains: List[ChainFull],
    aligner: Optional[PairwiseAligner] = None,
) -> Optional[Tuple[ChainFull, float]]:
    """Choose the chain whose ATOM sequence best matches ``target_seq``.

    ``aligner`` lets a caller pass one session-scoped
    :class:`Bio.Align.PairwiseAligner` so it is constructed once per
    ``get_dssp`` run instead of once per entry; the aligner is stateless
    across ``align`` calls, so reusing it is byte-identical to a fresh one.
    Defaults to building a fresh aligner when called standalone.
    """
    if not chains:
        return None
    if aligner is None:
        aligner = _make_aligner()
    best = None
    best_score = -1.0
    for record in chains:
        atom_seq = record[1]
        score = _identity_fraction(target_seq, atom_seq, aligner)
        if score > best_score:
            best_score = score
            best = record
    return (best, best_score)


def count_mismatches_full_(
    target_seq: str, atom_seq: str,
    aligner: Optional[PairwiseAligner] = None,
) -> int:
    """Number of target positions whose aligned partner is a different residue.

    ``aligner`` optionally supplies a session-scoped aligner (reused across
    entries); a fresh one is built when omitted. Output is identical either
    way since the aligner holds no per-call state.
    """
    if aligner is None:
        aligner = _make_aligner()
    alignment = aligner.align(target_seq, atom_seq)[0]
    a, b = str(alignment[0]), str(alignment[1])
    return sum(1 for x, y in zip(a, b)
               if x != "-" and y != "-" and x != y)


def align_chain_full_to_sequence_(
    target_seq: str,
    atom_seq: str,
    atom_ss: List[str],
    atom_asa: List[float],
    atom_phi: List[float],
    atom_psi: List[float],
    atom_hb_d_off: List[float],
    atom_hb_d_en: List[float],
    atom_hb_a_off: List[float],
    atom_hb_a_en: List[float],
    aligner: Optional[PairwiseAligner] = None,
) -> Tuple[List[str], List[float], List[float], List[float],
           List[float], List[float], List[float], List[float]]:
    """Map each ``target_seq`` position to its DSSP feature values.

    For positions where the alignment produces a gap on the ATOM side,
    ``ss`` becomes ``ut.STR_SS_GAP`` and all seven float streams become
    ``float('nan')``. Returns eight lists, each of length
    ``len(target_seq)``: ``(ss, asa, phi, psi, hb_donor_off, hb_donor_en,
    hb_acceptor_off, hb_acceptor_en)``.

    ``aligner`` optionally supplies a session-scoped aligner (reused across
    entries); a fresh one is built when omitted. The alignment string pair
    is identical either way.
    """
    if aligner is None:
        aligner = _make_aligner()
    alignment = aligner.align(target_seq, atom_seq)[0]
    a_aln, b_aln = str(alignment[0]), str(alignment[1])
    out_ss: List[str] = []
    out_asa: List[float] = []
    out_phi: List[float] = []
    out_psi: List[float] = []
    out_hb_d_off: List[float] = []
    out_hb_d_en: List[float] = []
    out_hb_a_off: List[float] = []
    out_hb_a_en: List[float] = []
    atom_idx = 0
    for ta, ab in zip(a_aln, b_aln):
        if ta == "-":
            if ab != "-":
                atom_idx += 1
            continue
        if ab == "-":
            out_ss.append(ut.STR_SS_GAP)
            out_asa.append(float("nan"))
            out_phi.append(float("nan"))
            out_psi.append(float("nan"))
            out_hb_d_off.append(float("nan"))
            out_hb_d_en.append(float("nan"))
            out_hb_a_off.append(float("nan"))
            out_hb_a_en.append(float("nan"))
        else:
            out_ss.append(atom_ss[atom_idx])
            out_asa.append(atom_asa[atom_idx])
            out_phi.append(atom_phi[atom_idx])
            out_psi.append(atom_psi[atom_idx])
            out_hb_d_off.append(atom_hb_d_off[atom_idx])
            out_hb_d_en.append(atom_hb_d_en[atom_idx])
            out_hb_a_off.append(atom_hb_a_off[atom_idx])
            out_hb_a_en.append(atom_hb_a_en[atom_idx])
            atom_idx += 1
    return (out_ss, out_asa, out_phi, out_psi,
            out_hb_d_off, out_hb_d_en, out_hb_a_off, out_hb_a_en)


def apply_ss_mode_full_(ss_list: List[str], ss_mode: str) -> List[str]:
    """Convert raw DSSP codes to ``ss3`` or ``ss8`` representation."""
    if ss_mode == ut.SS_MODE_3:
        return [ut.DICT_DSSP_3STATE.get(c, "C") if c != ut.STR_SS_GAP else ut.STR_SS_GAP
                for c in ss_list]
    return [ut.STR_SS_GAP if c == " " else c for c in ss_list]


def apply_gap_handling_full_(
    ss_list: List[str],
    asa_list: List[float], phi_list: List[float], psi_list: List[float],
    hb_d_off_list: List[float], hb_d_en_list: List[float],
    hb_a_off_list: List[float], hb_a_en_list: List[float],
    gap_handling: str,
) -> Tuple[List[str], List[float], List[float], List[float],
           List[float], List[float], List[float], List[float]]:
    """Either keep gap-pad positions or drop them across all eight streams.

    When ``gap_handling='omit'``, positions whose SS code is
    ``ut.STR_SS_GAP`` are dropped from all eight lists simultaneously so
    they remain length-aligned.
    """
    if gap_handling == ut.GAP_OMIT:
        keep = [i for i, c in enumerate(ss_list) if c != ut.STR_SS_GAP]
        return ([ss_list[i] for i in keep],
                [asa_list[i] for i in keep],
                [phi_list[i] for i in keep],
                [psi_list[i] for i in keep],
                [hb_d_off_list[i] for i in keep],
                [hb_d_en_list[i] for i in keep],
                [hb_a_off_list[i] for i in keep],
                [hb_a_en_list[i] for i in keep])
    return (ss_list, asa_list, phi_list, psi_list,
            hb_d_off_list, hb_d_en_list, hb_a_off_list, hb_a_en_list)
