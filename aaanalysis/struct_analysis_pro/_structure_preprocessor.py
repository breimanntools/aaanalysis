"""
This is a script for the frontend of the StructurePreprocessor class.
Converts PDB-derived per-residue features (DSSP secondary structure / ASA /
dihedrals, plus raw PDB B-factor and residue depth) into the ``dict_num``
shape that ``CPP.run_num`` consumes, and produces the companion
``(df_scales, df_cat)`` metadata pair that names the D dimensions.

The class is pro-extra gated: biopython is required (already an
``[pro]`` dependency) and DSSP needs ``mkdssp`` on PATH; the residue-depth
feature additionally needs the ``msms`` binary, checked at runtime.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from ._backend.structure_preprocessor.feature_registry import (
    REGISTRY, VALID_FEATURE_KEYS, ENCODER_DSSP, ENCODER_PDB, ENCODER_PAE,
    ENCODER_DOMAINS,
    INVERSE_FORMULAS, validate_feature_keys, get_total_dims, get_dim_names,
    get_categories, get_subcategories)
from ._backend.structure_preprocessor.run_dssp_full import (
    run_dssp_full_for_entry_)
from ._backend.structure_preprocessor.align_dssp_full import (
    pick_best_chain_full_, count_mismatches_full_,
    align_chain_full_to_sequence_, apply_ss_mode_full_,
    apply_gap_handling_full_)
from ._backend.structure_preprocessor.encode_dssp import (
    encode_ss, encode_rasa, encode_dihedrals_sincos,
    encode_hbond_donor, encode_hbond_acceptor)
from ._backend.structure_preprocessor.encode_pdb import (
    load_structure, encode_bfactor, encode_depth,
    encode_plddt, encode_plddt_disorder, encode_plddt_tier,
    encode_chi1_sincos, encode_chi2_sincos,
    encode_ca_centroid_dist, encode_ca_centroid_dist_norm,
    encode_contact_count_8A, encode_contact_count_12A,
    encode_hse, encode_disulfide)
from ._backend.structure_preprocessor._extras import (
    is_msms_available, check_msms_available)
from ._backend.structure_preprocessor._file_format import (
    resolve_structure_path, resolve_pae_path)
from ._backend.structure_preprocessor._pae_io import load_pae_matrix
from ._backend.structure_preprocessor.encode_pae import (
    encode_pae_row_mean, encode_pae_row_min, encode_pae_row_max,
    encode_pae_local_mean, encode_pae_distal_mean,
    encode_pae_asymmetry, encode_pae_band_means)
from ._backend.structure_preprocessor._domain_io import (
    load_chopping, resolve_domain_path)
from ._backend.structure_preprocessor.encode_domains import (
    encode_domain_boundary, encode_domain_relative_position,
    encode_domain_size, encode_n_domains_in_protein)


# Same path-safety regex used by the existing get_dssp; reject characters
# that would allow path escape or platform-specific quoting bugs.
_SAFE_ENTRY_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

# Output column names appended by ``get_dssp``. Local constants — single-use,
# not part of the cross-module domain bundle in ut.
_COL_ASA = "asa"
_COL_PHI = "phi"
_COL_PSI = "psi"
_COL_PDB_OK = "pdb_ok"
_COL_HBOND_DONOR_OFFSET = "hbond_donor_offset"
_COL_HBOND_DONOR_ENERGY = "hbond_donor_energy"
_COL_HBOND_ACCEPTOR_OFFSET = "hbond_acceptor_offset"
_COL_HBOND_ACCEPTOR_ENERGY = "hbond_acceptor_energy"

# Encoder-supplied options that ``get_dssp`` consumes. (These are the raw
# DSSP stream kinds, distinct from the registry's user-facing feature keys —
# e.g. registry key 'rasa' / 'phi_psi_sincos' both pull from the 'asa' /
# 'phi_psi' raw streams. Registry keys 'hbond_donor' / 'hbond_acceptor'
# both pull from the 'hbonds' raw stream.)
_LIST_GET_DSSP_FEATURES = ["ss", "asa", "phi_psi", "hbonds"]
_LIST_ON_FAILURE = ["nan", "drop", "raise"]


# I Helper Functions
def _check_mkdssp_installed():
    """Raise ``RuntimeError`` if neither ``mkdssp`` nor ``dssp`` is on PATH."""
    if not (shutil.which("mkdssp") or shutil.which("dssp")):
        raise RuntimeError(
            "'mkdssp' is not installed or not in PATH. Install the DSSP "
            "suite (e.g. `conda install -c bioconda dssp` or "
            "`apt install dssp`) to use StructurePreprocessor.")


def _check_entry_is_filesystem_safe(entry):
    """Reject entries that would escape ``pdb_folder`` when joined into a path."""
    if not isinstance(entry, str) or not _SAFE_ENTRY_RE.match(entry):
        raise ValueError(
            f"'entry' ({entry!r}) should match [A-Za-z0-9_.-]+ to be safe "
            f"for filesystem lookup")


def _check_no_get_dssp_collisions(df_seq):
    """Refuse to overwrite pre-existing get_dssp output columns."""
    existing = [c for c in (ut.COL_SS, _COL_ASA, _COL_PHI, _COL_PSI,
                            _COL_HBOND_DONOR_OFFSET, _COL_HBOND_DONOR_ENERGY,
                            _COL_HBOND_ACCEPTOR_OFFSET, _COL_HBOND_ACCEPTOR_ENERGY,
                            ut.COL_DSSP_OK)
                if c in df_seq.columns]
    if existing:
        raise ValueError(
            f"'df_seq' should not already contain get_dssp output columns "
            f"(found: {existing}). Drop them before calling get_dssp.")


def _check_no_pdb_ok_collision(df_seq):
    """Refuse to overwrite a pre-existing ``pdb_ok`` column on encode_pdb."""
    if _COL_PDB_OK in df_seq.columns:
        raise ValueError(
            f"'df_seq' should not already contain a '{_COL_PDB_OK}' column. "
            f"Drop it before calling encode_pdb.")


def _check_features_list(features, name="features"):
    """Lightweight pre-check: must be a non-empty list of strings."""
    if not isinstance(features, list):
        raise ValueError(
            f"'{name}' ({type(features).__name__}) should be a list of "
            f"feature keys (str)")
    if len(features) == 0:
        raise ValueError(
            f"'{name}' (len=0) should be a non-empty list of feature keys")
    for f in features:
        if not isinstance(f, str):
            raise ValueError(
                f"'{name}' items ({f!r}) should be str")


def _check_get_dssp_features(features):
    """Validate the ``features`` arg of ``get_dssp`` (different alphabet)."""
    _check_features_list(features, name="features")
    bad = [f for f in features if f not in _LIST_GET_DSSP_FEATURES]
    if bad:
        raise ValueError(
            f"'features' ({bad}) should be a subset of "
            f"{_LIST_GET_DSSP_FEATURES}")


def _check_handle_failure(on_failure):
    """Validate the ``on_failure`` policy string."""
    ut.check_str_options(name="on_failure", val=on_failure,
                         list_str_options=_LIST_ON_FAILURE)


def _dssp_features_to_get_dssp_kinds(features):
    """Translate ``encode_dssp`` feature keys into ``get_dssp`` feature kinds."""
    kinds = set()
    for f in features:
        if f in ("ss3", "ss8"):
            kinds.add("ss")
        elif f == "rasa":
            kinds.add("asa")
        elif f == "phi_psi_sincos":
            kinds.add("phi_psi")
        elif f in ("hbond_donor", "hbond_acceptor"):
            kinds.add("hbonds")
    return sorted(kinds)


def _drop_or_raise_failed_entries(df_seq, ok_per_row, on_failure,
                                  source_label):
    """Apply the ``on_failure`` policy to a per-row OK mask."""
    n_failed = sum(1 for ok in ok_per_row if not ok)
    if n_failed == 0:
        return df_seq, ok_per_row, list(range(len(df_seq)))
    if on_failure == "raise":
        raise RuntimeError(
            f"{source_label} failed for {n_failed} of {len(df_seq)} entries; "
            f"on_failure='raise'")
    if on_failure == "drop":
        keep = [i for i, ok in enumerate(ok_per_row) if ok]
        df_kept = df_seq.iloc[keep].reset_index(drop=True)
        return df_kept, [True] * len(df_kept), keep
    # on_failure == "nan" (default): keep everything; caller fills with NaNs.
    return df_seq, ok_per_row, list(range(len(df_seq)))


def _ensure_dssp_columns(df_seq, features_get_dssp, pdb_folder, ss_mode,
                         gap_handling, verbose):
    """Run get_dssp inline if ``df_seq`` is missing the requested DSSP columns.

    Used by ``encode_dssp`` so users can either pre-compute DSSP once and
    reuse, or pass a plain ``df_seq`` and let encode run DSSP transparently.
    """
    needed_cols = []
    if "ss" in features_get_dssp:
        needed_cols.append(ut.COL_SS)
    if "asa" in features_get_dssp:
        needed_cols.append(_COL_ASA)
    if "phi_psi" in features_get_dssp:
        needed_cols.extend([_COL_PHI, _COL_PSI])
    if "hbonds" in features_get_dssp:
        needed_cols.extend([_COL_HBOND_DONOR_OFFSET, _COL_HBOND_DONOR_ENERGY,
                            _COL_HBOND_ACCEPTOR_OFFSET, _COL_HBOND_ACCEPTOR_ENERGY])
    have_all = all(c in df_seq.columns for c in needed_cols)
    if have_all:
        return df_seq
    return _run_get_dssp_internal(
        df_seq=df_seq, pdb_folder=pdb_folder,
        features=features_get_dssp, ss_mode=ss_mode,
        gap_handling=gap_handling, verbose=verbose)


def _run_get_dssp_internal(df_seq, pdb_folder, features, ss_mode,
                           gap_handling, verbose):
    """Per-entry DSSP runner shared by the public ``get_dssp`` and ``encode_dssp``.

    Returns a copy of ``df_seq`` with appended list columns
    (only the requested ones from ``features``) plus a boolean
    ``dssp_ok`` column.
    """
    _check_mkdssp_installed()
    pdb_folder = Path(pdb_folder)
    entries = df_seq[ut.COL_ENTRY].tolist()
    sequences = df_seq[ut.COL_SEQ].tolist()
    ss_per_row: List[Optional[List[str]]] = []
    asa_per_row: List[Optional[List[float]]] = []
    phi_per_row: List[Optional[List[float]]] = []
    psi_per_row: List[Optional[List[float]]] = []
    hb_d_off_per_row: List[Optional[List[float]]] = []
    hb_d_en_per_row: List[Optional[List[float]]] = []
    hb_a_off_per_row: List[Optional[List[float]]] = []
    hb_a_en_per_row: List[Optional[List[float]]] = []
    ok_per_row: List[bool] = []
    # Session-scoped temp dir for any gz decompression done by the resolver.
    # Inner per-entry tempdir copying (for DSSP intermediates) is still
    # handled by run_dssp_full_for_entry_.
    session_tmp = tempfile.TemporaryDirectory()
    session_tmp_path = Path(session_tmp.name)

    def _append_nan_row():
        """Push NaN/None onto every per-row list for a failed entry."""
        ss_per_row.append(None)
        asa_per_row.append(None)
        phi_per_row.append(None)
        psi_per_row.append(None)
        hb_d_off_per_row.append(None)
        hb_d_en_per_row.append(None)
        hb_a_off_per_row.append(None)
        hb_a_en_per_row.append(None)
        ok_per_row.append(False)

    for entry, target_seq in zip(entries, sequences):
        _check_entry_is_filesystem_safe(entry)
        pdb_path, _file_fmt = resolve_structure_path(
            folder=pdb_folder, entry=entry, temp_dir=session_tmp_path)
        if pdb_path is None:
            warnings.warn(
                f"Structure file for entry '{entry}' not found in "
                f"'{pdb_folder}' (tried .pdb/.pdb.gz/.cif/.cif.gz); "
                f"row will have dssp_ok=False",
                UserWarning)
            _append_nan_row()
            continue
        try:
            chains = run_dssp_full_for_entry_(pdb_path)
        except RuntimeError as e:
            warnings.warn(
                f"DSSP failed for entry '{entry}': {e}; "
                f"row will have dssp_ok=False",
                UserWarning)
            _append_nan_row()
            continue
        best = pick_best_chain_full_(target_seq, chains)
        if best is None:
            warnings.warn(
                f"No chains with assigned secondary structure for entry "
                f"'{entry}'; row will have dssp_ok=False",
                UserWarning)
            _append_nan_row()
            continue
        record, identity = best
        (chain_id, atom_seq, atom_ss, atom_asa, atom_phi, atom_psi,
         atom_hb_d_off, atom_hb_d_en,
         atom_hb_a_off, atom_hb_a_en) = record
        n_mismatch = count_mismatches_full_(target_seq, atom_seq)
        if n_mismatch > 0:
            warnings.warn(
                f"Entry '{entry}': best-matching chain '{chain_id}' has "
                f"{n_mismatch} residue mismatch(es) against df_seq[sequence] "
                f"(identity={identity:.3f}).",
                UserWarning)
        (aligned_ss, aligned_asa, aligned_phi, aligned_psi,
         aligned_hb_d_off, aligned_hb_d_en,
         aligned_hb_a_off, aligned_hb_a_en) = \
            align_chain_full_to_sequence_(
                target_seq, atom_seq, atom_ss, atom_asa, atom_phi, atom_psi,
                atom_hb_d_off, atom_hb_d_en, atom_hb_a_off, atom_hb_a_en)
        encoded_ss = apply_ss_mode_full_(aligned_ss, ss_mode)
        (final_ss, final_asa, final_phi, final_psi,
         final_hb_d_off, final_hb_d_en,
         final_hb_a_off, final_hb_a_en) = apply_gap_handling_full_(
            encoded_ss, aligned_asa, aligned_phi, aligned_psi,
            aligned_hb_d_off, aligned_hb_d_en,
            aligned_hb_a_off, aligned_hb_a_en, gap_handling)
        ss_per_row.append(final_ss)
        asa_per_row.append(final_asa)
        phi_per_row.append(final_phi)
        psi_per_row.append(final_psi)
        hb_d_off_per_row.append(final_hb_d_off)
        hb_d_en_per_row.append(final_hb_d_en)
        hb_a_off_per_row.append(final_hb_a_off)
        hb_a_en_per_row.append(final_hb_a_en)
        ok_per_row.append(True)
        if verbose:
            ut.print_out(
                f"   get_dssp: entry={entry}, chain={chain_id}, "
                f"identity={identity:.3f}, n_ss={len(final_ss)}")

    df_out = df_seq.copy()
    if "ss" in features:
        df_out[ut.COL_SS] = ss_per_row
    if "asa" in features:
        df_out[_COL_ASA] = asa_per_row
    if "phi_psi" in features:
        df_out[_COL_PHI] = phi_per_row
        df_out[_COL_PSI] = psi_per_row
    if "hbonds" in features:
        df_out[_COL_HBOND_DONOR_OFFSET] = hb_d_off_per_row
        df_out[_COL_HBOND_DONOR_ENERGY] = hb_d_en_per_row
        df_out[_COL_HBOND_ACCEPTOR_OFFSET] = hb_a_off_per_row
        df_out[_COL_HBOND_ACCEPTOR_ENERGY] = hb_a_en_per_row
    df_out[ut.COL_DSSP_OK] = ok_per_row
    session_tmp.cleanup()
    return df_out


# II Main Functions
class StructurePreprocessor:
    """Preprocess PDB-derived per-residue features for ``CPP.run_num``.

    Mirrors :class:`EmbeddingPreprocessor`'s instance-based shape but is a
    PDB-side companion: produces the ``dict_num`` tensor that
    :meth:`NumericalFeature.get_parts` slices into per-part inputs for
    :meth:`CPP.run_num`, plus the ``(df_scales, df_cat)`` metadata pair that
    names the D dimensions.

    Six public methods, each returning ONE tightly-typed value:

    1. :meth:`get_dssp` runs DSSP and appends list columns to ``df_seq``
       (``ss``, ``asa``, ``phi``, ``psi`` — only the requested ones).
       Pre-compute once and reuse, or skip and let :meth:`encode_dssp`
       run DSSP internally.
    2. :meth:`encode_dssp` returns a single ``dict_dssp`` of per-residue
       DSSP-derived numerical features (SS one-hot, rASA, dihedral
       sin/cos) — all normalized to ``[0, 1]``.
    3. :meth:`encode_pdb` returns a single ``dict_pdb`` of per-residue
       features extracted directly from the structure file (mean B-factor,
       residue depth — msms-gated; AlphaFold pLDDT / disorder mask /
       tier; chi1 / chi2 side-chain dihedrals; CA centroid distance and
       Rg-normalized variant; CA-CA contact counts at 8 Å and 12 Å).
    4. :meth:`encode_pae` returns a single ``dict_pae`` of per-residue
       summaries of the AlphaFold PAE sidecar (row-mean / row-min /
       row-max; local-vs-distal split with ±``local_window``; asymmetry;
       three-band sequence-distance means).
    5. :meth:`build_pseudo_scales` returns the per-AA-averaged
       ``df_scales`` (and optionally ``df_stds``) from a user corpus.
       Required by :meth:`CPP.run_num`'s redundancy filter to make
       ``max_cor`` meaningful (the v1 all-zero df_scales silently
       disabled that gate).
    6. :meth:`build_cat` returns the corpus-free ``df_cat`` metadata
       frame from the feature-key registry.

    Use :func:`aaanalysis.combine_dict_nums` to stitch the encoder outputs
    (and optional PLM embeddings) into a single ``dict_num`` before passing
    to :meth:`NumericalFeature.get_parts`.

    Parameters
    ----------
    verbose : bool, default=True
        If ``True``, emit progress messages via ``ut.print_out``.

    See Also
    --------
    * :class:`EmbeddingPreprocessor` for the PLM-embedding analog.
    * :class:`NumericalFeature` and :class:`CPP` for the downstream consumers.
    * :func:`aaanalysis.combine_dict_nums` for stitching multiple dict_nums.

    Notes
    -----
    * **Feature value range — always normalized to ``[0, 1]``** (NaN for
      unresolved positions). Use the table below to de-normalize back to
      raw units if needed:

      ============================  ==========================  =========================================  ====================================
      Feature key                   Raw range                   Recipe → normalized                        Inverse (de-normalize)
      ============================  ==========================  =========================================  ====================================
      ``ss3`` / ``ss8``             {0, 1} (one-hot)            identity                                   identity
      ``rasa``                      [0, ~1.2]                   ``clip(x, 0, 1)``                          identity (clipped)
      ``phi_psi_sincos``            [-1, 1]                     ``(x + 1) / 2``                            ``x * 2 - 1``  (in [-1, 1])
      ``bfactor``                   [0, 100+] Å²                ``clip(x / 100, 0, 1)``                    ``x * 100``  (lossy when ≥1)
      ``depth``                     [0, ~15] Å                  ``clip(x / 15, 0, 1)``                     ``x * 15``  (lossy when ≥1)
      ``plddt``                     [0, 100]                    ``x / 100``                                ``x * 100``
      ``plddt_disorder``            {0, 1}                      identity                                   identity
      ``plddt_tier``                {0, 1} (4-dim one-hot)      identity                                   identity
      ``chi1_sincos`` / ``chi2_sincos``  [-1, 1]                ``(x + 1) / 2``                            ``x * 2 - 1``  (in [-1, 1])
      ``ca_centroid_dist``          [0, ~40] Å                  ``clip(x / 40, 0, 1)``                     ``x * 40``  (lossy when ≥1)
      ``ca_centroid_dist_norm``     [0, ~2] (Rg units)          ``clip(x / 2, 0, 1)``                      ``x * 2``  (lossy when ≥1)
      ``contact_count_8A``          [0, ~30]                    ``clip(x / 30, 0, 1)``                     ``x * 30``  (lossy when ≥1)
      ``contact_count_12A``         [0, ~80]                    ``clip(x / 80, 0, 1)``                     ``x * 80``  (lossy when ≥1)
      ``hse``                       [0, ~30]                    ``clip(x / 30, 0, 1)``                     ``x * 30``  (lossy when ≥1)
      ``pae_row_*`` / ``pae_local_mean`` / ``pae_distal_mean`` / ``pae_band_means``
                                    [0, 31.75] Å                ``clip(x / 31.75, 0, 1)``                  ``x * 31.75``
      ``pae_asymmetry``             [0, ~10] Å                  ``clip(x / 10, 0, 1)``                     ``x * 10``  (lossy when ≥1)
      ============================  ==========================  =========================================  ====================================

      The recipes are the source of truth in
      ``feature_registry.NORMALIZATION_RECIPES``; this table is generated
      to match.

    * **Feature categorization.** Every feature key emits
      ``category='Structure'`` (the top-level redundancy / color bucket;
      see ``ut.DICT_COLOR_CAT['Structure']`` = ``#2E6E5E`` deep teal-green).
      The fine-grained split (``Secondary structure (3-state)``,
      ``B-factor (CA mean)``, ``AlphaFold pLDDT (raw)``, etc.) lives in
      ``subcategory`` and is what ``CPPPlot.feature_map`` displays on the
      y-axis. Subcategory names follow the AAontology convention
      (descriptive name with source / detail in parentheses). The redundancy filter's
      ``check_cat=True`` arm therefore groups all Structure features into
      one bucket; ``build_pseudo_scales`` populates ``df_scales`` so the
      ``max_cor`` gate can discriminate within that bucket.
    * Requires ``aaanalysis[pro]`` (biopython) plus a ``mkdssp`` / ``dssp``
      binary on PATH. The ``depth`` feature additionally requires the
      ``msms`` binary; install via ``conda install -c bioconda msms``.
    * Single-chain PDBs only — the chain whose ATOM sequence best matches
      ``df_seq[sequence]`` is selected automatically.
    """

    def __init__(self, verbose: bool = True):
        self._verbose = ut.check_verbose(verbose)

    # ------------------------------------------------------------------
    # get_dssp
    # ------------------------------------------------------------------
    def get_dssp(
        self,
        df_seq: pd.DataFrame = None,
        pdb_folder: Union[str, Path] = None,
        features: List[str] = None,
        ss_mode: str = "ss3",
        gap_handling: str = "pad",
        verbose: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Run DSSP and append per-residue list columns to ``df_seq``.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein
            identifiers and a ``sequence`` column with full protein sequences.
            ``entry`` is used as the PDB-file basename (``<entry>.pdb``) and
            ``sequence`` is the target sequence to which DSSP output is
            aligned.
        pdb_folder : str or pathlib.Path
            Directory containing one ``<entry>.pdb`` file per row of
            ``df_seq``. Missing files emit a ``UserWarning`` and produce
            ``dssp_ok=False`` for that row.
        features : list of str, default=['ss', 'asa', 'phi_psi']
            Which DSSP feature streams to extract. Any subset of
            ``{'ss', 'asa', 'phi_psi'}``. Only the requested columns are
            appended; ``dssp_ok`` is always appended.
        ss_mode : {'ss3', 'ss8'}, default='ss3'
            Secondary-structure encoding for the ``ss`` column.
        gap_handling : {'pad', 'omit'}, default='pad'
            How to handle positions without DSSP coverage. ``'pad'``
            preserves length-alignment to ``df_seq[sequence]`` and fills
            with ``ut.STR_SS_GAP`` / NaN; ``'omit'`` drops them across all
            requested streams simultaneously.
        verbose : bool, optional
            Override instance verbosity for this call only.

        Returns
        -------
        df_out : pd.DataFrame
            A copy of ``df_seq`` with appended list columns for each
            requested feature stream plus a boolean ``dssp_ok`` column.

        Raises
        ------
        RuntimeError
            If ``mkdssp`` / ``dssp`` is not on PATH.
        ValueError
            On invalid arguments or pre-existing output columns in ``df_seq``.

        Examples
        --------
        .. include:: examples/structure_preprocessor_get_dssp.rst
        """
        # Validate
        verbose = ut.check_verbose(self._verbose if verbose is None else verbose)
        ut.check_df_seq(df_seq=df_seq)
        if ut.COL_SEQ not in df_seq.columns:
            raise ValueError(
                f"'df_seq' should contain a '{ut.COL_SEQ}' column for "
                f"get_dssp")
        _check_no_get_dssp_collisions(df_seq)
        if pdb_folder is None:
            raise ValueError("'pdb_folder' should not be None")
        ut.check_folder_path_exists(folder_path=str(pdb_folder),
                                    name="pdb_folder")
        if features is None:
            features = list(_LIST_GET_DSSP_FEATURES)
        _check_get_dssp_features(features)
        ut.check_str_options(name="ss_mode", val=ss_mode,
                             list_str_options=ut.LIST_SS_MODES)
        ut.check_str_options(name="gap_handling", val=gap_handling,
                             list_str_options=ut.LIST_GAP_HANDLING)
        ut.check_bool(name="verbose", val=verbose)
        # Run
        return _run_get_dssp_internal(
            df_seq=df_seq, pdb_folder=pdb_folder, features=features,
            ss_mode=ss_mode, gap_handling=gap_handling, verbose=verbose)

    # ------------------------------------------------------------------
    # encode_dssp
    # ------------------------------------------------------------------
    def encode_dssp(
        self,
        df_seq: pd.DataFrame = None,
        pdb_folder: Union[str, Path] = None,
        features: List[str] = None,
        ss_mode: str = "ss3",
        gap_handling: str = "pad",
        on_failure: str = "nan",
        verbose: Optional[bool] = None,
    ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """Run DSSP + per-feature encoders → ``dict_dssp`` (normalized to ``[0, 1]``).

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein
            identifiers and a ``sequence`` column with full protein sequences.
            Pre-computed DSSP columns (from :meth:`get_dssp`) are reused if
            present; otherwise DSSP is run inline.
        pdb_folder : str or pathlib.Path
            Directory containing one ``<entry>.pdb`` file per row. Required
            when ``df_seq`` does not already carry the necessary DSSP
            columns.
        features : list of str
            Feature keys from the StructurePreprocessor registry that belong
            to ``encode_dssp``: any subset of
            ``{'ss3', 'ss8', 'rasa', 'phi_psi_sincos'}``. Each key's output is
            normalized to ``[0, 1]`` per the table in the class docstring.
        ss_mode : {'ss3', 'ss8'}, default='ss3'
            Forwarded to :meth:`get_dssp` when DSSP is run inline. The chosen
            SS feature key (``'ss3'`` / ``'ss8'``) drives the actual one-hot
            dimensionality independently of this option.
        gap_handling : {'pad', 'omit'}, default='pad'
            Forwarded to :meth:`get_dssp` when DSSP is run inline.
        on_failure : {'nan', 'drop', 'raise'}, default='nan'
            What to do for entries whose DSSP run failed. ``'nan'`` fills
            with NaN-only tensors; ``'drop'`` removes failed entries from
            the output dict; ``'raise'`` raises ``RuntimeError`` if any
            entry failed.
        verbose : bool, optional
            Override instance verbosity for this call only.

        Returns
        -------
        dict_dssp : dict[str, np.ndarray]
            ``{entry: (L_entry, D_total) ndarray}`` per-residue DSSP
            features concatenated in the order of ``features``. Values are in
            ``[0, 1]`` (NaN for unresolved positions).
        df_seq_out : pd.DataFrame
            Echo of the (possibly DSSP-augmented) ``df_seq`` plus an
            ``encode_dssp_ok`` column flagging per-row success. Rows are
            dropped when ``on_failure='drop'``.

        Raises
        ------
        ValueError
            On invalid arguments or feature keys not in this method's
            registry slice.
        RuntimeError
            If ``mkdssp`` is unavailable, or if any entry failed under
            ``on_failure='raise'``.
        """
        verbose = ut.check_verbose(self._verbose if verbose is None else verbose)
        ut.check_df_seq(df_seq=df_seq)
        if ut.COL_SEQ not in df_seq.columns:
            raise ValueError(
                f"'df_seq' should contain a '{ut.COL_SEQ}' column for "
                f"encode_dssp")
        validate_feature_keys(features, allowed_method=ENCODER_DSSP)
        ut.check_str_options(name="ss_mode", val=ss_mode,
                             list_str_options=ut.LIST_SS_MODES)
        ut.check_str_options(name="gap_handling", val=gap_handling,
                             list_str_options=ut.LIST_GAP_HANDLING)
        _check_handle_failure(on_failure)
        ut.check_bool(name="verbose", val=verbose)
        get_dssp_kinds = _dssp_features_to_get_dssp_kinds(features)
        if pdb_folder is None:
            needed = []
            if "ss" in get_dssp_kinds:
                needed.append(ut.COL_SS)
            if "asa" in get_dssp_kinds:
                needed.append(_COL_ASA)
            if "phi_psi" in get_dssp_kinds:
                needed.extend([_COL_PHI, _COL_PSI])
            if "hbonds" in get_dssp_kinds:
                needed.extend([_COL_HBOND_DONOR_OFFSET, _COL_HBOND_DONOR_ENERGY,
                               _COL_HBOND_ACCEPTOR_OFFSET, _COL_HBOND_ACCEPTOR_ENERGY])
            missing = [c for c in needed if c not in df_seq.columns]
            if missing:
                raise ValueError(
                    f"'pdb_folder' (None) should be a directory path when "
                    f"'df_seq' lacks DSSP columns "
                    f"(missing: {missing})")
        else:
            ut.check_folder_path_exists(folder_path=str(pdb_folder),
                                        name="pdb_folder")
        # Ensure DSSP columns are present
        df_aug = _ensure_dssp_columns(
            df_seq=df_seq, features_get_dssp=get_dssp_kinds,
            pdb_folder=pdb_folder, ss_mode=ss_mode,
            gap_handling=gap_handling, verbose=verbose)
        # Apply on_failure policy
        if ut.COL_DSSP_OK in df_aug.columns:
            ok = df_aug[ut.COL_DSSP_OK].tolist()
        else:
            ok = [True] * len(df_aug)
        df_aug, ok, _ = _drop_or_raise_failed_entries(
            df_seq=df_aug, ok_per_row=ok, on_failure=on_failure,
            source_label="DSSP")
        # Encode per entry
        entries = df_aug[ut.COL_ENTRY].tolist()
        sequences = df_aug[ut.COL_SEQ].tolist()
        ss_col = df_aug[ut.COL_SS].tolist() if ut.COL_SS in df_aug.columns else None
        asa_col = df_aug[_COL_ASA].tolist() if _COL_ASA in df_aug.columns else None
        phi_col = df_aug[_COL_PHI].tolist() if _COL_PHI in df_aug.columns else None
        psi_col = df_aug[_COL_PSI].tolist() if _COL_PSI in df_aug.columns else None
        hb_d_off_col = (df_aug[_COL_HBOND_DONOR_OFFSET].tolist()
                        if _COL_HBOND_DONOR_OFFSET in df_aug.columns else None)
        hb_d_en_col = (df_aug[_COL_HBOND_DONOR_ENERGY].tolist()
                       if _COL_HBOND_DONOR_ENERGY in df_aug.columns else None)
        hb_a_off_col = (df_aug[_COL_HBOND_ACCEPTOR_OFFSET].tolist()
                        if _COL_HBOND_ACCEPTOR_OFFSET in df_aug.columns else None)
        hb_a_en_col = (df_aug[_COL_HBOND_ACCEPTOR_ENERGY].tolist()
                       if _COL_HBOND_ACCEPTOR_ENERGY in df_aug.columns else None)
        D_total = get_total_dims(features)
        dict_dssp: Dict[str, np.ndarray] = {}
        for i, (entry, seq, is_ok) in enumerate(zip(entries, sequences, ok)):
            L = len(seq)
            if not is_ok:
                dict_dssp[entry] = np.full((L, D_total), np.nan,
                                           dtype=np.float64)
                continue
            blocks: List[np.ndarray] = []
            for key in features:
                if key in ("ss3", "ss8"):
                    if ss_col is None:
                        raise RuntimeError(
                            f"Internal: 'ss' column missing for feature "
                            f"key {key!r}")
                    blocks.append(encode_ss(
                        ss_list=ss_col[i], feature_key=key))
                elif key == "rasa":
                    if asa_col is None:
                        raise RuntimeError(
                            f"Internal: 'asa' column missing for feature "
                            f"key {key!r}")
                    blocks.append(encode_rasa(
                        asa_list=asa_col[i], sequence=seq))
                elif key == "phi_psi_sincos":
                    if phi_col is None or psi_col is None:
                        raise RuntimeError(
                            f"Internal: 'phi'/'psi' columns missing for "
                            f"feature key {key!r}")
                    blocks.append(encode_dihedrals_sincos(
                        phi_list=phi_col[i], psi_list=psi_col[i]))
                elif key == "hbond_donor":
                    if hb_d_off_col is None or hb_d_en_col is None:
                        raise RuntimeError(
                            f"Internal: H-bond donor columns missing for "
                            f"feature key {key!r}")
                    blocks.append(encode_hbond_donor(
                        offset_list=hb_d_off_col[i],
                        energy_list=hb_d_en_col[i]))
                elif key == "hbond_acceptor":
                    if hb_a_off_col is None or hb_a_en_col is None:
                        raise RuntimeError(
                            f"Internal: H-bond acceptor columns missing for "
                            f"feature key {key!r}")
                    blocks.append(encode_hbond_acceptor(
                        offset_list=hb_a_off_col[i],
                        energy_list=hb_a_en_col[i]))
                else:
                    raise RuntimeError(
                        f"Internal: feature key {key!r} not in encode_dssp "
                        f"alphabet")
            dict_dssp[entry] = np.concatenate(blocks, axis=1) \
                if len(blocks) > 1 else blocks[0]
        df_out = df_aug.copy()
        df_out["encode_dssp_ok"] = list(ok)
        return dict_dssp, df_out

    # ------------------------------------------------------------------
    # encode_pdb
    # ------------------------------------------------------------------
    def encode_pdb(
        self,
        df_seq: pd.DataFrame = None,
        pdb_folder: Union[str, Path] = None,
        features: List[str] = None,
        plddt_disorder_threshold: float = 70.0,
        on_failure: str = "nan",
        verbose: Optional[bool] = None,
    ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """Extract per-residue features from PDB ATOM records → ``dict_pdb``.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein
            identifiers and a ``sequence`` column with full protein sequences.
            ``entry`` is the PDB-file basename; ``sequence`` is the target
            sequence used for chain selection and alignment.
        pdb_folder : str or pathlib.Path
            Directory containing one ``<entry>.pdb`` file per row of
            ``df_seq``.
        features : list of str
            Feature keys from the StructurePreprocessor registry that belong
            to ``encode_pdb``: any subset of ``{bfactor, depth}``. The
            ``depth`` feature requires the external ``msms`` binary on PATH;
            absence raises ``RuntimeError`` with an install hint.
        on_failure : {'nan', 'drop', 'raise'}, default='nan'
            Failure policy for entries whose PDB load fails (missing file,
            unparseable structure, no matched chain). ``'nan'`` fills with
            NaN-only tensors; ``'drop'`` removes those entries; ``'raise'``
            re-raises.
        verbose : bool, optional
            Override instance verbosity for this call only.

        Returns
        -------
        dict_pdb : dict[str, np.ndarray]
            ``{entry: (L_entry, D_total) ndarray}`` per-residue PDB
            features concatenated in the order of ``features``.
        df_seq_out : pd.DataFrame
            Echo of ``df_seq`` plus a boolean ``pdb_ok`` column.

        Raises
        ------
        ValueError
            On invalid arguments.
        RuntimeError
            If ``msms`` is not installed and ``'depth'`` is requested, or if
            any entry failed under ``on_failure='raise'``.
        """
        verbose = ut.check_verbose(self._verbose if verbose is None else verbose)
        ut.check_df_seq(df_seq=df_seq)
        if ut.COL_SEQ not in df_seq.columns:
            raise ValueError(
                f"'df_seq' should contain a '{ut.COL_SEQ}' column for "
                f"encode_pdb")
        _check_no_pdb_ok_collision(df_seq)
        if pdb_folder is None:
            raise ValueError("'pdb_folder' should not be None")
        ut.check_folder_path_exists(folder_path=str(pdb_folder),
                                    name="pdb_folder")
        validate_feature_keys(features, allowed_method=ENCODER_PDB)
        _check_handle_failure(on_failure)
        ut.check_bool(name="verbose", val=verbose)
        ut.check_number_range(name="plddt_disorder_threshold",
                              val=plddt_disorder_threshold,
                              min_val=0.0, max_val=100.0, just_int=False)
        if "depth" in features:
            check_msms_available()
        pdb_folder = Path(pdb_folder)
        entries = df_seq[ut.COL_ENTRY].tolist()
        sequences = df_seq[ut.COL_SEQ].tolist()
        D_total = get_total_dims(features)
        dict_pdb: Dict[str, np.ndarray] = {}
        ok_per_row: List[bool] = []
        # Session-scoped temp dir for any gz decompression done by the
        # file-format resolver (.pdb.gz / .cif.gz). Cleaned at function end.
        session_tmp = tempfile.TemporaryDirectory()
        session_tmp_path = Path(session_tmp.name)
        for entry, seq in zip(entries, sequences):
            _check_entry_is_filesystem_safe(entry)
            L = len(seq)
            pdb_path, _file_fmt = resolve_structure_path(
                folder=pdb_folder, entry=entry, temp_dir=session_tmp_path)
            if pdb_path is None:
                warnings.warn(
                    f"Structure file for entry '{entry}' not found in "
                    f"'{pdb_folder}' (tried .pdb/.pdb.gz/.cif/.cif.gz); "
                    f"row will have pdb_ok=False",
                    UserWarning)
                dict_pdb[entry] = np.full((L, D_total), np.nan,
                                          dtype=np.float64)
                ok_per_row.append(False)
                continue
            try:
                structure = load_structure(pdb_path)
            except Exception as e:
                warnings.warn(
                    f"PDB parse failed for entry '{entry}': {e}; "
                    f"row will have pdb_ok=False",
                    UserWarning)
                dict_pdb[entry] = np.full((L, D_total), np.nan,
                                          dtype=np.float64)
                ok_per_row.append(False)
                continue
            blocks: List[np.ndarray] = []
            entry_ok = True
            for key in features:
                try:
                    if key == "bfactor":
                        block, identity = encode_bfactor(structure, seq)
                    elif key == "depth":
                        block, identity = encode_depth(structure, seq)
                    elif key == "plddt":
                        block, identity = encode_plddt(structure, seq)
                    elif key == "plddt_disorder":
                        block, identity = encode_plddt_disorder(
                            structure, seq,
                            threshold=plddt_disorder_threshold)
                    elif key == "plddt_tier":
                        block, identity = encode_plddt_tier(structure, seq)
                    elif key == "chi1_sincos":
                        block, identity = encode_chi1_sincos(structure, seq)
                    elif key == "chi2_sincos":
                        block, identity = encode_chi2_sincos(structure, seq)
                    elif key == "ca_centroid_dist":
                        block, identity = encode_ca_centroid_dist(
                            structure, seq)
                    elif key == "ca_centroid_dist_norm":
                        block, identity = encode_ca_centroid_dist_norm(
                            structure, seq)
                    elif key == "contact_count_8A":
                        block, identity = encode_contact_count_8A(
                            structure, seq)
                    elif key == "contact_count_12A":
                        block, identity = encode_contact_count_12A(
                            structure, seq)
                    elif key == "hse":
                        block, identity = encode_hse(structure, seq)
                    elif key == "disulfide":
                        block, identity = encode_disulfide(
                            structure, seq)
                    else:
                        raise RuntimeError(
                            f"Internal: feature key {key!r} not in "
                            f"encode_pdb alphabet")
                    blocks.append(block)
                    if verbose:
                        ut.print_out(
                            f"   encode_pdb: entry={entry}, key={key}, "
                            f"identity={identity:.3f}, n_res={block.shape[0]}")
                except RuntimeError as e:
                    warnings.warn(
                        f"PDB encoder '{key}' failed for entry '{entry}': "
                        f"{e}; this row will have pdb_ok=False",
                        UserWarning)
                    entry_ok = False
                    break
            if not entry_ok:
                dict_pdb[entry] = np.full((L, D_total), np.nan,
                                          dtype=np.float64)
                ok_per_row.append(False)
                continue
            arr = np.concatenate(blocks, axis=1) if len(blocks) > 1 \
                else blocks[0]
            dict_pdb[entry] = arr
            ok_per_row.append(True)
        # Apply on_failure policy
        df_aug = df_seq.copy()
        df_aug[_COL_PDB_OK] = ok_per_row
        df_aug, ok_after, keep_idx = _drop_or_raise_failed_entries(
            df_seq=df_aug, ok_per_row=ok_per_row, on_failure=on_failure,
            source_label="encode_pdb")
        if on_failure == "drop":
            kept_entries = {entries[i] for i in keep_idx}
            dict_pdb = {k: v for k, v in dict_pdb.items() if k in kept_entries}
        session_tmp.cleanup()
        return dict_pdb, df_aug

    # ------------------------------------------------------------------
    # encode_pae
    # ------------------------------------------------------------------
    def encode_pae(
        self,
        df_seq: pd.DataFrame = None,
        pae_folder: Union[str, Path] = None,
        features: List[str] = None,
        local_window: int = 5,
        pae_band_edges: Tuple[int, int] = (5, 15),
        on_failure: str = "nan",
        verbose: Optional[bool] = None,
    ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """Load AlphaFold PAE sidecar JSONs and produce ``dict_pae``.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame with ``entry`` + ``sequence`` columns. The PAE matrix
            shape ``(L, L)`` must equal ``len(sequence)``; mismatched rows
            are treated as failures.
        pae_folder : str or pathlib.Path
            Directory containing one PAE JSON per row. The resolver tries,
            in order: ``<entry>.json``, ``<entry>.json.gz``, and the AF-DB
            canonical ``AF-<entry>-F1-predicted_aligned_error_v4.json``
            (and its ``.gz`` variant).
        features : list of str
            Feature keys belonging to ``encode_pae``: any subset of
            ``{pae_row_mean, pae_row_min, pae_row_max, pae_local_mean,
            pae_distal_mean, pae_asymmetry, pae_band_means}``. All outputs
            normalized to ``[0, 1]`` (divisor 31.75 Å for most keys, 10 Å
            for ``pae_asymmetry``).
        local_window : int, default=5
            Used by ``pae_local_mean`` / ``pae_distal_mean``. The ``±k``
            window in residue positions for the local mean (self excluded);
            distal mean takes the complement.
        pae_band_edges : tuple of (int, int), default=(5, 15)
            Used by ``pae_band_means`` only. Sequence-distance bins:
            ``(0, edges[0]]``, ``(edges[0], edges[1]]``, ``(edges[1], L]``.
        on_failure : {'nan', 'drop', 'raise'}, default='nan'
            What to do for entries whose PAE load fails (missing sidecar,
            malformed JSON, shape mismatch).
        verbose : bool, optional
            Override instance verbosity for this call only.

        Returns
        -------
        dict_pae : dict[str, np.ndarray]
            ``{entry: (L_entry, D_total) ndarray}`` per-residue PAE
            features concatenated in the order of ``features``.
        df_seq_out : pd.DataFrame
            Echo of ``df_seq`` plus a boolean ``pae_ok`` column.

        Raises
        ------
        ValueError
            On invalid arguments or feature keys not in this method's
            registry slice.
        RuntimeError
            If any entry failed under ``on_failure='raise'``.
        """
        verbose = ut.check_verbose(self._verbose if verbose is None else verbose)
        ut.check_df_seq(df_seq=df_seq)
        if ut.COL_SEQ not in df_seq.columns:
            raise ValueError(
                f"'df_seq' should contain a '{ut.COL_SEQ}' column for "
                f"encode_pae")
        if "pae_ok" in df_seq.columns:
            raise ValueError(
                f"'df_seq' should not already contain a 'pae_ok' column. "
                f"Drop it before calling encode_pae.")
        if pae_folder is None:
            raise ValueError("'pae_folder' should not be None")
        ut.check_folder_path_exists(folder_path=str(pae_folder),
                                    name="pae_folder")
        validate_feature_keys(features, allowed_method=ENCODER_PAE)
        _check_handle_failure(on_failure)
        ut.check_bool(name="verbose", val=verbose)
        ut.check_number_range(name="local_window", val=local_window,
                              min_val=0, just_int=True)
        if (not isinstance(pae_band_edges, (list, tuple))
                or len(pae_band_edges) != 2):
            raise ValueError(
                f"'pae_band_edges' ({pae_band_edges!r}) should be a "
                f"length-2 tuple of int with 0 < lo < hi")
        lo, hi = pae_band_edges
        ut.check_number_range(name="pae_band_edges[0]", val=lo,
                              min_val=1, just_int=True)
        ut.check_number_range(name="pae_band_edges[1]", val=hi,
                              min_val=2, just_int=True)
        if not lo < hi:
            raise ValueError(
                f"'pae_band_edges' ({pae_band_edges}) should satisfy "
                f"edges[0] < edges[1]")
        pae_folder = Path(pae_folder)
        entries = df_seq[ut.COL_ENTRY].tolist()
        sequences = df_seq[ut.COL_SEQ].tolist()
        D_total = get_total_dims(features)
        dict_pae: Dict[str, np.ndarray] = {}
        ok_per_row: List[bool] = []
        session_tmp = tempfile.TemporaryDirectory()
        session_tmp_path = Path(session_tmp.name)
        for entry, seq in zip(entries, sequences):
            _check_entry_is_filesystem_safe(entry)
            L = len(seq)
            pae_path = resolve_pae_path(folder=pae_folder, entry=entry,
                                        temp_dir=session_tmp_path)
            if pae_path is None:
                warnings.warn(
                    f"PAE sidecar for entry '{entry}' not found in "
                    f"'{pae_folder}' (tried .json/.json.gz/AF-DB canonical); "
                    f"row will have pae_ok=False",
                    UserWarning)
                dict_pae[entry] = np.full((L, D_total), np.nan,
                                          dtype=np.float64)
                ok_per_row.append(False)
                continue
            try:
                pae = load_pae_matrix(pae_path, expected_L=L)
            except RuntimeError as e:
                warnings.warn(
                    f"PAE load failed for entry '{entry}': {e}; "
                    f"row will have pae_ok=False",
                    UserWarning)
                dict_pae[entry] = np.full((L, D_total), np.nan,
                                          dtype=np.float64)
                ok_per_row.append(False)
                continue
            blocks: List[np.ndarray] = []
            entry_ok = True
            for key in features:
                try:
                    if key == "pae_row_mean":
                        block = encode_pae_row_mean(pae)
                    elif key == "pae_row_min":
                        block = encode_pae_row_min(pae)
                    elif key == "pae_row_max":
                        block = encode_pae_row_max(pae)
                    elif key == "pae_local_mean":
                        block = encode_pae_local_mean(
                            pae, local_window=local_window)
                    elif key == "pae_distal_mean":
                        block = encode_pae_distal_mean(
                            pae, local_window=local_window)
                    elif key == "pae_asymmetry":
                        block = encode_pae_asymmetry(pae)
                    elif key == "pae_band_means":
                        block = encode_pae_band_means(
                            pae, band_edges=tuple(pae_band_edges))
                    else:
                        raise RuntimeError(
                            f"Internal: feature key {key!r} not in "
                            f"encode_pae alphabet")
                    blocks.append(block)
                    if verbose:
                        ut.print_out(
                            f"   encode_pae: entry={entry}, key={key}, "
                            f"L={L}")
                except RuntimeError as e:
                    warnings.warn(
                        f"PAE encoder '{key}' failed for entry '{entry}': "
                        f"{e}; this row will have pae_ok=False",
                        UserWarning)
                    entry_ok = False
                    break
            if not entry_ok:
                dict_pae[entry] = np.full((L, D_total), np.nan,
                                          dtype=np.float64)
                ok_per_row.append(False)
                continue
            arr = np.concatenate(blocks, axis=1) if len(blocks) > 1 \
                else blocks[0]
            dict_pae[entry] = arr
            ok_per_row.append(True)
        df_aug = df_seq.copy()
        df_aug["pae_ok"] = ok_per_row
        df_aug, ok_after, keep_idx = _drop_or_raise_failed_entries(
            df_seq=df_aug, ok_per_row=ok_per_row, on_failure=on_failure,
            source_label="encode_pae")
        if on_failure == "drop":
            kept_entries = {entries[i] for i in keep_idx}
            dict_pae = {k: v for k, v in dict_pae.items() if k in kept_entries}
        session_tmp.cleanup()
        return dict_pae, df_aug

    # ------------------------------------------------------------------
    # encode_domains  (v1.2 commit 3)
    # ------------------------------------------------------------------
    def encode_domains(
        self,
        df_seq: pd.DataFrame = None,
        domain_folder: Union[str, Path] = None,
        features: List[str] = None,
        on_failure: str = "nan",
        verbose: Optional[bool] = None,
    ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """Read pre-computed domain segmentation files → ``dict_domains``.

        Bring-your-own-segmentation: the user pre-runs Merizo / ChainSaw /
        AFragmenter / a hand-curated domain table on their PDB files and
        saves the **chopping string** (Merizo/ChainSaw native format) to
        one file per entry in ``domain_folder``. Two file formats are
        accepted by the resolver (looked up by entry name):

          - ``<entry>.txt`` — first non-empty line is the chopping string,
            e.g. ``6-18_296-459,19-156``.
          - ``<entry>.tsv`` — Merizo/ChainSaw TSV output with a
            ``chopping`` header (first data row used).

        The chopping format: domains separated by commas, segments within
        a domain separated by underscores; segments are 1-based inclusive
        ``start-end``. Discontinuous domains supported.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein
            identifiers and a ``sequence`` column. ``len(sequence)`` is
            used as the L for each per-residue output tensor.
        domain_folder : str or pathlib.Path
            Directory containing one chopping file per row of ``df_seq``.
        features : list of str
            Feature keys belonging to ``encode_domains``: any subset of
            ``{domain_boundary, domain_relative_position, domain_size,
            n_domains_in_protein}``. All outputs normalized to ``[0, 1]``
            (NaN for residues unassigned to any domain).
        on_failure : {'nan', 'drop', 'raise'}, default='nan'
            What to do for entries whose chopping file is missing or
            unparseable. ``'nan'`` fills with NaN-only tensors;
            ``'drop'`` removes those entries; ``'raise'`` re-raises.
        verbose : bool, optional
            Override instance verbosity for this call only.

        Returns
        -------
        dict_domains : dict[str, np.ndarray]
            ``{entry: (L_entry, D_total) ndarray}`` per-residue
            domain-derived features in the order of ``features``.
        df_seq_out : pd.DataFrame
            Echo of ``df_seq`` plus a boolean ``domain_ok`` column.

        Raises
        ------
        ValueError
            On invalid arguments or feature keys not in this method's
            registry slice.
        RuntimeError
            If any entry failed under ``on_failure='raise'``.

        Notes
        -----
        - v1.2 deliberately does NOT bundle a segmentation tool runtime
          (no PyTorch, no model weights, no Merizo / ChainSaw / AFragmenter
          pinned). Keep ``aaanalysis[pro]`` lean; pre-run the tool of your
          choice, then ingest its chopping output here.
        - Merizo: https://github.com/psipred/Merizo (~2 s per 425-residue
          chain on CPU, bundled weights, pip-installable).
        - ChainSaw: https://github.com/JudeWells/Chainsaw (manual install).
        - Output chopping strings: same `chopping` column in both tools'
          TSV output, drop-in compatible.
        """
        verbose = ut.check_verbose(self._verbose if verbose is None else verbose)
        ut.check_df_seq(df_seq=df_seq)
        if ut.COL_SEQ not in df_seq.columns:
            raise ValueError(
                f"'df_seq' should contain a '{ut.COL_SEQ}' column for "
                f"encode_domains")
        if "domain_ok" in df_seq.columns:
            raise ValueError(
                f"'df_seq' should not already contain a 'domain_ok' column. "
                f"Drop it before calling encode_domains.")
        if domain_folder is None:
            raise ValueError("'domain_folder' should not be None")
        ut.check_folder_path_exists(folder_path=str(domain_folder),
                                    name="domain_folder")
        validate_feature_keys(features, allowed_method=ENCODER_DOMAINS)
        _check_handle_failure(on_failure)
        ut.check_bool(name="verbose", val=verbose)
        domain_folder = Path(domain_folder)
        entries = df_seq[ut.COL_ENTRY].tolist()
        sequences = df_seq[ut.COL_SEQ].tolist()
        D_total = get_total_dims(features)
        dict_domains: Dict[str, np.ndarray] = {}
        ok_per_row: List[bool] = []
        for entry, seq in zip(entries, sequences):
            _check_entry_is_filesystem_safe(entry)
            L = len(seq)
            dom_path = resolve_domain_path(folder=domain_folder, entry=entry)
            if dom_path is None:
                warnings.warn(
                    f"Domain file for entry '{entry}' not found in "
                    f"'{domain_folder}' (tried .txt/.tsv/.csv); "
                    f"row will have domain_ok=False",
                    UserWarning)
                dict_domains[entry] = np.full((L, D_total), np.nan,
                                              dtype=np.float64)
                ok_per_row.append(False)
                continue
            try:
                domains = load_chopping(dom_path)
            except RuntimeError as e:
                warnings.warn(
                    f"Domain load failed for entry '{entry}': {e}; "
                    f"row will have domain_ok=False",
                    UserWarning)
                dict_domains[entry] = np.full((L, D_total), np.nan,
                                              dtype=np.float64)
                ok_per_row.append(False)
                continue
            blocks: List[np.ndarray] = []
            entry_ok = True
            for key in features:
                try:
                    if key == "domain_boundary":
                        block = encode_domain_boundary(L, domains)
                    elif key == "domain_relative_position":
                        block = encode_domain_relative_position(L, domains)
                    elif key == "domain_size":
                        block = encode_domain_size(L, domains)
                    elif key == "n_domains_in_protein":
                        block = encode_n_domains_in_protein(L, domains)
                    else:
                        raise RuntimeError(
                            f"Internal: feature key {key!r} not in "
                            f"encode_domains alphabet")
                    blocks.append(block)
                    if verbose:
                        ut.print_out(
                            f"   encode_domains: entry={entry}, key={key}, "
                            f"n_domains={len(domains)}, L={L}")
                except RuntimeError as e:
                    warnings.warn(
                        f"Domain encoder '{key}' failed for entry "
                        f"'{entry}': {e}; this row will have "
                        f"domain_ok=False",
                        UserWarning)
                    entry_ok = False
                    break
            if not entry_ok:
                dict_domains[entry] = np.full((L, D_total), np.nan,
                                              dtype=np.float64)
                ok_per_row.append(False)
                continue
            arr = np.concatenate(blocks, axis=1) if len(blocks) > 1 \
                else blocks[0]
            dict_domains[entry] = arr
            ok_per_row.append(True)
        df_aug = df_seq.copy()
        df_aug["domain_ok"] = ok_per_row
        df_aug, ok_after, keep_idx = _drop_or_raise_failed_entries(
            df_seq=df_aug, ok_per_row=ok_per_row, on_failure=on_failure,
            source_label="encode_domains")
        if on_failure == "drop":
            kept_entries = {entries[i] for i in keep_idx}
            dict_domains = {k: v for k, v in dict_domains.items()
                            if k in kept_entries}
        return dict_domains, df_aug

    # ------------------------------------------------------------------
    # build_pseudo_scales + build_cat
    # ------------------------------------------------------------------
    def build_pseudo_scales(
        self,
        df_seq: pd.DataFrame = None,
        dict_num: Dict[str, np.ndarray] = None,
        features: List[str] = None,
        return_std: bool = False,
        dim_names_override: Optional[List[str]] = None,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Build ``df_scales`` by context-free per-AA averaging of the encoded corpus.

        Mirrors :meth:`EmbeddingPreprocessor.build_pseudo_scales`: for each
        canonical amino acid ``a`` and each D dimension ``d``, the pseudo-scale
        entry is the mean of ``dict_num[entry][i, d]`` over all (entry, i)
        pairs where ``df_seq[sequence][entry][i] == a``. Non-canonical residues
        are skipped; AAs absent from the corpus get NaN rows.

        This is the **dataset-dependent** step. The values feed
        :meth:`CPP.run_num`'s redundancy filter (``df_scales.corr()`` arm); a
        meaningful corpus is required to make ``max_cor`` discriminative.
        Compute pseudo-scales once on a fixed reference corpus and reuse for
        cross-dataset comparability.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein
            identifiers and a ``sequence`` column with full protein sequences.
            Used here as the source of empirical amino-acid contexts.
        dict_num : dict[str, np.ndarray]
            Combined per-residue tensors ``{entry: (L_entry, D_total)
            ndarray}`` — typically the output of
            :func:`aaanalysis.combine_dict_nums`. Every entry in ``df_seq``
            must be a key; per entry, ``L_entry == len(sequence)``;
            ``D_total`` must equal ``sum(REGISTRY[f]['num_dims'] for f in
            features)`` (i.e. the encoder outputs in feature-key order).
        features : list of str
            Feature keys from the StructurePreprocessor registry in the same
            order as the ``dict_num`` D-axis layout. Used to name the D
            dimensions of the output and to validate ``D_total``.
        return_std : bool, default=False
            If ``True``, also return per-AA standard deviations in a second
            DataFrame of the same shape. AAs occurring exactly once receive
            std=0; AAs absent from the corpus receive NaN.
        dim_names_override : list of str, optional
            Replacement names for the D columns; length must equal
            ``D_total``. ``None`` uses the registry default names.

        Returns
        -------
        df_scales : pd.DataFrame, shape (20, D_total)
            Pseudo-scale DataFrame. Rows are the 20 canonical AAs
            (``ut.LIST_CANONICAL_AA``); columns are dim names. Cells are
            context-free per-AA means of normalized encoder outputs (each in
            ``[0, 1]``); NaN where the AA is absent from the corpus.
        df_stds : pd.DataFrame, shape (20, D_total)
            Per-AA standard deviations, returned only when ``return_std=True``.

        Raises
        ------
        ValueError
            On missing ``df_seq`` / ``dict_num``, mismatched D, missing
            entries, or invalid feature keys.

        Warns
        -----
        UserWarning
            Pseudo-scales depend on the content of ``df_seq`` + ``dict_num``.
        """
        # Validate
        if df_seq is None or dict_num is None:
            raise ValueError(
                f"'df_seq' / 'dict_num' (None) should both be provided. "
                f"Pseudo-scales need a real corpus — fall back to "
                f"build_cat() if you only want the (D, 5) metadata.")
        ut.check_df_seq(df_seq=df_seq)
        if ut.COL_SEQ not in df_seq.columns:
            raise ValueError(
                f"'df_seq' should contain a '{ut.COL_SEQ}' column for "
                f"build_pseudo_scales")
        validate_feature_keys(features)
        ut.check_bool(name="return_std", val=return_std)
        D = get_total_dims(features)
        dim_names = self._resolve_dim_names(features=features, D=D,
                                            override=dim_names_override)
        # Validate dict_num shape against df_seq + D
        entries = df_seq[ut.COL_ENTRY].tolist()
        sequences = dict(zip(entries, df_seq[ut.COL_SEQ].tolist()))
        missing = [e for e in entries if e not in dict_num]
        if missing:
            preview = missing[:5] + (["..."] if len(missing) > 5 else [])
            raise ValueError(
                f"'dict_num' ({len(missing)} missing entries) should contain "
                f"every entry in 'df_seq'. Missing: {preview}")
        for entry in entries:
            arr = dict_num[entry]
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                raise ValueError(
                    f"'dict_num[{entry!r}]' should be a 2-D np.ndarray of "
                    f"shape (L, D)")
            if arr.shape[0] != len(sequences[entry]):
                raise ValueError(
                    f"'dict_num[{entry!r}].shape[0]' ({arr.shape[0]}) "
                    f"should equal len(sequence) ({len(sequences[entry])})")
            if arr.shape[1] != D:
                raise ValueError(
                    f"'dict_num[{entry!r}].shape[1]' ({arr.shape[1]}) "
                    f"should equal sum of num_dims across features ({D})")
        warnings.warn(
            "Pseudo-scales are dataset-dependent (averaged over df_seq + "
            "dict_num). For reproducible cross-dataset comparison, compute "
            "them once on a fixed reference corpus and reuse the resulting "
            "df_scales.",
            UserWarning,
            stacklevel=2,
        )
        # Build per-AA accumulators
        list_aa = list(ut.LIST_CANONICAL_AA)
        aa_to_idx = {a: i for i, a in enumerate(list_aa)}
        sums = np.zeros((len(list_aa), D), dtype=np.float64)
        sqs = np.zeros((len(list_aa), D), dtype=np.float64)
        counts = np.zeros((len(list_aa), D), dtype=np.float64)
        for entry in entries:
            arr = dict_num[entry]
            seq = sequences[entry]
            for i, a in enumerate(seq):
                if a not in aa_to_idx:
                    continue
                row = arr[i]
                mask = ~np.isnan(row)
                if not mask.any():
                    continue
                ai = aa_to_idx[a]
                sums[ai, mask] += row[mask]
                sqs[ai, mask] += row[mask] ** 2
                counts[ai, mask] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.where(counts > 0, sums / counts, np.nan)
        df_scales = pd.DataFrame(means, index=list_aa, columns=dim_names)
        if not return_std:
            return df_scales
        with np.errstate(divide="ignore", invalid="ignore"):
            # population variance: E[x^2] - E[x]^2
            mean_sq = np.where(counts > 0, sqs / counts, np.nan)
            var = np.maximum(mean_sq - means ** 2, 0.0)
            stds = np.where(counts > 0, np.sqrt(var), np.nan)
        df_stds = pd.DataFrame(stds, index=list_aa, columns=dim_names)
        return df_scales, df_stds

    def build_cat(
        self,
        features: List[str] = None,
        dim_names_override: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Build the ``df_cat`` metadata frame for ``features``.

        Pure registry lookup — corpus-free. ``df_cat[category]`` is always
        ``'Structure'`` for every StructurePreprocessor feature in v1.1;
        the per-key semantics live in ``df_cat[subcategory]`` (see registry).

        Parameters
        ----------
        features : list of str
            Feature keys from the StructurePreprocessor registry, in the
            order they appear along the D axis of the encoder outputs.
        dim_names_override : list of str, optional
            Replacement names for the D columns; length must equal the
            total dimensionality across ``features``.

        Returns
        -------
        df_cat : pd.DataFrame, shape (D_total, 5)
            One row per dimension: ``scale_id``, ``category``, ``subcategory``,
            ``scale_name``, ``scale_description``. ``category`` is the
            top-level color/redundancy-bucket bucket; ``subcategory`` carries
            the fine-grained semantic split (``'DSSP_SS_3state'``,
            ``'Flexibility_bfactor'``, etc.).
        """
        validate_feature_keys(features)
        D = get_total_dims(features)
        dim_names = self._resolve_dim_names(features=features, D=D,
                                            override=dim_names_override)
        categories = get_categories(features)
        subcategories = get_subcategories(features)
        return pd.DataFrame({
            ut.COL_SCALE_ID: dim_names,
            ut.COL_CAT: categories,
            ut.COL_SUBCAT: subcategories,
            ut.COL_SCALE_NAME: dim_names,
            ut.COL_SCALE_DES: [f"{c}/{s}" for c, s in zip(categories,
                                                          subcategories)],
        })

    @staticmethod
    def _resolve_dim_names(features, D, override):
        """Validate ``dim_names_override`` against D; fall back to registry."""
        if override is None:
            return get_dim_names(features)
        if not isinstance(override, list):
            raise ValueError(
                f"'dim_names_override' ({type(override).__name__}) "
                f"should be a list of {D} str")
        if len(override) != D:
            raise ValueError(
                f"'dim_names_override' (len={len(override)}) "
                f"should be a list of {D} str (one per output dim)")
        for n in override:
            if not isinstance(n, str):
                raise ValueError(
                    f"'dim_names_override' items ({n!r}) should be str")
        return list(override)
