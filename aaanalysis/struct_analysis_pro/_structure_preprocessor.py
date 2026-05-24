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
import warnings

import numpy as np
import pandas as pd

import aaanalysis.utils as ut
from ._backend.structure_preprocessor.feature_registry import (
    REGISTRY, VALID_FEATURE_KEYS, ENCODER_DSSP, ENCODER_PDB,
    validate_feature_keys, get_total_dims, get_dim_names,
    get_categories, get_subcategories)
from ._backend.structure_preprocessor.run_dssp_full import (
    run_dssp_full_for_entry_)
from ._backend.structure_preprocessor.align_dssp_full import (
    pick_best_chain_full_, count_mismatches_full_,
    align_chain_full_to_sequence_, apply_ss_mode_full_,
    apply_gap_handling_full_)
from ._backend.structure_preprocessor.encode_dssp import (
    encode_ss, encode_asa, encode_dihedrals)
from ._backend.structure_preprocessor.encode_pdb import (
    load_structure, encode_bfactor, encode_depth)
from ._backend.structure_preprocessor._extras import (
    is_msms_available, check_msms_available)


# Same path-safety regex used by the existing get_dssp; reject characters
# that would allow path escape or platform-specific quoting bugs.
_SAFE_ENTRY_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

# Output column names appended by ``get_dssp``. Local constants — single-use,
# not part of the cross-module domain bundle in ut.
_COL_ASA = "asa"
_COL_PHI = "phi"
_COL_PSI = "psi"
_COL_PDB_OK = "pdb_ok"

# Encoder-supplied options that ``get_dssp`` consumes.
_LIST_GET_DSSP_FEATURES = ["ss", "asa", "phi_psi"]
_LIST_ASA_KINDS = ["asa", "rasa"]
_LIST_DIHEDRAL_ENCODINGS = ["raw", "sin_cos"]
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
        elif f in ("asa", "rasa"):
            kinds.add("asa")
        elif f in ("phi_psi", "phi_psi_sincos"):
            kinds.add("phi_psi")
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
    ok_per_row: List[bool] = []

    for entry, target_seq in zip(entries, sequences):
        _check_entry_is_filesystem_safe(entry)
        pdb_path = pdb_folder / f"{entry}.pdb"
        if not pdb_path.is_file():
            warnings.warn(
                f"PDB file for entry '{entry}' not found at '{pdb_path}'; "
                f"row will have dssp_ok=False",
                UserWarning)
            ss_per_row.append(None)
            asa_per_row.append(None)
            phi_per_row.append(None)
            psi_per_row.append(None)
            ok_per_row.append(False)
            continue
        try:
            chains = run_dssp_full_for_entry_(pdb_path)
        except RuntimeError as e:
            warnings.warn(
                f"DSSP failed for entry '{entry}': {e}; "
                f"row will have dssp_ok=False",
                UserWarning)
            ss_per_row.append(None)
            asa_per_row.append(None)
            phi_per_row.append(None)
            psi_per_row.append(None)
            ok_per_row.append(False)
            continue
        best = pick_best_chain_full_(target_seq, chains)
        if best is None:
            warnings.warn(
                f"No chains with assigned secondary structure for entry "
                f"'{entry}'; row will have dssp_ok=False",
                UserWarning)
            ss_per_row.append(None)
            asa_per_row.append(None)
            phi_per_row.append(None)
            psi_per_row.append(None)
            ok_per_row.append(False)
            continue
        record, identity = best
        chain_id, atom_seq, atom_ss, atom_asa, atom_phi, atom_psi = record
        n_mismatch = count_mismatches_full_(target_seq, atom_seq)
        if n_mismatch > 0:
            warnings.warn(
                f"Entry '{entry}': best-matching chain '{chain_id}' has "
                f"{n_mismatch} residue mismatch(es) against df_seq[sequence] "
                f"(identity={identity:.3f}).",
                UserWarning)
        aligned_ss, aligned_asa, aligned_phi, aligned_psi = \
            align_chain_full_to_sequence_(
                target_seq, atom_seq, atom_ss, atom_asa, atom_phi, atom_psi)
        encoded_ss = apply_ss_mode_full_(aligned_ss, ss_mode)
        final_ss, final_asa, final_phi, final_psi = apply_gap_handling_full_(
            encoded_ss, aligned_asa, aligned_phi, aligned_psi, gap_handling)
        ss_per_row.append(final_ss)
        asa_per_row.append(final_asa)
        phi_per_row.append(final_phi)
        psi_per_row.append(final_psi)
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
    df_out[ut.COL_DSSP_OK] = ok_per_row
    return df_out


# II Main Functions
class StructurePreprocessor:
    """Preprocess PDB-derived per-residue features for ``CPP.run_num``.

    Mirrors :class:`EmbeddingPreprocessor`'s instance-based shape but is a
    PDB-side companion: produces the ``dict_num`` tensor that
    :meth:`NumericalFeature.get_parts` slices into per-part inputs for
    :meth:`CPP.run_num`, plus the ``(df_scales, df_cat)`` metadata pair that
    names the D dimensions.

    Four public methods, each returning ONE tightly-typed value:

    1. :meth:`get_dssp` runs DSSP and appends list columns to ``df_seq``
       (``ss``, ``asa``, ``phi``, ``psi`` — only the requested ones).
       Pre-compute once and reuse, or skip and let :meth:`encode_dssp`
       run DSSP internally.
    2. :meth:`encode_dssp` returns a single ``dict_dssp`` of per-residue
       DSSP-derived numerical features (SS one-hot, ASA, dihedrals).
    3. :meth:`encode_pdb` returns a single ``dict_pdb`` of per-residue
       features extracted directly from PDB ATOM records (mean B-factor,
       residue depth — the latter gated on the external ``msms`` binary).
    4. :meth:`build_scales` returns the ``(df_scales, df_cat)`` metadata pair
       matching a list of feature keys, for the :class:`CPP` constructor.

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
        asa_kind: str = "rasa",
        dihedral_encoding: str = "sin_cos",
        gap_handling: str = "pad",
        on_failure: str = "nan",
        verbose: Optional[bool] = None,
    ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """Run DSSP + per-feature encoders → ``dict_dssp``.

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
            to ``encode_dssp``: any subset of ``{ss3, ss8, asa, rasa,
            phi_psi, phi_psi_sincos}``.
        ss_mode : {'ss3', 'ss8'}, default='ss3'
            Forwarded to :meth:`get_dssp` when DSSP is run inline. Ignored
            otherwise. Note: the chosen SS feature key (``'ss3'`` or
            ``'ss8'``) drives the actual one-hot dimensionality independently
            of this option.
        asa_kind : {'rasa', 'asa'}, default='rasa'
            Whether the ``asa``/``rasa`` feature key produces absolute or
            relative ASA. ``'rasa'`` divides by per-AA max ASA (Tien et al.
            2013); ``'asa'`` returns the DSSP-reported absolute value in Å².
        dihedral_encoding : {'sin_cos', 'raw'}, default='sin_cos'
            Forwarded to :func:`encode_dihedrals` when encoding ``phi_psi`` /
            ``phi_psi_sincos``. Note: the chosen feature key (``'phi_psi'``
            or ``'phi_psi_sincos'``) drives the actual dimensionality
            independently of this option.
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
            features concatenated in the order of ``features``.
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
        ut.check_str_options(name="asa_kind", val=asa_kind,
                             list_str_options=_LIST_ASA_KINDS)
        ut.check_str_options(name="dihedral_encoding", val=dihedral_encoding,
                             list_str_options=_LIST_DIHEDRAL_ENCODINGS)
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
                    ss_list = ss_col[i]
                    blocks.append(encode_ss(
                        ss_list=ss_list,
                        ss_mode=ut.SS_MODE_3 if key == "ss3" else ut.SS_MODE_8))
                elif key in ("asa", "rasa"):
                    if asa_col is None:
                        raise RuntimeError(
                            f"Internal: 'asa' column missing for feature "
                            f"key {key!r}")
                    asa_list = asa_col[i]
                    blocks.append(encode_asa(
                        asa_list=asa_list, sequence=seq,
                        kind="rasa" if key == "rasa" else "asa"))
                elif key in ("phi_psi", "phi_psi_sincos"):
                    if phi_col is None or psi_col is None:
                        raise RuntimeError(
                            f"Internal: 'phi'/'psi' columns missing for "
                            f"feature key {key!r}")
                    blocks.append(encode_dihedrals(
                        phi_list=phi_col[i], psi_list=psi_col[i],
                        encoding="sin_cos" if key == "phi_psi_sincos"
                                 else "raw"))
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
        if "depth" in features:
            check_msms_available()
        pdb_folder = Path(pdb_folder)
        entries = df_seq[ut.COL_ENTRY].tolist()
        sequences = df_seq[ut.COL_SEQ].tolist()
        D_total = get_total_dims(features)
        dict_pdb: Dict[str, np.ndarray] = {}
        ok_per_row: List[bool] = []
        for entry, seq in zip(entries, sequences):
            _check_entry_is_filesystem_safe(entry)
            L = len(seq)
            pdb_path = pdb_folder / f"{entry}.pdb"
            if not pdb_path.is_file():
                warnings.warn(
                    f"PDB file for entry '{entry}' not found at "
                    f"'{pdb_path}'; row will have pdb_ok=False",
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
        return dict_pdb, df_aug

    # ------------------------------------------------------------------
    # build_scales
    # ------------------------------------------------------------------
    def build_scales(
        self,
        features: List[str] = None,
        dim_names_override: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build the ``(df_scales, df_cat)`` metadata pair for the CPP constructor.

        Parameters
        ----------
        features : list of str
            Feature keys from the StructurePreprocessor registry, in the
            order they appear along the D axis of the encoder outputs. May
            mix ``encode_dssp`` and ``encode_pdb`` keys; the order here must
            match the order in which the corresponding ``dict_num`` blocks
            are concatenated (typically via :func:`combine_dict_nums`).
        dim_names_override : list of str, optional
            Replacement names for the D columns; length must equal the
            total dimensionality across ``features``. If ``None``, the
            registry defaults are used.

        Returns
        -------
        df_scales : pd.DataFrame, shape (20, D_total)
            AAontology-style scale frame. Rows are the 20 canonical AAs
            (values unused in numerical mode — :meth:`CPP.run_num` consumes
            ``dict_num`` directly). Columns name the D dimensions.
        df_cat : pd.DataFrame, shape (D_total, 5)
            AAontology-style category frame with one row per dimension:
            ``scale_id``, ``category``, ``subcategory``, ``scale_name``,
            ``scale_description``. The redundancy filter in :meth:`CPP.run_num`
            groups dimensions by ``subcategory``.

        Raises
        ------
        ValueError
            On invalid feature keys or override-length mismatch.
        """
        validate_feature_keys(features)
        D = get_total_dims(features)
        dim_names = get_dim_names(features)
        if dim_names_override is not None:
            if not isinstance(dim_names_override, list):
                raise ValueError(
                    f"'dim_names_override' "
                    f"({type(dim_names_override).__name__}) "
                    f"should be a list of str of length {D}")
            if len(dim_names_override) != D:
                raise ValueError(
                    f"'dim_names_override' (len={len(dim_names_override)}) "
                    f"should be a list of {D} str (one per output dim)")
            for n in dim_names_override:
                if not isinstance(n, str):
                    raise ValueError(
                        f"'dim_names_override' items ({n!r}) should be str")
            dim_names = list(dim_names_override)
        categories = get_categories(features)
        subcategories = get_subcategories(features)
        # df_scales: (20, D) with the canonical AA index
        aa_list = list(ut.LIST_CANONICAL_AA)
        df_scales = pd.DataFrame(
            np.zeros((len(aa_list), D), dtype=np.float64),
            index=aa_list, columns=dim_names)
        # df_cat: (D, 5)
        df_cat = pd.DataFrame({
            ut.COL_SCALE_ID: dim_names,
            ut.COL_CAT: categories,
            ut.COL_SUBCAT: subcategories,
            ut.COL_SCALE_NAME: dim_names,
            ut.COL_SCALE_DES: [f"{c}/{s}" for c, s in zip(categories,
                                                          subcategories)],
        })
        return df_scales, df_cat
