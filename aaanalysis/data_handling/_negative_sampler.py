"""Negative/reference sequence sampling primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd

import aaanalysis.utils as ut


COL_SOURCE_PROTEIN = "source_protein"
COL_SOURCE_POSITION = "source_position"
COL_ROLE = "role"
COL_STRATEGY = "strategy"
COL_PROVENANCE = "provenance"
OUTPUT_SCHEMA = [
    ut.COL_SEQ,
    COL_SOURCE_PROTEIN,
    COL_SOURCE_POSITION,
    COL_ROLE,
    COL_STRATEGY,
    COL_PROVENANCE,
]
RESIDUE_CLASS_OPTIONS = ["P1", "P1_prime", "P1_P1_prime"]
SIMILARITY_METRIC_OPTIONS = ["pwm", "cpp", "embedding"]


@dataclass(frozen=True)
class SamplingFilters:
    """Configuration for filters applied by :class:`NegativeSampler`.

    The filtering implementation is added in later issue slices. This dataclass
    is public now so constructor validation and reproducible configs have one
    stable representation.
    """

    min_distance_to_positive: Optional[int] = None
    match_residue_class: Optional[str] = None
    match_structure: Optional[Any] = None
    structure_tolerance: Optional[Any] = None
    similarity_max: Optional[float] = None
    similarity_min: Optional[float] = None
    similarity_metric: str = "pwm"
    custom_filter: Optional[Callable[[pd.DataFrame, "NegativeSampler"], Any]] = None


class NegativeSampler:
    """Normalize positives and protein sequences for negative sampling.

    Parameters
    ----------
    df_pos : pandas.DataFrame
        Positive event table. Must contain ``source_protein`` and
        ``source_position``. If ``sequence`` is present, it is validated against
        the anchored window generated from the supplied protein sequence.
    proteins : dict, optional
        Mapping from protein identifier to protein sequence. Exactly one of
        ``proteins`` and ``df_seq`` must be supplied.
    df_seq : pandas.DataFrame, optional
        AAanalysis-style sequence table with ``entry`` and ``sequence`` columns.
    structure_features : pandas.DataFrame or object, optional
        Per-position structure or embedding features used by later filters.
    window_size : int, default=9
        Length of the centered sequence window. ``source_position`` is a 0-based
        anchor.
    filters : SamplingFilters or dict, optional
        Filter configuration. Dict inputs are converted to ``SamplingFilters``.
    random_state : int, optional
        Random state validated through ``aa.options`` via
        :func:`aaanalysis.utils.check_random_state`.
    """

    output_schema = OUTPUT_SCHEMA

    def __init__(
        self,
        df_pos: pd.DataFrame,
        proteins: Optional[dict[Any, str]] = None,
        df_seq: Optional[pd.DataFrame] = None,
        structure_features: Optional[Any] = None,
        window_size: int = 9,
        filters: Optional[SamplingFilters | dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        ut.check_number_range(
            name="window_size",
            val=window_size,
            min_val=1,
            accept_none=False,
            just_int=True,
        )
        random_state = ut.check_random_state(random_state=random_state)
        filters = self._normalize_filters(filters=filters)
        self._validate_filter_dependencies(
            filters=filters,
            structure_features=structure_features,
        )

        self.window_size = int(window_size)
        self._left_context = (self.window_size - 1) // 2
        self._right_context = self.window_size - 1 - self._left_context
        self._random_state = random_state
        self._filters = filters
        self._structure_features = (
            structure_features.copy().reset_index(drop=True)
            if isinstance(structure_features, pd.DataFrame)
            else structure_features
        )
        self._df_proteins = self._normalize_proteins(
            proteins=proteins,
            df_seq=df_seq,
        )
        self._protein_sequences = dict(
            zip(
                self._df_proteins[COL_SOURCE_PROTEIN],
                self._df_proteins[ut.COL_SEQ],
            )
        )
        self._df_pos = self._normalize_df_pos(df_pos=df_pos)

    @property
    def df_proteins_(self) -> pd.DataFrame:
        """Normalized protein table with ``source_protein`` and ``sequence``."""
        return self._df_proteins.copy()

    @property
    def df_pos_(self) -> pd.DataFrame:
        """Normalized positive table with generated/validated windows."""
        return self._df_pos.copy()

    @property
    def filters_(self) -> SamplingFilters:
        """Validated sampling filter configuration."""
        return self._filters

    def sample_same_protein(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Sample same-protein negatives.

        Implemented in issue 66.2.
        """
        raise NotImplementedError("sample_same_protein is implemented in issue 66.2.")

    def sample_different_protein(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Sample different-protein unlabeled or asserted-negative windows.

        Implemented in issue 66.2.
        """
        raise NotImplementedError("sample_different_protein is implemented in issue 66.2.")

    def sample_synthetic(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Sample synthetic control windows.

        Implemented in issue 66.4.
        """
        raise NotImplementedError("sample_synthetic is implemented in issue 66.4.")

    def sample_benchmark_set(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Sample and concatenate named benchmark arms.

        Implemented in issue 66.5.
        """
        raise NotImplementedError("sample_benchmark_set is implemented in issue 66.5.")

    @staticmethod
    def _normalize_filters(
        filters: Optional[SamplingFilters | dict[str, Any]],
    ) -> SamplingFilters:
        if filters is None:
            filters = SamplingFilters()
        elif isinstance(filters, dict):
            try:
                filters = SamplingFilters(**filters)
            except TypeError as error:
                raise ValueError(f"'filters' contains unsupported keys: {error}") from error
        elif not isinstance(filters, SamplingFilters):
            raise ValueError("'filters' should be a SamplingFilters object, a dict, or None.")

        ut.check_number_range(
            name="filters.min_distance_to_positive",
            val=filters.min_distance_to_positive,
            min_val=0,
            accept_none=True,
            just_int=True,
        )
        ut.check_str_options(
            name="filters.match_residue_class",
            val=filters.match_residue_class,
            accept_none=True,
            list_str_options=RESIDUE_CLASS_OPTIONS,
        )
        ut.check_number_range(
            name="filters.similarity_max",
            val=filters.similarity_max,
            min_val=-1,
            max_val=1,
            accept_none=True,
            just_int=False,
        )
        ut.check_number_range(
            name="filters.similarity_min",
            val=filters.similarity_min,
            min_val=-1,
            max_val=1,
            accept_none=True,
            just_int=False,
        )
        if (
            filters.similarity_min is not None
            and filters.similarity_max is not None
            and filters.similarity_min > filters.similarity_max
        ):
            raise ValueError("'filters.similarity_min' should be <= 'filters.similarity_max'.")
        ut.check_str_options(
            name="filters.similarity_metric",
            val=filters.similarity_metric,
            accept_none=False,
            list_str_options=SIMILARITY_METRIC_OPTIONS,
        )
        if filters.custom_filter is not None and not callable(filters.custom_filter):
            raise ValueError("'filters.custom_filter' should be callable or None.")
        return filters

    @staticmethod
    def _validate_filter_dependencies(
        filters: SamplingFilters,
        structure_features: Optional[Any],
    ) -> None:
        if filters.match_structure and structure_features is None:
            raise ValueError("'structure_features' is required when 'filters.match_structure' is set.")
        if filters.similarity_metric == "embedding" and (
            filters.similarity_min is not None or filters.similarity_max is not None
        ) and structure_features is None:
            raise ValueError(
                "'structure_features' is required for embedding similarity filtering."
            )
        if isinstance(structure_features, pd.DataFrame):
            ut.check_df(
                name="structure_features",
                df=structure_features,
                cols_required=[COL_SOURCE_PROTEIN, COL_SOURCE_POSITION],
                cols_nan_check=[COL_SOURCE_PROTEIN, COL_SOURCE_POSITION],
            )

    @staticmethod
    def _normalize_proteins(
        proteins: Optional[dict[Any, str]],
        df_seq: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        if (proteins is None) == (df_seq is None):
            raise ValueError("Provide exactly one protein source: 'proteins' or 'df_seq'.")

        if proteins is not None:
            ut.check_dict(name="proteins", val=proteins, accept_none=False)
            if len(proteins) == 0:
                raise ValueError("'proteins' should contain at least one sequence.")
            rows = []
            for protein_id, sequence in proteins.items():
                NegativeSampler._validate_protein_id(protein_id=protein_id, name="proteins")
                NegativeSampler._validate_sequence(sequence=sequence, name=f"proteins[{protein_id!r}]")
                rows.append({COL_SOURCE_PROTEIN: protein_id, ut.COL_SEQ: sequence})
            df_proteins = pd.DataFrame(rows)
        else:
            ut.check_df(
                name="df_seq",
                df=df_seq,
                cols_required=[ut.COL_ENTRY, ut.COL_SEQ],
                cols_nan_check=[ut.COL_ENTRY, ut.COL_SEQ],
            )
            df_proteins = df_seq[[ut.COL_ENTRY, ut.COL_SEQ]].copy()
            df_proteins = df_proteins.rename(columns={ut.COL_ENTRY: COL_SOURCE_PROTEIN})
            for protein_id in df_proteins[COL_SOURCE_PROTEIN]:
                NegativeSampler._validate_protein_id(protein_id=protein_id, name="df_seq.entry")
            for protein_id, sequence in zip(df_proteins[COL_SOURCE_PROTEIN], df_proteins[ut.COL_SEQ]):
                NegativeSampler._validate_sequence(sequence=sequence, name=f"df_seq.sequence ({protein_id!r})")

        if df_proteins[COL_SOURCE_PROTEIN].duplicated().any():
            duplicates = df_proteins.loc[
                df_proteins[COL_SOURCE_PROTEIN].duplicated(),
                COL_SOURCE_PROTEIN,
            ].tolist()
            raise ValueError(f"Protein identifiers should be unique, but duplicates were found: {duplicates}")
        return df_proteins.reset_index(drop=True)

    def _normalize_df_pos(self, df_pos: pd.DataFrame) -> pd.DataFrame:
        ut.check_df(
            name="df_pos",
            df=df_pos,
            cols_required=[COL_SOURCE_PROTEIN, COL_SOURCE_POSITION],
            cols_nan_check=[COL_SOURCE_PROTEIN, COL_SOURCE_POSITION],
        )
        df_pos_norm = df_pos.copy().reset_index(drop=True)

        generated_sequences = []
        for i, row in df_pos_norm.iterrows():
            protein_id = row[COL_SOURCE_PROTEIN]
            if protein_id not in self._protein_sequences:
                raise ValueError(
                    f"'df_pos.{COL_SOURCE_PROTEIN}' contains unknown protein "
                    f"{protein_id!r} at row {i}."
                )
            position = row[COL_SOURCE_POSITION]
            ut.check_number_range(
                name=f"df_pos.{COL_SOURCE_POSITION} (row {i})",
                val=position,
                min_val=0,
                accept_none=False,
                just_int=True,
            )
            position = int(position)
            sequence = self._protein_sequences[protein_id]
            if position >= len(sequence):
                raise ValueError(
                    f"'df_pos.{COL_SOURCE_POSITION}' ({position}) should be smaller "
                    f"than sequence length ({len(sequence)}) for protein {protein_id!r}."
                )
            df_pos_norm.at[i, COL_SOURCE_POSITION] = position
            generated_sequence = self._get_anchor_window(
                sequence=sequence,
                source_position=position,
            )
            generated_sequences.append(generated_sequence)

        if ut.COL_SEQ in df_pos_norm.columns:
            for i, (observed, expected) in enumerate(
                zip(df_pos_norm[ut.COL_SEQ], generated_sequences)
            ):
                self._validate_sequence(sequence=observed, name=f"df_pos.sequence (row {i})")
                if observed != expected:
                    raise ValueError(
                        f"'df_pos.sequence' at row {i} does not match the "
                        f"{self.window_size}-residue window generated from "
                        f"'{COL_SOURCE_PROTEIN}' and '{COL_SOURCE_POSITION}'."
                    )
        else:
            df_pos_norm[ut.COL_SEQ] = generated_sequences
        return df_pos_norm

    @staticmethod
    def _validate_protein_id(protein_id: Any, name: str) -> None:
        if pd.isna(protein_id):
            raise ValueError(f"'{name}' contains a missing protein identifier.")

    @staticmethod
    def _validate_sequence(sequence: Any, name: str) -> None:
        ut.check_str(name=name, val=sequence, accept_none=False)
        if len(sequence) == 0:
            raise ValueError(f"'{name}' should not be empty.")

    def _get_anchor_window(self, sequence: str, source_position: int) -> str:
        start = source_position - self._left_context
        stop = source_position + self._right_context
        chars = [
            sequence[pos] if 0 <= pos < len(sequence) else ut.STR_AA_GAP
            for pos in range(start, stop + 1)
        ]
        return "".join(chars)
