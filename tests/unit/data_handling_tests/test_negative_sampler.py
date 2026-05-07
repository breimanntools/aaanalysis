"""Tests for NegativeSampler input normalization."""
import pandas as pd
import pytest

import aaanalysis as aa


def _df_pos(sequence=None):
    data = {
        "source_protein": ["P1", "P2"],
        "source_position": [0, 2],
    }
    if sequence is not None:
        data["sequence"] = sequence
    return pd.DataFrame(data, index=[5, 6])


def test_negative_sampler_root_imports():
    """NegativeSampler and SamplingFilters are public root imports."""
    assert aa.NegativeSampler is not None
    assert aa.SamplingFilters is not None


def test_normalizes_proteins_dict_and_generates_positive_windows():
    """Protein dict input becomes a reset-index internal protein table."""
    sampler = aa.NegativeSampler(
        df_pos=_df_pos(),
        proteins={"P1": "ACDE", "P2": "FGHIK"},
        window_size=3,
        random_state=42,
    )

    expected_proteins = pd.DataFrame(
        {
            "source_protein": ["P1", "P2"],
            "sequence": ["ACDE", "FGHIK"],
        }
    )
    expected_pos = pd.DataFrame(
        {
            "source_protein": ["P1", "P2"],
            "source_position": [0, 2],
            "sequence": ["-AC", "GHI"],
        }
    )
    pd.testing.assert_frame_equal(sampler.df_proteins_, expected_proteins)
    pd.testing.assert_frame_equal(sampler.df_pos_, expected_pos)
    assert sampler._random_state == 42
    assert sampler.output_schema == [
        "sequence",
        "source_protein",
        "source_position",
        "role",
        "strategy",
        "provenance",
    ]


def test_normalizes_df_seq_entry_sequence_input():
    """AAanalysis df_seq input is normalized to source_protein/sequence."""
    df_seq = pd.DataFrame(
        {
            "entry": ["P1", "P2"],
            "sequence": ["ACDE", "FGHIK"],
            "label": [1, 0],
        },
        index=[10, 11],
    )
    sampler = aa.NegativeSampler(df_pos=_df_pos(), df_seq=df_seq, window_size=5)

    expected_pos = pd.DataFrame(
        {
            "source_protein": ["P1", "P2"],
            "source_position": [0, 2],
            "sequence": ["--ACD", "FGHIK"],
        }
    )
    pd.testing.assert_frame_equal(sampler.df_pos_, expected_pos)
    assert sampler.df_proteins_.columns.tolist() == ["source_protein", "sequence"]


def test_validates_provided_positive_sequence_against_generated_window():
    """Provided positive sequence must match the anchored protein window."""
    sampler = aa.NegativeSampler(
        df_pos=_df_pos(sequence=["-AC", "GHI"]),
        proteins={"P1": "ACDE", "P2": "FGHIK"},
        window_size=3,
    )
    assert sampler.df_pos_["sequence"].tolist() == ["-AC", "GHI"]

    with pytest.raises(ValueError, match="does not match"):
        aa.NegativeSampler(
            df_pos=_df_pos(sequence=["ACD", "GHI"]),
            proteins={"P1": "ACDE", "P2": "FGHIK"},
            window_size=3,
        )


def test_requires_exactly_one_protein_source():
    """Constructor rejects missing or ambiguous protein sequence sources."""
    with pytest.raises(ValueError, match="exactly one"):
        aa.NegativeSampler(df_pos=_df_pos())

    with pytest.raises(ValueError, match="exactly one"):
        aa.NegativeSampler(
            df_pos=_df_pos(),
            proteins={"P1": "ACDE", "P2": "FGHIK"},
            df_seq=pd.DataFrame({"entry": ["P1"], "sequence": ["ACDE"]}),
        )


def test_validates_required_columns_and_position_compatibility():
    """Positive rows must reference known proteins and valid 0-based anchors."""
    with pytest.raises(ValueError, match="missing required columns"):
        aa.NegativeSampler(
            df_pos=pd.DataFrame({"source_protein": ["P1"]}),
            proteins={"P1": "ACDE"},
        )

    with pytest.raises(ValueError, match="unknown protein"):
        aa.NegativeSampler(
            df_pos=pd.DataFrame({"source_protein": ["P3"], "source_position": [0]}),
            proteins={"P1": "ACDE"},
        )

    with pytest.raises(ValueError, match="smaller than sequence length"):
        aa.NegativeSampler(
            df_pos=pd.DataFrame({"source_protein": ["P1"], "source_position": [10]}),
            proteins={"P1": "ACDE"},
        )


def test_validates_window_size_random_state_and_filters():
    """Constructor validates issue 66.1 scalar and filter dependencies."""
    with pytest.raises(ValueError, match="window_size"):
        aa.NegativeSampler(df_pos=_df_pos(), proteins={"P1": "ACDE", "P2": "FGHIK"}, window_size=0)

    with pytest.raises(ValueError, match="random_state"):
        aa.NegativeSampler(
            df_pos=_df_pos(),
            proteins={"P1": "ACDE", "P2": "FGHIK"},
            random_state="seed",
        )

    with pytest.raises(ValueError, match="structure_features"):
        aa.NegativeSampler(
            df_pos=_df_pos(),
            proteins={"P1": "ACDE", "P2": "FGHIK"},
            filters=aa.SamplingFilters(match_structure=True),
        )

    with pytest.raises(ValueError, match="custom_filter"):
        aa.NegativeSampler(
            df_pos=_df_pos(),
            proteins={"P1": "ACDE", "P2": "FGHIK"},
            filters={"custom_filter": "not-callable"},
        )


def test_public_table_accessors_return_copies():
    """Accessors should not allow mutation of internal normalized tables."""
    sampler = aa.NegativeSampler(
        df_pos=_df_pos(),
        proteins={"P1": "ACDE", "P2": "FGHIK"},
        window_size=3,
    )

    df_pos = sampler.df_pos_
    df_pos.loc[0, "sequence"] = "XXX"
    assert sampler.df_pos_.loc[0, "sequence"] == "-AC"
