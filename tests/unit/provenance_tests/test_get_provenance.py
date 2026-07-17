"""This is a script to test aa.get_provenance()."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
from aaanalysis import _provenance

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

LIST_KEYS = ["aaanalysis_version", "python_version", "dependencies", "git_commit",
             "random_state", "deterministic", "input_hash"]


class TestGetProvenance:
    """Test get_provenance() with one parameter per test."""

    # Positive tests: random_state
    def test_random_state_none(self):
        dict_provenance = aa.get_provenance()
        assert dict_provenance["random_state"] is None

    def test_random_state_int(self):
        dict_provenance = aa.get_provenance(random_state=42)
        assert dict_provenance["random_state"] == 42

    def test_random_state_zero(self):
        """0 is a valid seed even though it is falsy."""
        dict_provenance = aa.get_provenance(random_state=0)
        assert dict_provenance["random_state"] == 0
        assert dict_provenance["deterministic"] is True

    @settings(max_examples=15)
    @given(random_state=some.integers(min_value=0, max_value=2**31 - 1))
    def test_random_state_hypothesis(self, random_state):
        dict_provenance = aa.get_provenance(random_state=random_state)
        assert dict_provenance["random_state"] == random_state

    def test_deterministic_true_when_seeded(self):
        assert aa.get_provenance(random_state=1)["deterministic"] is True

    def test_deterministic_false_when_unseeded(self):
        assert aa.get_provenance(random_state=None)["deterministic"] is False

    # Positive tests: data
    def test_data_none_gives_no_hash(self):
        assert aa.get_provenance(data=None)["input_hash"] is None

    def test_data_dataframe(self):
        df = pd.DataFrame({"entry": ["P1"], "sequence": ["MKV"]})
        assert aa.get_provenance(data=df)["input_hash"].startswith("sha256:")

    def test_data_series(self):
        assert aa.get_provenance(data=pd.Series([1, 2, 3]))["input_hash"].startswith("sha256:")

    def test_data_ndarray(self):
        assert aa.get_provenance(data=np.arange(6))["input_hash"].startswith("sha256:")

    def test_data_list_of_str(self):
        assert aa.get_provenance(data=["MKV", "AAC"])["input_hash"].startswith("sha256:")

    def test_data_list_of_int(self):
        assert aa.get_provenance(data=[0, 1, 1, 0])["input_hash"].startswith("sha256:")

    def test_data_2d_array(self):
        assert aa.get_provenance(data=np.ones((3, 2)))["input_hash"].startswith("sha256:")

    @settings(max_examples=10)
    @given(data=some.lists(some.integers(), min_size=1, max_size=20))
    def test_data_hypothesis(self, data):
        assert aa.get_provenance(data=data)["input_hash"].startswith("sha256:")

    # Structural tests
    def test_returns_plain_dict(self):
        """The record must be a plain dict, never a bespoke envelope type."""
        assert type(aa.get_provenance()) is dict

    def test_record_keys(self):
        assert sorted(aa.get_provenance().keys()) == sorted(LIST_KEYS)

    def test_json_serializable(self):
        """KPI: the record round-trips through json.dumps."""
        dict_provenance = aa.get_provenance(random_state=42)
        assert json.loads(json.dumps(dict_provenance)) == dict_provenance

    def test_version_matches_package_attribute(self):
        assert aa.get_provenance()["aaanalysis_version"] == aa.__version__

    def test_python_version_is_str(self):
        assert isinstance(aa.get_provenance()["python_version"], str)

    def test_dependencies_reported(self):
        dict_deps = aa.get_provenance()["dependencies"]
        assert sorted(dict_deps) == sorted(["numpy", "pandas", "scikit-learn", "scipy"])

    def test_dependency_versions_are_str(self):
        """These are hard dependencies, so every one resolves in a real install."""
        assert all(isinstance(v, str) for v in aa.get_provenance()["dependencies"].values())

    def test_git_commit_str_or_none(self):
        git_commit = aa.get_provenance()["git_commit"]
        assert git_commit is None or isinstance(git_commit, str)

    def test_no_timestamp_field(self):
        """The record is a reproducibility key: no field may vary between runs."""
        assert aa.get_provenance(random_state=1) == aa.get_provenance(random_state=1)

    # Negative tests: random_state
    def test_negative_random_state_raises(self):
        with pytest.raises(ValueError):
            aa.get_provenance(random_state=-1)

    def test_float_random_state_raises(self):
        with pytest.raises(ValueError):
            aa.get_provenance(random_state=1.5)

    def test_str_random_state_raises(self):
        with pytest.raises(ValueError):
            aa.get_provenance(random_state="42")

    def test_list_random_state_raises(self):
        with pytest.raises(ValueError):
            aa.get_provenance(random_state=[1])

    def test_dict_random_state_raises(self):
        with pytest.raises(ValueError):
            aa.get_provenance(random_state={"seed": 1})

    def test_invalid_random_states_sweep(self):
        for random_state in [-1, -100, 1.5, "a", "42", [1], (1,), {"a": 1}, np.nan]:
            with pytest.raises(ValueError):
                aa.get_provenance(random_state=random_state)

    @settings(max_examples=10)
    @given(random_state=some.integers(max_value=-1))
    def test_negative_random_state_hypothesis(self, random_state):
        with pytest.raises(ValueError):
            aa.get_provenance(random_state=random_state)

    # Negative tests: data
    def test_ragged_data_raises(self):
        """A ragged nested sequence has no well-defined array form."""
        with pytest.raises(ValueError):
            aa.get_provenance(data=[[1, 2], [3]])


class TestGetProvenanceComplex:
    """Test get_provenance() with interacting parameters and the options override."""

    # Positive tests
    def test_options_random_state_overrides_argument(self):
        """The contract's surprise: options wins over the passed value.

        This is the record's reason to exist -- the effective seed is not
        recoverable from the call site alone.
        """
        aa.options["random_state"] = 7
        assert aa.get_provenance(random_state=42)["random_state"] == 7

    def test_options_random_state_used_when_no_argument(self):
        aa.options["random_state"] = 7
        assert aa.get_provenance()["random_state"] == 7

    def test_options_off_falls_back_to_argument(self):
        aa.options["random_state"] = "off"
        assert aa.get_provenance(random_state=42)["random_state"] == 42

    def test_options_random_state_makes_unseeded_call_deterministic(self):
        aa.options["random_state"] = 0
        dict_provenance = aa.get_provenance()
        assert dict_provenance["random_state"] == 0
        assert dict_provenance["deterministic"] is True

    def test_same_seed_and_data_give_identical_record(self):
        """KPI: two runs sharing provenance reproduce the same result."""
        df = pd.DataFrame({"entry": ["P1", "P2"], "sequence": ["MKV", "AAC"]})
        assert aa.get_provenance(random_state=42, data=df) == \
               aa.get_provenance(random_state=42, data=df)

    def test_seed_and_data_combine(self):
        dict_provenance = aa.get_provenance(random_state=3, data=[1, 2])
        assert dict_provenance["random_state"] == 3
        assert dict_provenance["input_hash"].startswith("sha256:")

    def test_json_round_trip_with_data(self):
        df = pd.DataFrame({"a": [1, 2]})
        dict_provenance = aa.get_provenance(random_state=1, data=df)
        assert json.loads(json.dumps(dict_provenance)) == dict_provenance

    def test_different_data_gives_different_hash(self):
        hash_a = aa.get_provenance(data=[1, 2, 3])["input_hash"]
        hash_b = aa.get_provenance(data=[1, 2, 4])["input_hash"]
        assert hash_a != hash_b

    def test_renamed_column_gives_different_hash(self):
        """Column names are part of the input identity."""
        hash_a = aa.get_provenance(data=pd.DataFrame({"a": [1, 2]}))["input_hash"]
        hash_b = aa.get_provenance(data=pd.DataFrame({"b": [1, 2]}))["input_hash"]
        assert hash_a != hash_b

    def test_reordered_rows_give_different_hash(self):
        df = pd.DataFrame({"a": [1, 2]})
        hash_a = aa.get_provenance(data=df)["input_hash"]
        hash_b = aa.get_provenance(data=df.iloc[::-1])["input_hash"]
        assert hash_a != hash_b

    def test_reshaped_array_gives_different_hash(self):
        """Shape is folded in, so the same buffer at a different shape differs."""
        hash_a = aa.get_provenance(data=np.arange(6))["input_hash"]
        hash_b = aa.get_provenance(data=np.arange(6).reshape(2, 3))["input_hash"]
        assert hash_a != hash_b

    def test_seed_does_not_affect_input_hash(self):
        assert aa.get_provenance(random_state=1, data=[1, 2])["input_hash"] == \
               aa.get_provenance(random_state=2, data=[1, 2])["input_hash"]

    # Negative tests
    def test_invalid_option_rejected_at_assignment(self):
        """``options`` validates on set, so a bad seed never reaches the record.

        The record can therefore only ever report a seed that is valid to use.
        """
        for random_state in [-5, 1.5, "bad", [1]]:
            with pytest.raises(ValueError):
                aa.options["random_state"] = random_state
        assert aa.get_provenance(random_state=42)["random_state"] == 42

    def test_invalid_option_leaves_previous_value_intact(self):
        aa.options["random_state"] = 7
        with pytest.raises(ValueError):
            aa.options["random_state"] = -5
        assert aa.get_provenance()["random_state"] == 7

    def test_invalid_seed_raises_even_with_valid_data(self):
        with pytest.raises(ValueError):
            aa.get_provenance(random_state=-1, data=[1, 2])

    def test_invalid_data_raises_even_with_valid_seed(self):
        with pytest.raises(ValueError):
            aa.get_provenance(random_state=42, data=[[1, 2], [3]])

    def test_invalid_seed_raises_before_hashing_data(self):
        """Validation comes first, so a bad seed is reported, not a hash error."""
        with pytest.raises(ValueError, match="random_state"):
            aa.get_provenance(random_state=-1, data=[[1, 2], [3]])


class TestGetProvenanceGitCommit:
    """Test that git_commit reports *this package's* checkout and never a host repo.

    ``git`` searches upward for a repository, so an install nested under an unrelated
    repository (the common ``<user-repo>/.venv/...`` layout) would otherwise answer with
    the user's commit -- misleading provenance, which is worse than none.
    """

    @staticmethod
    def _fake_git(top_level=None, commit=None):
        def _run_git(package_dir=None, *args):
            if args == ("rev-parse", "--show-toplevel"):
                return top_level
            if args == ("rev-parse", "HEAD"):
                return commit
            return None
        return _run_git

    def test_commit_when_toplevel_is_the_package_parent(self, monkeypatch):
        """A real source checkout: the repo root directly contains the package."""
        package_parent = Path(_provenance.__file__).resolve().parent.parent
        monkeypatch.setattr(_provenance, "_run_git",
                            self._fake_git(top_level=str(package_parent), commit="abc123"))
        assert _provenance._get_git_commit() == "abc123"

    def test_none_when_toplevel_is_an_unrelated_host_repo(self, monkeypatch):
        """The leak case: a venv inside the user's own repository."""
        monkeypatch.setattr(_provenance, "_run_git",
                            self._fake_git(top_level="/some/user/project",
                                           commit="userrepocommit"))
        assert _provenance._get_git_commit() is None

    def test_none_when_not_in_a_repository(self, monkeypatch):
        """A regular PyPI install: no repository at all."""
        monkeypatch.setattr(_provenance, "_run_git", self._fake_git(top_level=None))
        assert _provenance._get_git_commit() is None

    def test_none_when_git_is_unavailable(self, monkeypatch):
        """git missing / not executable must degrade to None, never raise."""
        def _boom(*args, **kwargs):
            raise OSError("git not found")
        monkeypatch.setattr(_provenance.subprocess, "run", _boom)
        assert _provenance._get_git_commit() is None

    def test_none_when_git_returns_nonzero(self, monkeypatch):
        class _Proc:
            returncode = 128
            stdout = ""
        monkeypatch.setattr(_provenance.subprocess, "run", lambda *a, **k: _Proc())
        assert _provenance._get_git_commit() is None

    def test_record_never_raises_without_git(self, monkeypatch):
        """The record must still be produced when git is unavailable."""
        def _boom(*args, **kwargs):
            raise OSError("git not found")
        monkeypatch.setattr(_provenance.subprocess, "run", _boom)
        dict_provenance = aa.get_provenance(random_state=42)
        assert dict_provenance["git_commit"] is None
        assert dict_provenance["random_state"] == 42
