"""Shared, tiny, seeded pipeline builders for the integration + e2e test tiers.

These helpers wire the *real* public components together (no mocks) at a small,
fixed size so the cross-component tests stay fast under xdist. Both
``tests/integration`` (via fixtures in its ``conftest.py``) and ``tests/e2e``
import from here, so a seam's call pattern is defined once. Assertions live in
the test files; this module only builds artifacts. See ADR-0031.
"""
import aaanalysis as aa

# A small balanced run: load_dataset(n=N) returns N rows per class (2N total).
SEED = 0
N_PER_CLASS = 10
N_FILTER = 20
N_SCALES = 20


def small_scales(n_scales: int = N_SCALES):
    """A small, fixed ``df_scales`` (rows = amino acids, cols = scales)."""
    return aa.load_scales(top60_n=n_scales).T.head(n_scales).T


def load_dom_gsec(n: int = N_PER_CLASS):
    """Balanced domain-level df_seq (position-based: has ``tmd_start``/``tmd_stop``)."""
    return aa.load_dataset(name="DOM_GSEC", n=n)


def build_df_feat(df_parts, labels, df_scales, n_filter: int = N_FILTER):
    """Run CPP on the given parts/labels/scales and return the ranked ``df_feat``."""
    cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False, random_state=SEED)
    return cpp.run(labels=labels, n_filter=n_filter, n_jobs=1)


def feature_matrix(df_feat, df_parts, df_scales):
    """Build ``X`` from a feature set + parts, using the SAME scales as ``build_df_feat``."""
    sf = aa.SequenceFeature()
    return sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts,
                             df_scales=df_scales, n_jobs=1)


def build_pipeline(n: int = N_PER_CLASS, n_filter: int = N_FILTER, n_scales: int = N_SCALES):
    """Build the full load -> parts -> CPP -> feature-matrix spine once, seeded.

    Returns a dict of the shared artifacts so callers can assert on any seam
    without re-running the (relatively expensive) CPP step.
    """
    df_seq = load_dom_gsec(n=n)
    labels = df_seq["label"].to_list()
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
    df_scales = small_scales(n_scales=n_scales)
    df_feat = build_df_feat(df_parts=df_parts, labels=labels, df_scales=df_scales, n_filter=n_filter)
    X = feature_matrix(df_feat=df_feat, df_parts=df_parts, df_scales=df_scales)
    return dict(df_seq=df_seq, labels=labels, df_parts=df_parts,
                df_scales=df_scales, df_feat=df_feat, X=X)
