import numpy as np
import pytest
from aaanalysis.aaclust._aaclust_bic import bic_score


def test_bic_known_dataset():
    """Test the BIC score with known dataset."""
    X = np.array([[1, 2], [5, 3], [1, 3], [5, 6], [8, 8]])
    labels = np.array([1, 2, 1, 2, 21])
    bic = round(bic_score(X, labels), 1)
    assert np.isclose(bic, -24.8)

def test_label_mapping():
    """Test with different label sets to ensure label mapping."""
    X = np.array([[1, 2], [5, 6], [1, 2], [5, 7]])

    # Test with labels not starting from 0
    labels_1 = np.array([1, 2, 1, 2])
    bic_1 = bic_score(X, labels_1)

    # Test with shuffled labels
    labels_2 = np.array([2, 1, 2, 1])
    bic_2 = bic_score(X, labels_2)

    assert np.isclose(bic_1, bic_2)  # The BIC score should be the same, regardless of the label order

def test_empty_input():
    """Test with empty input arrays."""
    with pytest.raises(ValueError):
        bic_score(np.array([]), np.array([]))

def test_invalid_labels():
    """Test with invalid label size."""
    X = np.array([[1, 2], [5, 6], [1, 2], [5, 7]])
    labels = np.array([1, 2, 1])  # Shorter than X
    with pytest.raises(Exception):  # Adjust the exception type if necessary
        bic_score(X, labels)

if __name__ == "__main__":
    pytest.main([__file__])
