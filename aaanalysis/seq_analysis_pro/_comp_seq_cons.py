"""
This is a script for sequence conservation analysis in proteins.
It retrieves MSAs, computes conservation scores, and maps known disease mutations.
"""

from typing import Optional
import os
import shutil
import time
import requests
import pandas as pd
import numpy as np
from Bio import AlignIO
from collections import Counter

# I. Helper Functions

import time
import pandas as pd
import numpy as np


# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe


# I Helper Functions
def check_is_tool(name: Optional[str] = None):
    """Check whether `name` is on PATH and marked as executable."""
    if not shutil.which(name):
        raise ValueError(f"{name} is not installed or not in the PATH.")


def compute_shannon_entropy(column: list) -> float:
    """Compute sequence conservation per position using Shannon entropy."""
    counts = Counter(column)
    total = sum(counts.values())
    probabilities = [count / total for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)


def normalize_scores(scores: list) -> np.ndarray:
    """Normalize conservation scores to range [0,1] (higher = more conserved)."""
    max_val = np.max(scores)
    return (max_val - scores) / max_val if max_val else scores


# TODO compute sequence conservation per position (input df_seq or df_parts)
# TODO interface to biopython (use different tools to get MSA, MSA for conersvation)
# II Main Functions
def get_msa(uniprot_id: str, output_file: str = "msa.fasta") -> str:
    """
    Retrieve multiple sequence alignment (MSA) from UniProt.

    Parameters
    ----------
    uniprot_id : str
        UniProt ID of the target protein.
    output_file : str, default="msa.fasta"
        File path to save the retrieved MSA.

    Returns
    -------
    str
        Path to the saved MSA file.

    Raises
    ------
    ValueError
        If the MSA retrieval fails.
    """
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)

    if response.status_code == 200:
        with open(output_file, "w") as f:
            f.write(response.text)
        return output_file
    else:
        raise ValueError(f"Failed to retrieve MSA for {uniprot_id} (HTTP {response.status_code})")


def comp_seq_cons(msa_file: str) -> pd.DataFrame:
    """
    Compute sequence conservation per position from a given MSA.

    Parameters
    ----------
    msa_file : str
        Path to the MSA file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing residue positions and conservation scores.

    Raises
    ------
    ValueError
        If the MSA file is not found or invalid.
    """
    if not os.path.exists(msa_file):
        raise ValueError(f"MSA file '{msa_file}' not found.")

    alignment = AlignIO.read(msa_file, "fasta")
    conservation_scores = [compute_shannon_entropy(column) for column in zip(*alignment)]
    conservation_scores = normalize_scores(conservation_scores)

    df_conservation = pd.DataFrame({
        "Position": range(1, len(conservation_scores) + 1),
        "Conservation_Score": conservation_scores
    })
    return df_conservation


def map_known_mutations(df_conservation: pd.DataFrame, df_mutations: pd.DataFrame) -> pd.DataFrame:
    """
    Map known disease-associated mutations onto sequence conservation scores.

    Parameters
    ----------
    df_conservation : pd.DataFrame
        DataFrame with conservation scores (columns: 'Position', 'Conservation_Score').
    df_mutations : pd.DataFrame
        DataFrame with known mutations (columns: 'Position', 'Mutation').

    Returns
    -------
    pd.DataFrame
        Merged DataFrame showing conservation scores for mutated positions.
    """
    df_merged = df_conservation.merge(df_mutations, on="Position", how="inner")
    return df_merged
