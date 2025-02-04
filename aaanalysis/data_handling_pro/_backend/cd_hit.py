"""This is a script for the backend of the CH-HIT method for the filter_seq() function."""
import pandas as pd
import os

from ._utils import make_temp_dir, remove_temp, run_command, save_entries_to_fasta

from aaanalysis import utils as ut


# I Helper functions
def _select_word_size(st=0.5):
    """
    Determines an optimal word size based on the similarity threshold (st).

    The word size is the length of the sequence fragments (words) used in the initial scanning phase of clustering.
    Shorter words increase sensitivity and are suitable for lower similarity thresholds,
    while longer words enhance performance and precision at higher thresholds.
    """
    # 5: Optimal for high similarity (>= 0.7)
    # 4: For moderate to high similarity (>= 0.6)
    # 3: For moderate similarity (>= 0.5)
    # 2: For lower similarity (< 0.5)
    word_size = 5 if st >= 0.7 else (4 if st >= 0.6 else (3 if st >= 0.5 else 2))
    return word_size


def _get_df_clust_cd_hit(file_cd_hit_out):
    """
    Parse the .clstr file from CD-HIT and return a DataFrame with cluster information.
    """
    list_entries = []
    with open(file_cd_hit_out, 'r') as file:
        cluster_id = None
        for line in file:
            line = line.strip()
            if line.startswith('>Cluster'):
                cluster_id = int(line.split()[1])  # Capture the cluster number
            else:
                parts = line.split(', >')
                entry = parts[1].split('...')[0]
                is_rep = 1 if '*' in parts[1] else 0
                if is_rep:
                    identity = 100.0
                else:
                    str_iden = parts[-1].split("at ")[1]
                    if "/" in str_iden:
                        str_iden = str_iden.split("/")[1]
                    identity = float(str_iden.strip("%"))
                list_entries.append([entry, cluster_id, identity, is_rep])
    df_clust = pd.DataFrame(list_entries, columns=[ut.COL_ENTRY, ut.COL_CLUST, ut.COL_REP_IDEN, ut.COL_IS_REP])
    return df_clust


# II Main functions
def run_cd_hit(df_seq=None,
               similarity_threshold=0.7, word_size=None,
               global_identity=True,
               coverage_long=None, coverage_short=None,
               n_jobs=None, sort_clusters=False, verbose=False):
    """Run CD-HIT command to perform redundancy-reduction via clustering"""
    # Create temporary folder for input and temporary output
    result_prefix = "cdhit_"
    temp_dir = make_temp_dir(prefix=result_prefix)
    file_in = os.path.join(temp_dir, f"_{result_prefix}_in")
    save_entries_to_fasta(df_seq=df_seq, file_path=file_in)
    file_out = os.path.join(temp_dir, f"_{result_prefix}_out")
    # Create CD-HIT command
    if word_size is None:
        word_size = _select_word_size(st=similarity_threshold)
    cmd = ["cd-hit", "-i", file_in,
           "-o", file_out,
           "-c", str(similarity_threshold),
           "-n", str(word_size),
           "-T", str(n_jobs if n_jobs is not None else 1),
           "-G", "1" if global_identity else "0",
           ]

    if not global_identity:
        # Use common 80% values for -A, -aS, and -aL options (if not specified)
        cmd.extend(["-A", "0.8"])
        coverage_long = 0.8 if coverage_long is None else coverage_long
        coverage_short = 0.8 if coverage_short is None else coverage_short
    if coverage_long:
        cmd.extend(["-aL", str(coverage_long)])
    if coverage_short:
        cmd.extend(["-aS", str(coverage_short)])
    if sort_clusters:
        cmd.extend(["-sc", "1"])

    # Run CD-HIT command
    if verbose:
        ut.print_out("Run CD-HIT filtering")
    run_command(cmd=cmd, verbose=verbose, temp_dir=temp_dir)

    # Convert CD-Hit output to clustering DataFrame
    file_cd_hit_out = file_out + ".clstr"
    df_clust = _get_df_clust_cd_hit(file_cd_hit_out=file_cd_hit_out)
    # Remove temporary file
    remove_temp(path=temp_dir)
    return df_clust
