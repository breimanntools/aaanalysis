"""This is a script for the backend of the MMseqs2 method for the filter_seq() function."""
import os
import pandas as pd

from aaanalysis import utils as ut
from .comp_seq_sim import comp_seq_sim_
from ._utils import make_temp_dir, remove_temp, run_command, save_entries_to_fasta


COL_QUERY = "query"     # ID of cluster representative
COL_TARGET = "target"


# I Helper functions
def _get_df_mmseq(cluster_tsv):
    """Read MMseqs2 output to DataFrame"""
    df = pd.read_csv(cluster_tsv, sep='\t', header=None, names=[COL_QUERY, COL_TARGET, '_'])
    df[ut.COL_IS_REP] = (df[COL_QUERY] == df[COL_TARGET])
    return df


def _get_df_clust_mmseq(df_mmseq=None, df_seq=None, sort_clusters=False):
    """Get DataFrame with clustering information (consistent with CD-Hit output)"""
    df_seq = df_seq.copy()
    list_seq = df_seq[ut.COL_SEQ].to_list()
    dict_id_seq = dict(zip(df_seq[ut.COL_ENTRY].to_list(), list_seq))
    list_ids = df_mmseq[COL_TARGET].to_list()
    dict_id_rep = dict(zip(list_ids, df_mmseq[ut.COL_IS_REP]))

    # Pre-process sequence DataFrame
    if sort_clusters:
        df_seq["len_seq"] = [len(x) for x in list_seq]
        df_seq = df_seq.sort_values(by="len_seq", ascending=False)
        list_ids = df_seq[ut.COL_ENTRY].to_list()

    # Generate clusters
    dict_clust = {}
    clust_id = 0
    for entry_target in list_ids:
        if dict_id_rep[entry_target]:
            dict_clust[entry_target] = clust_id
            clust_id += 1

    # Build the cluster DataFrame
    list_entries = []
    for entry_target in list_ids:
        entry_query = df_mmseq[df_mmseq[COL_TARGET] == entry_target][COL_QUERY].values[0]
        cluster_id = dict_clust[entry_query]
        if entry_target != entry_query:
            identity = comp_seq_sim_(dict_id_seq[entry_target], dict_id_seq[entry_query])
            is_rep = 0
        else:
            identity = 100.0
            is_rep = 1
        list_entries.append([entry_target, cluster_id, identity, is_rep])
    df_clust = pd.DataFrame(list_entries, columns=[ut.COL_ENTRY, ut.COL_CLUST, ut.COL_REP_IDEN, ut.COL_IS_REP])
    df_clust = df_clust.sort_values(by=ut.COL_CLUST).reset_index(drop=True)
    return df_clust


# II Main functions
def run_mmseqs2(df_seq=None, similarity_threshold=0.7, word_size=None,
                coverage_long=None, coverage_short=None,
                n_jobs=None, sort_clusters=False, verbose=False):
    """Run MMseqs2 command to perform redundancy-reduction via clustering"""
    # Create temporary folder for input and temporary output
    result_prefix = "mmseq_"
    temp_dir = make_temp_dir(prefix=result_prefix)
    file_in = os.path.join(temp_dir, "_temp_mmseqs_in.fasta")
    save_entries_to_fasta(df_seq=df_seq, file_path=file_in)
    db_name = os.path.join(temp_dir, result_prefix + "DB")
    cluster_name = os.path.join(temp_dir, result_prefix + "Clu")
    cluster_tsv = os.path.join(temp_dir, f"{result_prefix}Clu.tsv")

    # Create the database
    cmd_db = ["mmseqs", "createdb", file_in, db_name]
    if verbose:
        ut.print_out("Run MMseqs2 filtering")
        ut.print_out("1. Create Database")
    run_command(cmd=cmd_db, verbose=verbose)

    # Perform clustering
    cmd_cluster = ["mmseqs", "cluster", db_name, cluster_name, temp_dir,
                   "--min-seq-id", str(similarity_threshold),
                   "--threads", str(n_jobs)]
    if word_size is not None:
        cmd_cluster.extend(["-k", str(word_size)])
    if coverage_long:
        cmd_cluster.extend(["--cov-mode", "0", "-c", str(coverage_long)])
    elif coverage_short:
        cmd_cluster.extend(["--cov-mode", "1", "-c", str(coverage_short)])
    if verbose:
        ut.print_out("2. Perform Clustering")
    run_command(cmd=cmd_cluster, verbose=verbose)

    # Create TSV file from clustering result
    cmd_tsv = ["mmseqs", "createtsv", db_name, db_name, cluster_name, cluster_tsv]
    if verbose:
        ut.print_out("3. Convert results into DataFrame")
    run_command(cmd=cmd_tsv, verbose=verbose)

    # Convert MMseqs2 output to clustering DataFrame
    df_mmseq = _get_df_mmseq(cluster_tsv)
    df_clust = _get_df_clust_mmseq(df_mmseq=df_mmseq, df_seq=df_seq, sort_clusters=sort_clusters)

    # Remove temporary file
    remove_temp(path=temp_dir)
    return df_clust
