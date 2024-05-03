import subprocess
import shutil

# TODO test, adjust, finish (see ChatGPT: Model Performance Correlation Analysis including STD)


# I Helper functions
def _is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return shutil.which(name) is not None


def _select_longest_representatives(cluster_tsv, all_sequences_file, output_file):
    seq_dict = {}
    with open(all_sequences_file, 'r') as file:
        current_id = None
        for line in file:
            if line.startswith('>'):
                current_id = line.strip().split()[0][1:]
                seq_dict[current_id] = ""
            else:
                seq_dict[current_id] += line.strip()

    clusters = {}
    with open(cluster_tsv, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            cluster_id, seq_id = parts[0], parts[1]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(seq_id)

    representatives = {}
    for cluster_id, seq_ids in clusters.items():
        longest_seq_id = max(seq_ids, key=lambda x: len(seq_dict[x]))
        representatives[longest_seq_id] = seq_dict[longest_seq_id]

    with open(output_file, 'w') as out_file:
        for seq_id, sequence in representatives.items():
            out_file.write(f">{seq_id}\n{sequence}\n")


def _run_cd_hit(input_file, output_file, similarity_threshold, word_size, threads, coverage_long=None, coverage_short=None, verbose=False):
    """Helper function to run CD-HIT with provided parameters."""
    cmd = [
        "cd-hit", "-i", input_file, "-o", output_file,
        "-c", str(similarity_threshold), "-n", str(word_size), "-T", str(threads)
    ]
    if coverage_long:
        cmd.extend(["-aL", str(coverage_long)])
    if coverage_short:
        cmd.extend(["-aS", str(coverage_short)])
    if verbose:
        cmd.extend(["-d", "0"])
    subprocess.run(cmd, check=True)
    print("CD-HIT clustering completed. Representatives are saved in:", output_file)


def _run_mmseq(input_file, output_file, similarity_threshold, word_size, threads, verbose=False):
    """Helper function to run MMSeq2 with provided parameters."""
    tmp_directory = "tmp"
    result_prefix = "result_"
    db_name = result_prefix + "DB"
    cluster_name = result_prefix + "Clu"

    subprocess.run(["mmseqs", "createdb", input_file, db_name], check=True)
    cmd = [
        "mmseqs", "cluster", db_name, cluster_name, tmp_directory,
        "--min-seq-id", str(similarity_threshold), "-k", str(word_size), "--threads", str(threads)
    ]
    if verbose:
        cmd.extend(["-v", "3"])

    subprocess.run(cmd, check=True)
    cluster_tsv = f"{result_prefix}Clu.tsv"
    subprocess.run(["mmseqs", "createtsv", db_name, db_name, cluster_name, cluster_tsv], check=True)

    _select_longest_representatives(cluster_tsv, input_file, output_file)
    print("MMseq2 clustering completed. Representatives are saved in:", output_file)


# II Main function
def filter_seq(method, input_file, output_file, similarity_threshold=0.7, word_size=5, coverage_long=None, coverage_short=None, threads=1, verbose=False):
    """Perform redundancy-reduction of sequences by calling CD-Hit or MMSeq2 algorithm."""
    if method not in ['cd-hit', 'mmseq2']:
        raise ValueError("Invalid method specified. Use 'cd-hit' or 'mmseq2'.")

    if method == 'cd-hit' and not _is_tool('cd-hit'):
        raise RuntimeError("CD-HIT is not installed or not in the PATH.")
    elif method == 'mmseq2' and not _is_tool('mmseqs'):
        raise RuntimeError("MMseq2 is not installed or not in the PATH.")

    if method == "cd-hit":
        _run_cd_hit(input_file=input_file, output_file=output_file,
                    similarity_threshold=similarity_threshold, word_size=word_size,
                    threads=threads, coverage_long=coverage_long, coverage_short=coverage_short, verbose=verbose)
    else:
        _run_mmseq(input_file=input_file, output_file=output_file,
                   similarity_threshold=similarity_threshold, word_size=word_size,
                   threads=threads, verbose=verbose)


# Example usage
"""
input_fasta = ".fasta"
final_output = "representatives.fasta"
method = "cd-hit"  # Change to "mmseq2" to use MMseq2
filter_seq(method, input_fasta, final_output, similarity_threshold=0.7, word_size=5,
           coverage_long=0.8, coverage_short=0.8, threads=4, verbose=True)
"""

