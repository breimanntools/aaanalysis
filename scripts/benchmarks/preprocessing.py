"""
This is a script for ...
"""
import time
import pandas as pd
from Bio import SeqIO
import numpy as np

import scripts._utils as ut


# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe


# I Helper Functions
def _split_seq_label(seq_label=None):
    """"""
    if "0" in seq_label and "1" in seq_label:
        i0 = seq_label.find("0")
        i1 = seq_label.find("1")
        i = min(i0, i1)
    elif "0" in seq_label:
        i = seq_label.find("0")
    elif "1" in seq_label:
        i = seq_label.find("1")
    else:
        raise ValueError("label not in sequence")
    seq, labels = seq_label[0:i], seq_label[i:]
    if len(seq) != len(labels):
        raise ValueError(f"seq {len(seq)} and labels {len(labels)} do not match")
    return seq, labels

def read_fasta(fasta=None, label=False):
    """"""
    list_data = []
    for record in SeqIO.parse(fasta, "fasta"):
        entry = record.name
        if label:
            print(entry)
            seq, labels = _split_seq_label(seq_label=str(record.seq))
        else:
            seq = str(record.seq)
            labels = np.NaN
        list_data.append([entry, seq, labels])
    df = pd.DataFrame(list_data, columns=["entry", "sequence", "label"])
    return df

# II Main Functions
def pre_processing_data():
    """"""
    COLUMNS = ["entry", "sequence", "label"]
    FOLDER_OUT = ut.FOLDER_DATA + "benchmarks" + ut.SEP
    FOLDER_IN = FOLDER_OUT + "material" + ut.SEP

    # Amyloidogenic
    """
    file = "AMYLO_SEQ"
    df = pd.read_csv(FOLDER_IN + file + "_.tsv", sep="\t")
    df["label"] = [int(x == "positive") for x in df["label"]]
    df["entry"] = [f"AMYLO{n}" for n, _ in enumerate(df["label"])]
    df = df[COLUMNS]
    df.to_csv(FOLDER_OUT + "AMYLO_SEQ.tsv", index=False, sep="\t")
    # CAPSID
    df_pos = read_fasta(fasta=FOLDER_IN + "CAPSID_SEQ_positive.txt")
    df_pos["label"] = 1
    df_neg = read_fasta(fasta=FOLDER_IN + "CAPSID_SEQ_negative.txt")
    df_neg["label"] = 0
    df = pd.concat([df_pos, df_neg])
    df["entry"] = [f"Protein{n}" for n, _ in enumerate(df["label"])]
    df = df[COLUMNS]
    print(df)
    df.to_csv(FOLDER_OUT + "CAPSID_SEQ.tsv", index=False, sep="\t")
    # DISULFIDE
    file = "DISULFIDE_SEQ"
    df_pos = read_fasta(fasta=FOLDER_IN + f"{file}_positive.txt")
    df_pos["label"] = 1
    df_neg = read_fasta(fasta=FOLDER_IN + f"{file}_negative.txt")
    df_neg["label"] = 0
    df = pd.concat([df_pos, df_neg])
    df.to_csv(FOLDER_OUT + file + ".tsv", index=False, sep="\t")
    # LDR
    file = "LDR_AA"
    df = read_fasta(fasta=FOLDER_IN + "LDR_AA.txt", label=True)
    df["label"] = [",".join([x for x in l]) for l in df["label"]]
    df.to_csv(FOLDER_OUT + file + ".tsv", index=False, sep="\t")
    # LOCATION
    file = "LOCATION_SEQ"
    df = pd.read_csv(FOLDER_IN + file + "_.tsv", sep="\t")
    df["entry"] = [f"Protein{n}" for n, _ in enumerate(df["label"])]
    df = df[COLUMNS]
    df.to_csv(FOLDER_OUT + file + "_MULTI.tsv", sep="\t", index=False)
    # RNA BINDING
    file = "RNABIND_AA"
    df = pd.read_csv(FOLDER_IN + file + ".tsv", sep="\t")
    df.columns = COLUMNS
    df["label"] = [",".join([x for x in l]) for l in df["label"]]
    df.to_csv(FOLDER_OUT + file + ".tsv", index=False, sep="\t")
    """
    # SOLUBLE
    file = "SOLUBLE_SEQ"
    df = read_fasta(fasta=FOLDER_IN + file + ".txt")
    df.columns = COLUMNS
    f = lambda s: 1 if s.split("_")[1] == "S" else 0
    df["label"] = [f(x) for x in df["entry"]]
    df.to_csv(FOLDER_OUT + file + ".tsv", index=False, sep="\t")


# III Test/Caller Functions


# IV Main
def main():
    t0 = time.time()
    pre_processing_data()
    t1 = time.time()
    print("Time:", t1 - t0)


if __name__ == "__main__":
    main()
