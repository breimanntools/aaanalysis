"""This is a script for the for backend of the fasta reader function"""
import pandas as pd


# II Main Functions
def get_entries_from_fasta(file_path=None, col_id="entry", col_seq="sequence", col_db=None, sep=None):
    """Read information from FASTA file and convert to DataFrame"""
    list_entries = []
    dict_current_entry = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if dict_current_entry:
                    # Save the previous sequence before starting a new one
                    list_entries.append(dict_current_entry)
                # Parse the header and prepare a new entry
                list_info = line[1:].split(sep)
                if col_db and len(list_info) > 1:
                    dict_current_entry = {col_id: list_info[1], col_seq: "", col_db: list_info[0]}
                    list_info = list_info[1:]
                else:
                    dict_current_entry = {col_id: list_info[0], col_seq: ""}
                if len(list_info) > 1:
                    for i in range(1, len(list_info[1:])+1):
                        dict_current_entry[f'info{i}'] = list_info[i]
            else:
                dict_current_entry[col_seq] += line
        if dict_current_entry:
            list_entries.append(dict_current_entry)
    df = pd.DataFrame(list_entries)
    return df


def save_entries_to_fasta(df_seq=None, file_path=None, col_id="entry", col_seq="sequence",
                          sep="|", col_db=None, cols_info=None):
    """Write sequence DataFrame to a FASTA file."""
    with open(file_path, 'w') as fasta:
        for _, row in df_seq.iterrows():
            header_parts = [str(row.get(col)) for col in [col_db, col_id] + (cols_info if cols_info is not None else []) if
                            col in row and pd.notna(row[col])]
            header = sep.join(header_parts)
            sequence = row[col_seq]
            fasta.write(f">{header}\n{sequence}\n")

