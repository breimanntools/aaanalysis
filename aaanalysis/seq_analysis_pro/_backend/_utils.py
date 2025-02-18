"""This is a backend script for making temporary folders/files and managing commands"""
import os
import shutil
import tempfile
import subprocess
import glob
import pandas as pd

from aaanalysis import utils as ut


def make_temp_dir(prefix=None, remove_existing=False):
    """ Creates a temporary directory within the script's running directory. If 'remove_existing' is True,
    any existing directory with the same name will be removed and recreated"""
    temp_dir_base = tempfile.gettempdir()
    # Remove existing directories with the same prefix
    if remove_existing:
        for temp_dir in glob.glob(os.path.join(temp_dir_base, prefix + '*')):
            shutil.rmtree(temp_dir)
    # Create a unique temporary directory with the prefix
    temp_dir_path = tempfile.mkdtemp(prefix=prefix, dir=temp_dir_base)
    return temp_dir_path


def remove_temp(path=None):
    """Removes the specified temporary directory and all its contents."""
    if path is None or not os.path.exists(path):
        return  # Skip
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)


def run_command(cmd=None, verbose=False, temp_dir=None):
    """Run command using subprocess and delete temporary folder if fails"""
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with error: {err.decode('utf-8')}")
        if verbose:
            ut.print_out(out.decode('utf-8'))
    except Exception as e:
        remove_temp(path=temp_dir)
        str_error = f"Command with following arguments failed: {cmd}."
        if verbose:
            str_error += f"\n\tError message: {e}"
        raise RuntimeError(str_error)


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