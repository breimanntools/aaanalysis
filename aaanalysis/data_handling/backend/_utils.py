"""This is a backend script for making temporary folders/files and managing commands"""
import os
import shutil
import subprocess

from aaanalysis import utils as ut


NAME_TEMP_FOLDER = "_temp_dir"


def make_temp_dir(dir_name=None, remove_existing=True):
    """ Creates a temporary directory within the script's running directory. If 'remove_existing' is True,
    any existing directory with the same name will be removed and recreated"""
    if dir_name is None:
        dir_name = NAME_TEMP_FOLDER
    current_directory = os.getcwd()
    temp_dir_path = os.path.join(current_directory, dir_name)
    if remove_existing and os.path.exists(temp_dir_path):
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path, exist_ok=True)
    return temp_dir_path


def remove_temp(path=None):
    """Removes the specified temporary directory and all its contents."""
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
