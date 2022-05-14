#! /usr/bin/python3
"""
Config with folder structure
"""
import os
import platform
from pathlib import Path


# Helper Function
def _folder_path(super_folder, folder_name):
    """Modification of separator (OS depending)"""
    path = os.path.join(super_folder, folder_name + SEP)
    return path


# Folder
SEP = "\\" if platform.system() == "Windows" else "/"
FOLDER_PROJECT = str(Path(__file__).parent.parent).replace('/', SEP) + SEP
FOLDER_DATA = _folder_path(FOLDER_PROJECT, 'data')