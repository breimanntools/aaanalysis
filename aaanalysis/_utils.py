#! /usr/bin/python3
"""
Config with folder structure
"""
import os
import platform
from pathlib import Path

import os
import platform


# Helper Function
def _folder_path(super_folder, folder_name):
    """Modification of separator (OS depending)"""
    path = os.path.join(super_folder, folder_name + SEP)
    return path


# Folder
SEP = "\\" if platform.system() == "Windows" else "/"
FOLDER_PROJECT = os.path.dirname(os.path.abspath(__file__))
FOLDER_DATA = _folder_path(FOLDER_PROJECT, 'data')
URL_DATA = "https://github.com/breimannlab/aaanalysis/tree/master/aaanalysis/data/"

# Default data for protein analysis
STR_SCALES = "scales"
STR_SCALES_RAW = "scales_raw"
STR_SCALE_CAT = "scale_categories"
