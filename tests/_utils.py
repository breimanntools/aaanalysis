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
FOLDER_PROJECT += "tests" + SEP
FOLDER_RESULTS = _folder_path(FOLDER_PROJECT, 'results')
FOLDER_DATA = _folder_path(FOLDER_PROJECT, 'data')

# General Columns and strings
COL_SCALE_ID = "scale_id"
COL_CAT = "category"
COL_SUBCAT = "subcategory"
COL_NAME = "scale_name"
COL_SCALE_DESCRIPTION = "scale_description"
COL_SUBCAT_DESCRIPTION = "subcategory_description"
COL_COUNT = "n_scales"
COL_PROPERTY = "property"
COLS_SCALE_INFOS = [COL_SCALE_ID, COL_CAT, COL_SUBCAT, COL_NAME, COL_SCALE_DESCRIPTION]
