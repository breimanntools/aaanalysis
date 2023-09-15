"""
Config with folder structure
"""
import os
import platform

# Import utility functions for specific purposes
from aaanalysis._utils._utils_constants import *
from aaanalysis._utils._utils_check import *
from aaanalysis._utils._utils_output import *

# Import utility function for specific modules
from aaanalysis._utils.utils_aaclust import *
from aaanalysis._utils.utils_cpp import *


# I Folder structure
def _folder_path(super_folder, folder_name):
    """Modification of separator (OS depending)"""
    path = os.path.join(super_folder, folder_name + SEP)
    return path


SEP = "\\" if platform.system() == "Windows" else "/"
FOLDER_PROJECT = os.path.dirname(os.path.abspath(__file__))
FOLDER_DATA = _folder_path(FOLDER_PROJECT, '_data')
URL_DATA = "https://github.com/breimanntools/aaanalysis/tree/master/aaanalysis/data/"


# II MAIN FUNCTIONS
