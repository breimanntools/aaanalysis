"""This is a script to test the SequenceFeature().get_feature_names() method ."""
from hypothesis import given, settings
import hypothesis.strategies as st
import pytest
import numpy as np
import random
import pandas as pd
import aaanalysis as aa
aa.options["verbose"] = False
