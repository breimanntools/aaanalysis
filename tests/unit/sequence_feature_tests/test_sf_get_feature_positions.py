"""This is a script to test the SequenceFeature().get_feature_positions() method ."""
from hypothesis import given, settings, strategies as st
import pytest
import numpy as np
import random
import pandas as pd
import aaanalysis as aa
aa.options["verbose"] = False

def get_random_features(n_feat=100):
    """"""
    sf = aa.SequenceFeature()
    features = sf.get_features()
    return random.sample(features, n_feat)
