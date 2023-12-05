"""
This is a script for testing the  aa.AAclustPlot().medoids() method.
"""
import hypothesis.strategies as some
from hypothesis import given, settings
import pytest
from matplotlib import pyplot as plt
import numpy as np
import aaanalysis as aa
