"""This script tests the TreeModel.predict_proba() method."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa
import hypothesis.extra.numpy as npst
aa.options["verbose"] = False