"""
This is a script for the plotting class of AAclust.
"""
import time
import pandas as pd
import numpy as np


# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe


# I Helper Functions


# II Main Functions
class AAclustPlot:
    """Plot results of AAclust analysis"""
    def __int__(self, model):
        self.model = model


    @staticmethod
    def eval():
        """Plot eval output of BIC, CH, SC"""

    def center(self):
        """PCA plot of clustering with centers highlighted"""

    def medoids(self):
        """PCA plot of clustering with medoids highlighted"""

    @staticmethod
    def correlation():
        """Heatmap for correlation"""
