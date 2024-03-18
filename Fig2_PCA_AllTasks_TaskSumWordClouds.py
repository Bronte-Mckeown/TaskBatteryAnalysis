# -*- coding: utf-8 -*-
"""

@author: Bronte Mckeown

This script creates wordclouds from task emmeans from LMM comparing PCA scores.

"""

# %% Import libraries

import os
import pandas as pd
import ThoughtSpace.plotting as plotting

# %% Read in data
# read in coefficients data
folder_path = "C:\\Users\\bront\\Documents\\CanadaPostdoc\\MegaProject\\TaskBatteryAnalysis\\scratch\\results\\pca\\lmm\\"
loadings_file = "pcascores_emmeans.csv"
loadings_path = folder_path + loadings_file
df1 = pd.read_csv(loadings_path, header = None)
df1.set_index(df1.columns[0], inplace=True)

plotting.save_wordclouds(df1, "scratch//results//pca//lmm")