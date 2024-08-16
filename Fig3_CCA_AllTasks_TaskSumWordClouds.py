# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:17:30 2021

@author: Bronte Mckeown

This script creates wordclouds from task emmeans from LMM comparing CCA summed variates.

"""

# %% Import libraries

import os
import pandas as pd

import ThoughtSpace.plotting as plotting

# %% Read in data
# read in coefficients data
folder_path = "scratch\\results\\cca\\lmm\\"
loadings_file = "summed_cca_emmeans.csv"
loadings_path = folder_path + loadings_file
df1 = pd.read_csv(loadings_path, header=None)

df1.set_index(df1.columns[0], inplace=True)

# call word cloud function
plotting.save_wordclouds(df1, "scratch//results//cca//lmm")
