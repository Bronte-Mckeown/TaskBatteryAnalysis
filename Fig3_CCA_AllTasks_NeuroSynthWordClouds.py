# -*- coding: utf-8 -*-
"""

@author: Bronte Mckeown

This script creates wordclouds from neurosynth terms for all CCA dimension brain maps.

Reads in: 
1) 1 csv file with component loadings (1 column for each component and 1 row for each item)
2) 1 csv file with labels for each item in the same order as the first csv file

Outputs:
1) 1 png word cloud image for each component 
    Optional:
2) 1 csv file for each component that shows the information used to create each word cloud
"""

# %% Import libraries

import os
import pandas as pd
import glob
import ThoughtSpace.plotting as plotting

# function to format words
def capitalize_and_remove_spaces(column):
    # Assuming 'column' is a list of strings
    modified_strings = [word.title().replace(' ', '') for word in column]
    return modified_strings

# %% Read in data
# read in  data
paths = sorted(glob.glob("scratch\\results\\cca\\allTasks\\neurosynth\\cca*_pca5_neurosynth_for_wordclouds.csv"))

# loop over paths (i.e., CCAs)
for iter, path in enumerate(paths):
    df1 = pd.read_csv(path, header = None)
    df1[0] = df1[0].str.strip()
    df1[0] = capitalize_and_remove_spaces(df1[0])
    df1.set_index(df1.columns[0], inplace=True)
    plotting.save_wordclouds(df1, f"scratch\\results\\cca\\allTasks\\Neurosynth\\wordclouds\\cca{iter+1}")
    print (f"wordcloud {iter+1} done")

print ("end")
