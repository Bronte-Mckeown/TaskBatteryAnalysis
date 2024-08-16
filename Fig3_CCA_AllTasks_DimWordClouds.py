# -*- coding: utf-8 -*-
"""
This script creates word clouds to represent the X weights of the overall CCA model.

"""

import os
import pandas as pd
from scipy.stats import zscore
import ThoughtSpace.plotting as plotting

############################ Functions ##############################################


# projection function
def project_ccaPCA(cca_loadings, cca_list, pca_loadings, pca_ncomp):
    wordCloud_dict = {}

    for u, (cca_dim) in enumerate(cca_list):
        cca_dict = {}
        for i in range(pca_ncomp):
            z_pca_loadings = zscore(pca_loadings[f"PC{i+1}"])
            cca_dict["{}".format(i + 1)] = cca_loadings[cca_dim][i] * z_pca_loadings

        # dict to dataframe
        cca_df = pd.DataFrame(cca_dict, columns=cca_dict.keys())
        # sum
        ccasum = cca_df.sum(axis=1)

        # add to dataframe to dictionary for word cloud function
        wordCloud_dict["cca{}".format(u + 1)] = ccasum

        wordCloud_df = pd.DataFrame(wordCloud_dict)

    return wordCloud_df


############################ Data read ##############################################
# read in data
pca_loadings = pd.read_csv(
    "scratch\\results\\pca\\allTasks\\allTasksNoRotation5_24112023_10-19-32\\csvdata\\pca_loadings.csv"
)
cca_weights = pd.read_csv(
    "scratch\\results\\cca\\allTasks\\pca_weights_noRot_5ncomp.csv"
)

ncomp = 5
cca_cols = [col for col in cca_weights.columns if col.startswith("X_weight_")]

############################ Project  ##############################################
# call project function
coefficients_df = project_ccaPCA(cca_weights, cca_cols, pca_loadings, ncomp)
coefficients_df.index = pca_loadings["Unnamed: 0"]
if not os.path.exists("scratch\\results\\cca\\allTasks\\projectedWordClouds"):
    os.makedirs("scratch\\results\\cca\\allTasks\\projectedWordClouds")
# call word cloud function
plotting.save_wordclouds(
    coefficients_df, "scratch\\results\\cca\\allTasks\\projectedWordClouds"
)

print("end")
