# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:21:26 2023

@author: bront
"""

import nibabel as nib
import numpy as np
from nilearn import datasets,  maskers, plotting
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# collect all inputfiles using glob
inputfiles = glob.glob('scratch\\results\\cca\\allTasks\\projectedBrainMaps\\cca*_ncomp_pca5.nii.gz')# get all maps

# read in gradient mask
mask_path = "scratch\\data\\gradients\\combinedmask_cortical_subcortical.nii.gz"
maskimg = nib.load(mask_path)

# Define Yeo Networks
yeo = datasets.fetch_atlas_yeo_2011()

# Create a masker to extract data within Yeo networks
masker = maskers.NiftiLabelsMasker(labels_img=yeo.thick_17,mask_img = maskimg,
                                       standardize=False)

# set network labels for 7 networks
network_labels = ["VisCent",
    "VisPeri",
    "SM-a",
    "SM-b",
    "DAN-a",
    "DAN-b",
    "VAN-a",
    "VAN-b",
    "LMB-b",
    "LMB-a",
    "FPN-a",
    "FPN-b",
    "FPN-c",
    "DMN-a",
    "DMN-b",
    "DMN-c",
    "TempPar"
]

# create dictionary to add results to
results_dict = {}

for path in inputfiles:
    # extract task id
    splits = path.split('\\')

    taskid = [i for i in splits if 'ncomp' in i][0]
    
    splits = taskid.split('_')
    
    taskid = [i for i in splits if 'cca' in i][0]

    # load image
    state = nib.load(path)
    
    # use yeo mask to extract values
    mean_values = masker.fit_transform(state)
    mean_values = mean_values.mean(axis = 0)
    
    # z-score [not sure if needed]
    zmean_values = zscore(mean_values)
    
    # save as dataframe with labels
    mean_values_df = pd.DataFrame({
        'yeo_network': network_labels,
        'mean_value': mean_values
    })
    
    # add dataframe to dictionary
    results_dict[taskid] = mean_values_df

# Concatenate all DataFrames in grad_dict into a single DataFrame
result_df = pd.concat(results_dict.values(), keys=results_dict.keys()).reset_index(level=1, drop=True).reset_index()

# Rename the columns if needed
result_df.columns = ['ccaDim','yeo_network', 'mean_value']

result_df.to_csv('scratch\\results\\cca\\allTasks\\projectedBrainMaps\\ccaDim_yeo_avgs.csv', index = False)