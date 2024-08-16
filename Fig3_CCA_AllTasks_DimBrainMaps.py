# -*- coding: utf-8 -*-
"""

This script multiplies gradient images by CCA weights to create brain maps per CCA dimension.

"""

# %% Libraries

import nibabel as nib
import pandas as pd
from nilearn import plotting
import numpy as np
import glob
from scipy.stats import zscore
import re
from nilearn import image as nimg
import os


# %% Functions
# function to do projection
def project_gradCCA(cca_values, gradients, title, output, niftioutput):
    cca_list = []

    for cca, gradient in zip(cca_values, gradients):
        # Z-score the gradient image
        gradient_zscored = zscore(gradient.get_fdata(), axis=None, ddof=1)

        # Multiply by the CCA value
        new = gradient_zscored * cca
        cca_list.append(new)

    # Sum the CCA-weighted gradient images
    cca_sum = np.sum(cca_list, axis=0)

    # Create a Nifti object with the CCA-projected gradient image
    ccaimage = nib.Nifti1Image(cca_sum, gradient.affine, gradient.header)

    # Save the Nifti image to file
    nib.save(ccaimage, niftioutput)

    # Plot the CCA-projected gradient image as a glass brain
    plotting.plot_glass_brain(
        ccaimage, plot_abs=False, colorbar=True, output_file=output, title=title
    )


# %% Read in gradient data

# read in gradient images
gradient_paths = sorted(glob.glob("scratch\\data\\gradients\\gradient*.nii.gz"))

# read in mask
mask_path = "scratch\\data\\gradients\\combinedmask_cortical_subcortical.nii.gz"
maskimg = nib.load(mask_path)

gradientimgs = []
for gradient_path in gradient_paths:
    gradientimg = nib.load(gradient_path)
    # apply mask
    multmap = nimg.math_img("a*b", a=gradientimg, b=maskimg)
    # add to list
    gradientimgs.append(multmap)

# %% Read in CCA data

cca_weights = pd.read_csv(
    "scratch\\results\\cca\\allTasks\\grad_weights_noRot_5ncomp.csv"
)
ncomp = 5
if not os.path.exists("scratch\\results\\cca\\allTasks\\projectedBrainMaps"):
    os.makedirs("scratch\\results\\cca\\allTasks\\projectedBrainMaps")
for cca_dim in range(ncomp):
    # stores cca dimension variables in a list
    cca_values = cca_weights[f"Y_weight_{cca_dim+1}"].tolist()
    # calls projection function
    project_gradCCA(
        cca_values,
        gradientimgs,
        f"Thought-Brain Dimension {cca_dim+1}",
        f"scratch\\results\\cca\\allTasks\\projectedBrainMaps\\cca{cca_dim+1}_ncomp_pca{ncomp}.png",
        f"scratch\\results\\cca\\allTasks\\projectedBrainMaps\\cca{cca_dim+1}_ncomp_pca{ncomp}.nii.gz",
    )
