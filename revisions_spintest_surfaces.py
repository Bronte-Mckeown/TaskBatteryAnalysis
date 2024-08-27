# -*- coding: utf-8 -*-
"""task_battery_spin_test_surf.ipynb

"""

# Commented out IPython magic to ensure Python compatibility.

# %cd /content/drive/MyDrive/task_battery/

import nibabel as nib
from neuromaps import transforms
from neuromaps.resampling import resample_images
import neuromaps.resampling as resampling
from neuromaps.datasets import fetch_atlas
from nilearn import image as nimg
import pandas as pd
import os

os.chdir(r"W:/Y_Kingston/task_battery")

def applymask(img, maskimg):
    """
    Return masked image.

    Args:
        img (nibabel image object): Image to apply mask to.
        maskimg (nibabel image object): Mask image to apply to image.

    Returns:
        Masked img.
    """
    # try to apply without reshaping
    try:
        return nimg.math_img('a*b',a=img, b=maskimg) #element wise multiplication - return the resulting map
    # if shapes don't match, reshape img
    except ValueError:
        print('Shapes of images do not match')
        print(f'mask-image shape: {maskimg.shape}, image shape {img.shape}')
        print('Reshaping image to mask-image dimensions...')
        img = nimg.resample_to_img(source_img=img,target_img=maskimg,interpolation='nearest')
        return nimg.math_img('a*b',a=img, b=maskimg) #element wise multiplication - return the resulting map

task_maps_folder = r'task_maps'
gradient_maps_folder = r'gradient_maps'

task_maps = os.listdir(task_maps_folder)
gradient_maps = os.listdir(gradient_maps_folder)

comb= nib.load('combinedmask_cortical_subcortical.nii')

from neuromaps import stats
from neuromaps import datasets, images, nulls, resampling
import matplotlib.pyplot as plt


results = []
null_dists = {}

for task in task_maps:
  for gradient in gradient_maps:
    gradmap = nib.load(f'gradient_maps/{gradient}')
    grad_masked = applymask(gradmap,comb)
    gradient_srf = transforms.mni152_to_fslr(grad_masked, '32k', method='nearest')

    taskmap = nib.load(f'task_maps/{task}')
    task_srf = transforms.mni152_to_fslr(taskmap, '32k', method='nearest')

    rotated = nulls.alexander_bloch(task_srf, atlas='fslr', density='32k', n_perm=1000, seed=1420)

    corr, pval, nulls_dist  = stats.compare_images(task_srf, gradient_srf, nulls=rotated, return_nulls=True, metric= 'spearmanr')

    null_dists[task] = nulls_dist

    results.append({
        'task': task,
        'gradient': gradient,
        'corr_surf': corr,
        'pval_surf': pval,
        })



results_df = pd.DataFrame(results)
results_to_save = pd.DataFrame(results)

results_to_save['task'] = results_to_save['task'].apply(lambda x: os.path.basename(x).split('.')[0].split('_')[0])
results_to_save['gradient'] = results_to_save['gradient'].apply(lambda x: os.path.basename(x).split('.')[0].split('_')[0])


results_to_save.to_csv('task_spin_results.csv', index=False)

import pickle

# Save null_dists to a file
with open('task_null_dists.pkl', 'wb') as f:
    pickle.dump(null_dists, f)

# ... later, to load the saved null_dists ...
with open('task_null_dists.pkl', 'rb') as f:
    loaded_null_dists = pickle.load(f)

import matplotlib.pyplot as plt

# Assuming you have 5 gradients and 14 tasks
fig, axes = plt.subplots(14, 5, figsize=(15, 15))  # Adjust figsize as needed

for i, task in enumerate(results_df['task'].unique()):
    for j, gradient in enumerate(results_df['gradient'].unique()):
        corr = results_df[(results_df['task'] == task) & (results_df['gradient'] == gradient)]['corr_surf'].values[0]
        nulls = null_dists[task]  # Use the matching key to access null distribution

        axes[i, j].hist(nulls, bins=20, density=True)
        axes[i, j].axvline(corr, ls="--", color="r")

        if i == 0:
            axes[i, j].set_title(gradient.split('_')[0])
        if j == 0:
            axes[i, j].set_ylabel(task.split('_')[0])

plt.tight_layout()
plt.savefig('all_tasks_gradients.svg')
plt.show()

