"""

This script runs overall PCA on the original sample.

It then projects original sample's PCA onto replication sample.

Finally, it runs overall PCA on the replication sample (just for checking).

All results saved to scratch/data/pca

"""

import pandas as pd
from ThoughtSpace.pca import basePCA
import numpy as np
from utils.cca import select_cols, distances, heatmap_allDim, otherTask_og_dist, otherTask_rep_dist

#################################### Data #####################################

# read in old lab data 
esq_data = pd.read_csv("scratch//data//esq_demo_forPCA.csv")
esq_data['Age'] = esq_data['Age'].astype(str) # set to str so not used in PCA

# read in new lab data
esq_rep = pd.read_csv("scratch//data//esq_demo_replication.csv")
esq_rep['Age'] = esq_rep['Age'].astype(str)

# remove 'replication' from end of strings as they need to match og data
replace_dict_tasks = {
                'HardMath-Rep': 'HardMath',
                'Memory-Rep': 'Memory',
                'GoNoGo-Rep': 'GoNoGo',
                'Documentary-Rep': 'Documentary'
                }
esq_rep = esq_rep.replace({'Task_name': replace_dict_tasks})

################################### PCA  ######################################

# calculate 5 component solution for both samples
n_comp = 5

# fit and transform PCA on original data
model_og = basePCA(n_components=n_comp, rotation=False)
model_og.fit(esq_data)
output_og = model_og.transform(esq_data)
model_og.save(path="scratch//results//pca//allTasks",pathprefix=f"allTasksNoRotation{n_comp}")

# project PCA from original data onto replication data
model_project = basePCA(n_components=n_comp, rotation=False)
model_project.fit(esq_data)
output_projected = model_project.transform(esq_rep)
model_project.save(path="scratch//results//pca//replication",pathprefix=f"projected_replicationNoRotation{n_comp}")

# fit and transform PCA on replication data
model_rep = basePCA(n_components=n_comp, rotation=False)
model_rep.fit(esq_rep)
output_rep = model_rep.transform(esq_rep)
model_rep.save(path="scratch//results//pca//replication",pathprefix=f"replicationNoRotation{n_comp}")

print ("end")

################################# Distances #######################################
# did for checking, not used in manuscript.
task_list = ['Memory',
             'Documentary',
             'HardMath', 'GoNoGo']

# set custom order for plotting heatmap below
custom_order = ['Memory',
             'Documentary',
             'HardMath', 'GoNoGo']

task_og = {}
task_rep = {}
task_distances_perDim = {}
task_distances_allDim = {}

# for every task name provided in task list
for task in task_list:
    # select rows matching that task name
    og_mask = output_og['Task_name'] == task
    og_rows = output_og[og_mask]

    rep_mask = output_projected['Task_name'] == task
    rep_rows = output_projected[rep_mask]

    # select scaled PCA and grad columns
    og = select_cols(og_rows, 'PCA')
    rep = select_cols(rep_rows, 'PCA')

    # average
    og_avg = og.mean().to_frame().T
    rep_avg = rep.mean().to_frame().T

    # add to dicts
    task_og[task] = og_avg
    task_rep[task] = rep_avg

    # calculate distances at per-dim level and across dims
    perdim_dist, alldim_dist = distances(og_avg, rep_avg)
    task_distances_perDim[task] = perdim_dist # store in dict
    task_distances_allDim[task] = alldim_dist # store in dict


other_rep_distances_perDim, other_rep_distances_allDim = otherTask_rep_dist(task_og, task_rep)
other_og_distances_perDim, other_og_distances_allDim = otherTask_og_dist(task_og, task_rep)

############################### Heatmaps ################################################
# Heatmaps (not used in manuscript, just used for checking)

# ACROSS PCA dimensions
# provide length of task list, the custom order for plotting, the true task distances, other task distances, and filename and outputdir string
heatmap_allDim(len(task_list), custom_order, task_distances_allDim, other_rep_distances_allDim,
               "Replication","Original",
               "Distance between Original and Replication",
               "pca_otherRep",
              "scratch//results//pca//replication")

heatmap_allDim(len(task_list), custom_order, task_distances_allDim, other_og_distances_allDim,
                "Original","Replication",
                "Distance between Replication and Original",
                "pca_otherOG",
               "scratch//results//pca//replication")

print ("end")
