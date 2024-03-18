"""

This script runs overall PCA and CCA on task battery data minus the 4 tasks which it will predict.

It then uses this 10-task CCA to predict Y values for the 4 tasks in task list (new task data).

This is done to demonstrate more conservative test of accuracy (supplementary analysis).

"""

import pandas as pd
import numpy as np
from ThoughtSpace.pca import basePCA # import for PCA
from utils.cca import prepare_X, prepare_Y, train_CCA, tasklist_prediction, otherTask_pred_dist, heatmap_allDim, ttests, print_ttests

#################################### Data #####################################

## MDES
# read in original (og) lab data 
# this data is at observation level; 38 obs * 190 PS = 7220 obs
esq_data1 = pd.read_csv("scratch//data//esq_demo_forPCA.csv")
esq_data1['Age'] = esq_data1['Age'].astype(str) # set to str so not inc. in PCA

# remove all rows related to the 4 replication tasks
conditions = ~esq_data1['Task_name'].isin(['Memory',
             'Documentary',
             'HardMath', 'GoNoGo'])

esq_data = esq_data1[conditions]

# read in new lab data
# this data is also at observation level; 11 obs (except 1) * 95 PS = 1044 obs
esq_rep = pd.read_csv("scratch//data//esq_demo_replication.csv")
esq_rep['Age'] = esq_rep['Age'].astype(str) # set to str so not inc. in PCA

## GRADIENT SCORES (same data in both instances, just different lengths)
# og = trial level, replication = task level
# So in the og data, there are 27 obs (38-11 obs) per PS
# before overall training CCA, this data is scaled and the scaler is saved
# and used to scale the replication data, where there are 4 obs per PS

# read in grad data matching og data length for training CCA
grad_data1 = pd.read_csv("scratch//data//gradscores_spearman_combinedmask_cortical_subcortical_forCCA.csv")
# remove all rows related to 4 replication tasks
conditions = ~grad_data1['Task_name'].isin(['Memory',
             'Documentary',
             'HardMath', 'GoNoGo'])

grad_data = grad_data1[conditions]

# read in grad data matching new data length for assessing distance of predictions
grad_rep = pd.read_csv("scratch//data//gradscores_spearman_combinedmask_cortical_subcortical_forCCA_prediction.csv")

n_grads = 5 # set number of grads (used in heatmap)

############################### PCA training  ###################################

# calculate 5 component solution on og 10-task mDES data (5130 obs)
n_comp = 5

model = basePCA(n_components=n_comp, rotation = False)
output = model.fit_transform(esq_data) # output is same as esq_data + 5 cols for PCA scores
# model.save(path="scratch//results//pca//allTasks",pathprefix=f"allTasks{n_comp}")

############################# CCA ###############################################

# Prepare PCA and grad data for CCA and save scalers for new data
pca_scaled, og_pca_scaler = prepare_X(output)
grad_scaled, og_grad_scaler = prepare_Y(grad_data)

## call train_CCA on all og true data
# save model, variates, summed variates, scalers used on variates before summing, and cca correlations
ca, X_c, Y_c, X_Y_sum, X_c_scaler, Y_c_scaler, corrs = train_CCA(pca_scaled, grad_scaled, n_comp, save=False)

############################### PCA projecting  ###################################

# apply already-trained PCA to new replication data
output_rep = model.transform(esq_rep)

############################### Prepare data for prediction ########################
# prepare PCA for prediction
# scales projected PCA data by mean and std of og data
# 95 PS * 11 = 1044
pca_scaled_rep = prepare_X(output_rep, scaler = og_pca_scaler)[0]

# prepare grad data for prediction
# scales 4-task gradient coordinates according by mean and std of all 14 coordinates
grad_scaled_rep = prepare_Y(grad_rep, scaler = og_grad_scaler)[0]

# remove 'replication' from end of strings as they need to match og data
replace_dict_tasks = {
                'HardMath-Rep': 'HardMath',
                'Memory-Rep': 'Memory',
                'GoNoGo-Rep': 'GoNoGo',
                'Documentary-Rep': 'Documentary'
                }
pca_scaled_rep = pca_scaled_rep.replace({'Task_name': replace_dict_tasks})

# set up task list for predictions (all 4 tasks in replication sample)
task_list = ['Memory',
             'Documentary',
             'HardMath', 'GoNoGo']

# set custom order for plotting heatmap below
custom_order = ['Memory',
             'Documentary',
             'HardMath', 'GoNoGo']

# call task list prediction function
# returns predictions of grad scores for each obs, distance from true for each gradient, and distance from true across all gradients
true_task_predictions, true_task_distances_perDim, true_task_distances_allDim = tasklist_prediction(task_list, pca_scaled_rep, grad_scaled_rep, ca)

# save true_task_predictions for 3d plotting in R
data_list = []
for task_name, values in true_task_predictions.items():
    for value in values:
        data_dict = {'Task_name': task_name}
        for i, column_value in enumerate(value):
            data_dict[f'Zgradient{i+1}'] = column_value
        data_list.append(data_dict)

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(data_list)

df.to_csv("scratch//results//cca//allTasks//reptasks_cca_predictgrads_10tasks.csv", index = False)
# grad_scaled_rep.to_csv("scratch//results//cca//allTasks//reptasks_cca_realgrads.csv", index = False)

##################### Calculate distances between tasks #############################
# for each task in task_predictions:

# call otherTask_pred_dist to:
# calculate 1) distance between that tasks's real gradient value
# and that task's predictions and 2) distances between that task's real gradient value and the 
# other tasks' predictions.

otherTask_pred_dist_perDim, otherTask_pred_dist_allDim = otherTask_pred_dist(true_task_predictions, grad_scaled_rep)

############################### T-tests ###################################################

other_pred_dist_ttest,other_pred_dist_ttest_allDim =  ttests(true_task_distances_perDim, true_task_distances_allDim,otherTask_pred_dist_perDim, otherTask_pred_dist_allDim)
print_ttests(other_pred_dist_ttest,other_pred_dist_ttest_allDim)

############################### Plotting results ###########################################

# save data out for R (bar graphs)
for task in true_task_distances_allDim.keys():
    print (task)

    #Create a dataframe for the current key
    df1 = pd.DataFrame({
    'task': [task] * len(true_task_distances_allDim[task]),
    'distance': true_task_distances_allDim[task]
    })

    for other_task in true_task_distances_allDim.keys():
        if task != other_task:
            print (other_task)

            df2 = pd.DataFrame({
            'task': [other_task] * len(otherTask_pred_dist_allDim[task][other_task]),
            'distance': otherTask_pred_dist_allDim[task][other_task]
            })

            # concat
            df1 = pd.concat([df1, df2], axis = 0)

    df1.to_csv(f'scratch//results//cca//allTasks//{task}_dists_for_bargraphs_10tasks.csv', index=False)

# Heatmaps

# ACROSS all gradients
# provide length of task list, the custom order for plotting, the true task distances, other task distances, and filename string
heatmap_allDim(len(task_list), custom_order, true_task_distances_allDim, otherTask_pred_dist_allDim, 
               "Predicted", "True",
                "otherPred_noRotation_10tasks","scratch//results//cca//allTasks")


print ("end")