"""

This script reads in X and Y variates (and summed) from the overall CCA of
original sample (N=190).

It then uses the summed variates to make task selection.

"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

#################################### Data #####################################

# read in lab data 
cca_data = pd.read_csv("scratch//results//cca//allTasks//allTask_variates.csv")

# set number of tasks you want and number of CCA components you want to include
n_tasks = 4
n_comp = 4

############################## Selection  ####################################
#  these summed variates were scaled first, then summed, then averaged

# # # Calculate the mean summed CCA coordinates for each task
x_task_mean_dict = {}
y_task_mean_dict = {}
sum_task_mean_dict = {}

unique_task_names = cca_data['Task_name'].unique() # store task names in list

# select X, Y and summed cols from cca_data
X_cols = [col for col in cca_data.columns if col.startswith("pca")]
Y_cols = [col for col in cca_data.columns if col.startswith("grad")]
sum_cols = [col for col in cca_data.columns if col.startswith("sum")]

# loop over task names and calculate the task means for X, Y and summed variates
for task_label in unique_task_names:
    task_indices = cca_data[cca_data['Task_name'] == task_label].index
    x_task_mean_dict[task_label] = np.mean(cca_data.loc[task_indices, X_cols].iloc[:, :n_comp], axis=0)
    y_task_mean_dict[task_label] = np.mean(cca_data.loc[task_indices, Y_cols].iloc[:, :n_comp], axis=0)
    sum_task_mean_dict[task_label] = np.mean(cca_data.loc[task_indices, sum_cols].iloc[:, :n_comp], axis=0)

# convert to dataframes
x_task_mean_df = pd.DataFrame(x_task_mean_dict).T
y_task_mean_df = pd.DataFrame(y_task_mean_dict).T
sum_task_mean_df = pd.DataFrame(sum_task_mean_dict).T

# select 4 most extreme tasks on first two dimensions
# Identify the tasks with the highest and lowest values for the first dimension
most_positive_tasks_dim1 = sum_task_mean_df.nlargest(1, 'sum_CCAloading_1').index
most_negative_tasks_dim1 = sum_task_mean_df.nsmallest(1, 'sum_CCAloading_1').index

# Identify the tasks with the highest and lowest values for the second dimension
most_positive_tasks_dim2 = sum_task_mean_df.nlargest(1, 'sum_CCAloading_2').index
most_negative_tasks_dim2 = sum_task_mean_df.nsmallest(1, 'sum_CCAloading_2').index

# Print or use the selected tasks as needed
print("Most Positive Task (Dim1):", most_positive_tasks_dim1)
print("Most Negative Task (Dim1):", most_negative_tasks_dim1)
print("Most Positive Task (Dim2):", most_positive_tasks_dim2)
print("Most Negative Task (Dim2):", most_negative_tasks_dim2)

print ("end")
