"""

This script runs hold-out PCA and CCA on task battery data.

Holds one task out at a time.

PCA with no rotation.

"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import os

# import custom functions from utils
from utils.cca import (
    prepare_X,
    prepare_Y,
    train_CCA,
    test_CCA,
    prepare_variates,
    save_variates,
    heldout_pca,
)

from sklearn.model_selection import LeaveOneGroupOut

################################### Data #####################################
if not os.path.exists("scratch//results//cca//holdOut"):
    os.makedirs("scratch//results//cca//holdOut")
# read in lab data
esq_data = pd.read_csv("scratch//data//esq_demo_forPCA.csv")
esq_data["Age"] = esq_data["Age"].astype(str)  # set to str so not used in PCA

# read in grad data
grad_data = pd.read_csv(
    "scratch//data//gradscores_spearman_combinedmask_cortical_subcortical_forCCA.csv"
)
n_grads = 5  # set number of grads (used in heatmap)

print("data read in")

############################### Held out PCA and CCA ##########################
# set number of components for PCA and CCA
pca_n_comp = 5
ca_n_comp = 5

# # initiate leave one group out class
cv = LeaveOneGroupOut()
Tasklabels, Taskindices = np.unique(esq_data["Task_name"], return_inverse=True)

# loop over training datasets
# train and test = indices of training and testing data
for train, test in cv.split(esq_data, grad_data, Taskindices):

    # 1. Select ESQ data for PCA
    esq_train = esq_data.iloc[train]  # select training ESQ data
    esq_test = esq_data.iloc[test]  # for saving prefix! (i.e., task name)
    test_task = esq_test["Task_name"].unique().tolist()[0]  # save task name of test

    # 2. train PCA
    print("about to PCA")
    output_train, output_test = heldout_pca(
        esq_data,
        esq_train,
        test_task,
        train,
        test,
        pca_n_comp,
        rotation=False,
        save=False,
    )

    print(f"done {test_task} PCA")

    # 3. Prepare X (scale) for CCA
    pca_train_scaled, pca_train_scaler = prepare_X(output_train)
    pca_test_scaled = prepare_X(output_test, pca_train_scaler)[0]

    # 4. Select Grad Data for scaling
    grad_train = grad_data.loc[grad_data["Task_name"] != test_task]
    grad_test = grad_data.loc[grad_data["Task_name"] == test_task]

    # 5. Prepare Y for CCA
    grad_train_scaled, grad_train_scaler = prepare_Y(grad_train)
    grad_test_scaled = prepare_Y(grad_test, grad_train_scaler)[0]

    # 6. Train CCA
    ca, X_c_train, Y_c_train, X_Y_sum_train, X_c_scaler, Y_c_scaler, corrs = train_CCA(
        pca_train_scaled, grad_train_scaled, ca_n_comp, save=False
    )

    # 7. Test CCA
    X_c_test, Y_c_test, X_Y_sum_test = test_CCA(
        pca_test_scaled, grad_test_scaled, ca, X_c_scaler, Y_c_scaler
    )

    # 8. Prepare, concat, and save variates
    train_variates = prepare_variates(
        pca_train_scaled, X_c_train, Y_c_train, X_Y_sum_train
    )
    test_variates = prepare_variates(pca_test_scaled, X_c_test, Y_c_test, X_Y_sum_test)
    combined_variates = pd.concat([train_variates, test_variates])
    save_variates(combined_variates, "scratch//results//cca//holdOut", test_task)
