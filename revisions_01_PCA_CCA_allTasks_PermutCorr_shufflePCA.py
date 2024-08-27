"""

This script runs omnibus PCA (no rotation) and CCA on original (n = 190) 
task battery data.

It asks for 5 PCA components (>50% explained variances)
and performs CCA on trial-level.

Main analysis: shuffles PCA observations consistently across PS.

Multiple shuffle methods available if required.

"""

import pandas as pd
import matplotlib.pyplot as plt
from ThoughtSpace.pca import basePCA
from ThoughtSpace.utils import returnhighest
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# import custom functions from utils
from utils.cca import prepare_X, prepare_Y, train_CCA, shuff, select_cols

#################################### Data #####################################

# read in lab data with 7220 rows
esq_data = pd.read_csv("scratch//data//esq_demo_forPCA.csv")
esq_data["Age"] = esq_data["Age"].astype(str)  # set to str so not inc. in PCA

esq_data = esq_data.drop(['Age','Gender','Language','Country'], axis=1)

# Step 1 & 2: Group by 'Id_number' and 'task_name', then calculate the mean
esq_data = esq_data.groupby(["Id_number", "Task_name"]).mean()

# Step 3: Reset the index to make 'Id_number' and 'task_name' regular columns again
esq_data.reset_index(inplace=True)

# read in grad data with 7220 rows, trial-level
grad_data = pd.read_csv(
    "scratch//data//gradscores_spearman_combinedmask_cortical_subcortical_forCCA.csv"
)
n_grads = 5  # set number of grads to 5

# Step 1 & 2: Group by 'Id_number' and 'task_name', then calculate the mean
grad_data = grad_data.groupby(["Id_number", "Task_name"]).mean()

# Step 3: Reset the index to make 'Id_number' and 'task_name' regular columns again
grad_data.reset_index(inplace=True)

############################ PCA calculation ###################################

# calculate 5 component solution
n_comp = 5

# fit and transform,no rotation, 5 comps
model = basePCA(n_components=n_comp, rotation=False, verbosity=1)
output = model.fit_transform(esq_data)

# save results out
# model.save(path="scratch//results//pca//allTasks",pathprefix=f"allTasks{n_comp}")

############################# CCA ###############################################

# Prepare PCA and grad data for CCA (i.e., scale variables)
pca_scaled = prepare_X(output)[0]
grad_scaled = prepare_Y(grad_data)[0]

# save out for sanity checking results in R
# pca_scaled.to_csv(f"scratch//data//pca{n_comp}_for_CCA_inR.csv")
# grad_scaled.to_csv("scratch//data//grads_for_CCA_inR.csv")

## call train_CCA on all 'true' data (i.e., not shuffled)
# stores model (ca), variates (X_c, Y_c, X_Y_sum), scaler of variates (X_c_scaler, Y_c_scaler), CCA correlations (corrs)
ca, X_c, Y_c, X_Y_sum, X_c_scaler, Y_c_scaler, corrs = train_CCA(
    pca_scaled, grad_scaled, n_comp, "scratch//results//cca//allTasks", save=False
)

# Standardize weights of CCA model for making CCA word clouds and brain maps
scaler = StandardScaler()
std_x_weights = scaler.fit_transform(ca.x_weights_)
std_y_weights = scaler.fit_transform(ca.y_weights_)

# select cols for using col names below when saving weights
pca_cols = select_cols(pca_scaled, "Z")
grad_cols = select_cols(grad_scaled, "Z")
if not os.path.exists("scratch\\results\\cca\\groupedTasks"):
    os.mkdir("scratch\\results\\cca\\groupedTasks")
# save X and Y weights
x_weights_df = pd.DataFrame(
    std_x_weights,
    index=pca_cols.columns,
    columns=[f"X_weight_{i+1}" for i in range(ca.x_weights_.shape[1])],
)
x_weights_df.to_csv(
    f"scratch\\results\\cca\\groupedTasks\\pca_weights_noRot_{n_comp}ncomp.csv"
)

y_weights_df = pd.DataFrame(
    std_y_weights,
    index=grad_cols.columns,
    columns=[f"Y_weight_{i+1}" for i in range(ca.y_weights_.shape[1])],
)
y_weights_df.to_csv(
    f"scratch\\results\\cca\\groupedTasks\\grad_weights_noRot_{n_comp}ncomp.csv"
)


print("end")
