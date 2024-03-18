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

# import custom functions from utils
from utils.cca import prepare_X, prepare_Y, train_CCA, shuff, select_cols

#################################### Data #####################################

# read in lab data with 7220 rows
esq_data = pd.read_csv("scratch//data//esq_demo_forPCA.csv")
esq_data['Age'] = esq_data['Age'].astype(str) # set to str so not inc. in PCA

# read in grad data with 7220 rows, trial-level
grad_data = pd.read_csv("scratch//data//gradscores_spearman_combinedmask_cortical_subcortical_forCCA.csv")
n_grads = 5 # set number of grads to 5

############################ PCA calculation ###################################

# calculate 5 component solution
n_comp = 5

# fit and transform,no rotation, 5 comps
model = basePCA(n_components=n_comp, rotation = False, verbosity = 1)
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
ca, X_c, Y_c, X_Y_sum, X_c_scaler, Y_c_scaler, corrs = train_CCA(pca_scaled, grad_scaled, n_comp,
                                                                  "scratch//results//cca//allTasks", save=False)

# Standardize weights of CCA model for making CCA word clouds and brain maps
scaler = StandardScaler()
std_x_weights = scaler.fit_transform(ca.x_weights_)
std_y_weights = scaler.fit_transform(ca.y_weights_)

# select cols for using col names below when saving weights
pca_cols = select_cols(pca_scaled, "Z")
grad_cols = select_cols(grad_scaled, "Z")

# save X and Y weights
x_weights_df = pd.DataFrame(std_x_weights, index=pca_cols.columns, columns=[f"X_weight_{i+1}" for i in range(ca.x_weights_.shape[1])])
x_weights_df.to_csv(f"scratch\\results\\cca\\allTasks\\pca_weights_noRot_{n_comp}ncomp.csv")

y_weights_df = pd.DataFrame(std_y_weights, index=grad_cols.columns, columns=[f"Y_weight_{i+1}" for i in range(ca.y_weights_.shape[1])])
y_weights_df.to_csv(f"scratch\\results\\cca\\allTasks\\grad_weights_noRot_{n_comp}ncomp.csv")

############################# Null CCAs ############################################

## call train_CCA on all null (shuffled) data
# first shuffle the PCA data
num_iterations = 1000
type_shuff = "byPS_same" # "basic", "byPS_diff" or "byPS_same"
shuffled_dict = shuff(num_iterations,pca_scaled,type_shuff)

# create empty dictionaries for storing null results
null_corrs_dict = {} 
null_ca_dict = {}

for iter_num, shuff_pca in shuffled_dict.items():
    null_ca, null_X_c, null_Y_c, null_X_Y_sum, null_X_c_scaler, null_Y_c_scaler, null_corrs = train_CCA(shuff_pca, grad_scaled, n_comp, save=False)
    null_corrs_dict[iter_num] = null_corrs
    null_ca_dict[iter_num] = null_ca

###################################### p-value for correlations #####################
# first create dataframes of all the real and null correlations
real_corrs_df = pd.DataFrame(corrs).T
null_corrs_df = pd.DataFrame.from_dict(null_corrs_dict).T

# rename columns
null_corrs_df.columns = [f"Correlation {i+1}" for i in range(null_corrs_df.shape[1])]
real_corrs_df.columns = [f"Correlation {i+1}" for i in range(real_corrs_df.shape[1])]

# calculate p-value for each CCA correlation by counting the number of null correlations that are 
# greater than the real correlation and dividing by the number of permutations

p_values = [] # create list for storing p-values

# loop over columns in null dataframe
for col in range(null_corrs_df.shape[1]):
    # count the number of null correlations that are greater or equal to the real correlation (+1)
    # divide by the number of permutations (+1)
    p_value = (((null_corrs_df.iloc[:, col] >= real_corrs_df.iloc[0, col]).sum())+1) / ((null_corrs_df.shape[0])+1)
    p_values.append(p_value)

print (p_values)

######################### Plotting correlation permutations ###########################

# want each column to be a subplot of a figure with 4 rows and 1 column
# for each column in real_corrs_df and null_corrs_df, plot the real correlation as a line and the null correlations as a density plot

# create figure
fig, axs = plt.subplots(nrows=n_comp, ncols=1, figsize=(10, 10))

# loop over columns
for col in range(null_corrs_df.shape[1]):
    # plot real correlation as a dotted green line
    axs[col].axvline(x=real_corrs_df.iloc[0, col], color="green", linestyle="dotted", label="Real Correlation")
    # plot null correlations as a density plot (kde)
    sns.kdeplot(data=null_corrs_df.iloc[:, col], ax=axs[col], color="grey", fill = True, alpha = 0.5, linewidth=0, label = "Null Correlations")
    # set title
    axs[col].set_title(f"")
    # set x and y labels
    axs[col].set_xlabel(f"Canonical correlation {col+1}")
    axs[col].set_ylabel("Density")
    # set x limits
    axs[col].set_xlim(-1, 1)
    # set legend
    if col == 0:
        axs[col].legend()

# adjust spacing between subplots
fig.tight_layout()
# save figure
fig.savefig(f"scratch//results//cca//allTasks//cca_corrs_noRot_pca{n_comp}_{type_shuff}_pcashuff_{num_iterations}iters.png")

# also save out for plotting in R
null_corrs_df.to_csv(f"scratch\\results\\cca\\allTasks\\nullcorrs_{type_shuff}_pca.csv", index = False)
real_corrs_df.to_csv(f"scratch\\results\\cca\\allTasks\\realcorrs.csv", index = False)

print ("end")