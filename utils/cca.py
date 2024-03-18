import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ThoughtSpace.pca import basePCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import seaborn as sns
import random

def prepare_X(output, scaler=None):
    """
    Performs data preparation steps on PCA data for CCA. 
    Returns a scaled PCA dataframe.

    Parameters:
        output (pandas.DataFrame): The output dataframe from ThoughtSpace containing PCA scores.
        scaler_data (pandas.DataFrame): The data to use for fitting the scaler. 
            If None, the scaler is fit on the input 'output' and returned.

    Returns:
        pandas.DataFrame: The scaled PCA dataframe.
        StandardScaler: The fitted scaler if 'scaler_data' is None.
    """
    original_index = output.index
    pca_cols = [col for col in output.columns if col.startswith("PCA")]
    pca_scores = output[pca_cols]

    if scaler is None:
        # Fit scaler on the provided data (default behavior)
        pca_scaler = StandardScaler()
        pca_scaler.fit(pca_scores)
    else:
        # Use the scaler fitted on 'scaler_data'
        pca_scaler = scaler

    Zpca_scores = pca_scaler.transform(pca_scores)

    Zpca_scores_df = pd.DataFrame(Zpca_scores, columns=[f"Z{col}" for col in pca_cols])
    Zpca_scores_df.index = original_index
    pca_df = pd.concat([output, Zpca_scores_df], axis=1)

    # Return PCA df
    return pca_df, pca_scaler

def prepare_Y(y_data, scaler=None):
    """
    Scales the Y data (gradient scores) using a scaler. If a scaler is not provided, it will be fitted
    on the y_data. Returns the scaled Y data.

    Parameters:
        y_data (pandas.DataFrame): The data to be scaled.
        scaler (StandardScaler): A pre-fitted scaler to be used for scaling y_data. If None, a new
            scaler is fitted on y_data.

    Returns:
        pandas.DataFrame: The scaled Y data.
        StandardScaler: The fitted scaler if 'scaler' is None.
    """
    y_cols = [col for col in y_data.columns if col.startswith("gradient")]

    if scaler is None:
        # Fit a new scaler on the provided data (default behavior)
        grad_scaler = StandardScaler()
        grad_scaler.fit(y_data[y_cols])
    else:
        # Use the provided scaler
        grad_scaler = scaler

    Zy_data = grad_scaler.transform(y_data[y_cols])

    Zy_data_df = pd.DataFrame(Zy_data, columns=[f"Z{col}" for col in y_cols])
    Zy_data_df.index = y_data.index

    scaled_y_data = pd.concat([y_data, Zy_data_df], axis=1)

    # Return the scaled target data and the fitted scaler
    return scaled_y_data, grad_scaler

# Function to select cols starting with a particular string
def select_cols(inputdata, string):
    data_cols = [col for col in inputdata.columns if col.startswith(string)]
    outputdata = inputdata[data_cols]
    return outputdata

# function to scale X and Y variates (before summing)
def scale_variates(_c, scaler= None):
    # if no scaler provdided, create new scaler object
    if scaler is None:
        _c_scaler = StandardScaler()
        _c_scaler.fit(_c)
    # otherwise, use provided scaler
    else:
        _c_scaler = scaler

    Zc_train = _c_scaler.transform(_c)
    return Zc_train, _c_scaler

# function to sum X and Y variates (after scaling)
def sum_variates(ZX_c, ZY_c):
    X_Y_sum = ZX_c + ZY_c
    return X_Y_sum

# Function to fit CCA model on training data
def fit_CCA(X_train, Y_train, ca_n_comp):
    # fit CCA model on training data
    ca = CCA(scale=False, n_components=ca_n_comp)
    ca.fit(X_train, Y_train)
    # this saves X and Y variates
    X_c_train, Y_c_train = ca.transform(X_train, Y_train)
    # returns cca model, X variates, and Y variates
    return ca, X_c_train, Y_c_train

# Function to prepare X and Y variates for saving results dataframe of CCA
def prepare_variates(pca_data, X_c, Y_c, summed_c):
    # create dataframe with columns from PCA output
    cc_res = pd.DataFrame({"Task_name": pca_data.Task_name.tolist(),
                           "Id_number": pca_data.Id_number.tolist(),
                           "Age": pca_data.Age.tolist(),
                           "Gender": pca_data.Gender.tolist()})

    # add CCA results to dataframe
    for i, (x_col, y_col, summed_col) in enumerate(zip(X_c.T, Y_c.T, summed_c.T)):
        cc_res[f'pca_CCAloading_{i+1}'] = x_col
        cc_res[f'grad_CCAloading_{i+1}'] = y_col
        cc_res[f'sum_CCAloading_{i+1}'] = summed_col

    # return results dataframe with x, y, and summed variates
    return cc_res

# Function to save CCA results dataframe
def save_variates(cc_res, outputdir, taskname = None):
    if taskname is None:
        cc_res.to_csv(f"{outputdir}//allTask_variates.csv", index=False)
    else:
        cc_res.to_csv(f"{outputdir}//{taskname}_variates.csv", index=False)

# Function to calculate Canonical Correlations
def calculate_correlations(X_c, Y_c):
    corrs = []
    for col in range(X_c.shape[1]):
        corr = np.corrcoef(X_c[:, col], Y_c[:, col])
        corr = corr[0, 1]
        corrs.append(corr)
    # print to user
    print("The Canonical Correlations are:", corrs)
    return corrs

# Wrapper function for running training CCA
def train_CCA(pca_train, grad_train, ca_n_comp, outputdir = None, save=False):
    # select columns
    X_train = select_cols(pca_train, 'Z')
    Y_train = select_cols(grad_train, 'Z')

    # fit CCA
    ca, X_c_train, Y_c_train = fit_CCA(X_train, Y_train, ca_n_comp)

    # Scale X and Y variates
    ZX_c_train, X_c_scaler = scale_variates(X_c_train)
    ZY_c_train, Y_c_scaler = scale_variates(Y_c_train)

    # sum scaled variates
    X_Y_sum_train = sum_variates(ZX_c_train, ZY_c_train)
    
    # optionally save
    if save:
        variates = prepare_variates(pca_train, X_c_train, Y_c_train, X_Y_sum_train)
        save_variates(variates, outputdir)

    # calculate and print out correlations
    corrs = calculate_correlations(X_c_train, Y_c_train)

    # return cca model, X variates, Y variates, Summed variates, scalers for variates, and correlations
    return ca, X_c_train, Y_c_train, X_Y_sum_train, X_c_scaler, Y_c_scaler, corrs

# Wrapper function for applying CCA to test data
def test_CCA(pca_test, grad_test, ca, X_c_scaler, Y_c_scaler):
    # select cols
    X_test = select_cols(pca_test, 'Z')
    Y_test = select_cols(grad_test, 'Z')

    # apply already trained CCA model to test data
    X_c_test, Y_c_test = ca.transform(X_test, Y_test)

    # scale X and Y variates and then sum scaled variates
    ZX_c_test = scale_variates(X_c_test, X_c_scaler)[0]
    ZY_c_test = scale_variates(Y_c_test, Y_c_scaler)[0]
    X_Y_sum_test = sum_variates(ZX_c_test, ZY_c_test)

    # return X, Y, and summed variates from test data
    return X_c_test, Y_c_test, X_Y_sum_test

# create shuffled data for permutation testing
def shuff(num_iterations, data, shuff_type):
    shuffled_dict = {}
    for i in range(num_iterations):
        print (i) # print iteration
        # most basic shuffle = shuffle all rows of provided data
        if shuff_type == "basic":
            seed_value = i # set seed based on iteration number for reproducibility
            shuffled_df = data.sample(frac=1, random_state=seed_value)
            shuffled_dict[i] = shuffled_df

        # this shuffles rows within each participant, diff shuffle per PS
        elif shuff_type == 'byPS_diff':
            data['Numeric_Id'] = data['Id_number'].apply(lambda x: int(''.join(filter(str.isdigit, str(x))))) # create numeric ID col for random state below
            # set random state to iteration number + Numberic ID for that group; reproducible but different for each participant and iteration
            shuffled_df = data.groupby('Id_number').apply(lambda x: x.sample(frac=1, random_state=(i + int(x['Numeric_Id'].iloc[0])))).reset_index(drop=True)
            #shuffled_df['Task_name'] = data['Task_name'] # really don't think this is needed anymore (keeping in case bug comes up later)
            shuffled_dict[i] = shuffled_df

        # this shuffles rows within each participant, same shuffle per PS
        elif shuff_type == 'byPS_same':
            # Group by 'Id_number'
            grouped = data.groupby('Id_number')

            # List to store shuffled groups
            shuffled_groups = []

            for _, group_data in grouped:
                # Set seed for each group to be iteration number (same shuffle per PS on each iteration and reproducible)
                seed = i
                random.seed(seed)
                # Convert the group to a list to shuffle it
                group_list = group_data.values.tolist()
                random.shuffle(group_list)
                
                # Append the shuffled group to the list
                shuffled_groups.append(pd.DataFrame(group_list, columns=group_data.columns))

            # Concatenate all shuffled groups into a single DataFrame
            shuffled_df = pd.concat(shuffled_groups, ignore_index=True)
            shuffled_dict[i] = shuffled_df
        
    return shuffled_dict

# function for predicting Y (gradients) based on X (pca)
def cca_predict(X_test, ca_model):
    predictions = ca_model.predict(X_test)
    return predictions

# function for calculating distance between predicted Y and real Y
def distances(Y_test, predictions):
    # takes real Y data (gradient coords) and substracts predictions
    perDim_dist =  Y_test - predictions
    abs_perDim_dist = abs(perDim_dist) # makes distance absolute
    # calculate across gradients
    euclidean_dist = np.sqrt(np.sum(perDim_dist**2, axis=1))
    # return per-dim and across-dim distances
    return abs_perDim_dist, euclidean_dist

# predict Y values for every task provided in task list
def tasklist_prediction(task_list, pca_data, grad_data, ca_model):
    # dictionaries for adding predictions and distances to
    task_predictions = {}
    task_distances_perDim = {}
    task_distances_allDim = {}

    for task in task_list:
        # select x and y rows matching that task name
        X_mask = pca_data['Task_name'] == task
        X_rows = pca_data[X_mask]

        Y_mask = grad_data['Task_name'] == task
        Y_rows = grad_data[Y_mask]

        # select scaled PCA and grad columns
        X_test = select_cols(X_rows, 'Z')
        Y_test = select_cols(Y_rows, 'Z')
        
        # make a prediction of Y by providing X and already trained CCA model
        prediction = cca_predict(X_test, ca_model)

        # convert predictions to dataframe
        prediction = pd.DataFrame(prediction)
        
        # add ID number column and average by ID for comparing distances
        # results in one prediction per PS in each task
        X_rows.reset_index(inplace = True)
        prediction["Id_number"] = X_rows["Id_number"]
        avg_prediction = prediction.groupby("Id_number").mean().reset_index().copy()

        # drop ID column
        avg_prediction.drop("Id_number", axis=1, inplace=True)

        # convert back to array
        avg_prediction =  avg_prediction.to_numpy()

        task_predictions[task] = avg_prediction # save avg predictions in dictionary

        # calculate distances at per-dim level and across dims
        # Y_test = that task's gradient coords, scaled by all task coords in og data
        perdim_dist, alldim_dist = distances(Y_test, avg_prediction)
        task_distances_perDim[task] = perdim_dist # store in dict
        task_distances_allDim[task] = alldim_dist # store in dict
    
    return task_predictions, task_distances_perDim, task_distances_allDim

# compare distance of predicted values to other predicted values
def otherTask_pred_dist(task_predictions, grad_data):
    other_distances_perDim = {}
    other_distances_allDim = {}
    # loop over tasks in task prediction dictionary
    for task, task_prediction in task_predictions.items():
        # real grad score = current task

        # select real gradient coordinates for that task
        Y_mask = grad_data['Task_name'] == task
        Y_rows = grad_data[Y_mask]

        # average by person so same number of rows per task
        # Y_rows = Y_rows.groupby("Id_number").mean().reset_index()
        Y_test = select_cols(Y_rows, 'Z')

        # dictionaries for storing distances
        other_distances_perDim[task] = {}
        other_distances_allDim[task] = {}

        # loop over again to loop over other tasks, skipping current task
        for other_task, other_task_prediction in task_predictions.items():
            if task != other_task:
                # calculate distance between real grad coords for current task and other tasks' prediction
                other_task_distance_perDim, other_task_distance_allDim = distances(Y_test, other_task_prediction)
                # add distances to dictionaries and return
                other_distances_perDim[task][other_task] = other_task_distance_perDim
                other_distances_allDim[task][other_task] = other_task_distance_allDim
    
    return other_distances_perDim, other_distances_allDim

# function for formatting t-test results as they print out
def format_results(results):
    formatted_results = {
        task: {
            other_task: {
                't_statistic': [
                    round(val, 2) for val in t_test_result['t_statistic']
                ],
                'p_value': [round(val, 3) for val in t_test_result['p_value']],
            }
            for other_task, t_test_result in task_results.items()
        }
        for task, task_results in results.items()
    }
    return formatted_results

# function to plot and optionally save heatmap across all dimensions
def heatmap_allDim(num_tasks, custom_order, true_task_distances_allDim, other_distances_allDim, x_lab, y_lab,
                   filename, outputdir):
    heatmap_data = np.zeros((num_tasks, num_tasks))

    # Fill the matrix with true distances on the diagonal and other distances elsewhere
    for i, task in enumerate(true_task_distances_allDim.keys()):
        heatmap_data[i, i] = np.mean(true_task_distances_allDim[task])  # Use the mean of true distances
        for j, other_task in enumerate(true_task_distances_allDim.keys()):
            if task != other_task:
                heatmap_data[i, j] = np.mean(other_distances_allDim[task][other_task])  # Use the mean of other distances

    # Reorder rows and columns of the heatmap_data matrix
    custom_order_indices = [list(true_task_distances_allDim.keys()).index(task) for task in custom_order]
    heatmap_data = heatmap_data[custom_order_indices][:, custom_order_indices]

    # Create a seaborn heatmap
    fig_width_cm = 3.4
    fig_height_cm = 2.8
    fig_width_inches = fig_width_cm / 2.54
    fig_height_inches = fig_height_cm / 2.54

    plt.figure(figsize=(fig_width_inches , fig_height_inches),dpi=1000)  # Adjust the figsize as needed

    # Set the font size in points
    plt.rcParams.update({'font.size': 6})

    # Adjust the layout to prevent clipping of labels
    plt.tight_layout()

    heatmap = sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".2f", linewidths=.5, square=True,cbar_kws={'shrink': 0.6,
                                                                                                                    "pad": 0.02},
                 xticklabels=custom_order, yticklabels=custom_order)

    #plt.xlabel(f'{x_lab}')
    #plt.ylabel(f'{y_lab}')

    # Rotate x-axis labels
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right')

    if outputdir:
        plt.savefig(f"{outputdir}//heatmap_allDim_{filename}.png", bbox_inches='tight',pad_inches=0.01)

    plt.show()

# function to plot and optionally save heatmap for each dimension
def heatmap_perDim(num_tasks, custom_order, num_dims, true_task_distances_perDim, other_distances_perDim, x_lab, y_lab,
                   title,
                   filename, outputdir):
    for k in range(num_dims):
        heatmap_data = np.zeros((num_tasks, num_tasks))

        # Fill the matrix with true distances for the last row and last column
        for i, task in enumerate(true_task_distances_perDim.keys()):
            true_distance_matrix = np.array(true_task_distances_perDim[task])
            heatmap_data[i, i] = np.mean(true_distance_matrix[:, k])

            for j, other_task in enumerate(true_task_distances_perDim.keys()):
                if task != other_task:
                    other_distance_matrix = np.array(other_distances_perDim[task][other_task])
                    heatmap_data[i, j] = np.mean(other_distance_matrix[:, k])

        # Reorder rows and columns of the heatmap_data matrix
        custom_order_indices = [list(true_task_distances_perDim.keys()).index(task) for task in custom_order]
        heatmap_data = heatmap_data[custom_order_indices][:, custom_order_indices]

        # Create a seaborn heatmap
        plt.figure(figsize=(4, 4))  # Adjust the figsize as needed
        sns.set(font_scale=0.7)  # Adjust the font scale for axis labels

        sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".2f", linewidths=.5, square=True,cbar_kws={'shrink': 0.6},
                 xticklabels=custom_order, yticklabels=custom_order)
        
        plt.title(f'{title}: Dimension {k+1}')  # Seaborn automatically adjusts the font size of the title
        plt.xlabel(f'{x_lab}')
        plt.ylabel(f'{y_lab}')
        
        if outputdir:
            plt.savefig(f"{outputdir}//heatmap_dim{k+1}_{filename}.png", bbox_inches='tight')

        #plt.show()

## functions for running t-tests of distances
def ttests(true_task_distances_perDim, true_task_distances_allDim,other_distances_perDim, other_distances_allDim):
    """
    Calculates t-tests to show that the distances are smaller for a given task compared to 'other task'
    distances which reflect:
    1. distance between between current task's real gradient value and the other tasks' predictions.
    NOT 2. distance between between other task's real gradient value and current task's predictions.
    
    Returns dictionaries of t-test results at per-dim level and across-dim level.

    Parameters:
        true_task_distances_perDim (dictionary): Dictionary containing per-dim distances for current task.
        true_task_distances_allDim (dictionary): Dictionary containing across-dim distances for current task.
        other_distances_perDim (dictionary): Dictionary containing per-dim distances for other tasks.
        other_distances_allDim (dictionary): Dictionary containing across-dim distances for other tasks.

    Returns:
        t_test_results: Dictionary of t-test results for each task at per-dim level.
        t_test_results_eluc: Dictionary of t-test results for each task at across-dim level.
    """
    t_test_results = {} # for storing results
    t_test_results_eluc = {} # for storing results

    # Iterate through each task in true_distance_dict
    for task in true_task_distances_perDim.keys():
        t_test_results[task] = {}
        t_test_results_eluc[task] = {}

        # Get the distribution of the true distances for the current task
        true_distances = true_task_distances_perDim[task]
        true_distances_eluc = true_task_distances_allDim[task]

        # Iterate through the other tasks in other_distance_dict for the current task
        # selects using 'task' as that selects all distances that were calculated in relation to that task
        for other_task, other_distances in other_distances_perDim[task].items():
            
            # Perform a paired sample t-test, 1 tailed because hypothesis = distances for current task are smaller
            t_statistic, p_value = stats.ttest_rel(true_distances, other_distances, alternative='less')
            t_statistic_eluc, p_value_eluc = stats.ttest_rel(true_distances_eluc, other_distances_allDim[task][other_task], alternative='less')
            
            # Store the t-statistic and p-value in the results dictionary
            t_test_results[task][other_task] = {
                't_statistic': t_statistic,
                'p_value': p_value
            }

            t_test_results_eluc[task][other_task] = {
                't_statistic': t_statistic_eluc,
                'p_value': p_value_eluc
            }
    return t_test_results, t_test_results_eluc

# function for printing out formatted t-test results
def print_format(other_task, t_statistic_formatted, p_value_formatted):
    print(f"    Comparison with {other_task}:")
    print(f"        t-statistic: [{t_statistic_formatted}]")
    print(f"        one-tailed p-value: [{p_value_formatted}]")

# Wrapper function to print the one-tailed t-test results with all values formatted
def print_ttests(t_test_results, t_test_results_eluc):
    for task, results in t_test_results.items():
        print(f"Task: {task}")
        for other_task, t_test_result in results.items():
            t_statistic_formatted = ", ".join("{:.2f}".format(val) for val in t_test_result['t_statistic'])
            p_value_formatted = ", ".join("{:.3f}".format(val) for val in t_test_result['p_value'])
            print_format(
                other_task, t_statistic_formatted, p_value_formatted
            )
    for task, results in t_test_results_eluc.items():
        print(f"Task: {task}")
        for other_task, t_test_result in results.items():
            t_statistic_formatted = "{:.4f}".format(t_test_result['t_statistic'])
            p_value_formatted = "{:.4f}".format(t_test_result['p_value'])
            print_format(
                other_task, t_statistic_formatted, p_value_formatted
            )

## PCA function...

# function for running held-out PCA
def heldout_pca(esq_all, esq_train, test_task, train, test, ncomp, rotation, save = False):

    # apply PCA to training ESQ data and project solution on all data
    pca_model = basePCA(n_components=ncomp, rotation=rotation)
    pca_model.fit(esq_train) # train PCA model on ESQ training data
    output = pca_model.transform(esq_all) # fit model to all esq data

    if save:
        pca_model.save(path="scratch//results//pca//holdOut",pathprefix=test_task)
    
    # select train and test to return for CCA
    output_train = output.iloc[train]
    output_test = output.iloc[test]

    return output_train, output_test
