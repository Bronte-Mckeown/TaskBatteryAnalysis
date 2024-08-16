# -*- coding: utf-8 -*-
"""

@author: bronte mckeown

Wrangle gradient scores and replication ESQ output for analyses.

"""

# %% import libraries

import pandas as pd
import os

# %% Functions


def participant_section(data):
    print("Number of PS with demographics:", len(data["Id_number"].unique()))

    # checking values for participant section
    print(data["Gender"].unique())
    print(data["Age"].unique())

    # Average age and age range of wide format data
    modes = data[["Age", "Gender"]].mode().iloc[0]
    average_age = data["Age"].mean()
    min_age = data["Age"].min()
    max_age = data["Age"].max()
    age_counts = data["Age"].value_counts(dropna=False)

    # Print the results
    print("Average Age:", average_age)
    print("Mode Age:", modes["Age"])
    print("Age Range:", min_age, "to", max_age)
    print(age_counts)

    # Standard Deviation of Age
    age_std = data["Age"].std()
    print("Standard Deviation of Age:", age_std)

    # Gender Counts
    print("Mode Gender:", modes["Gender"])
    # Count occurrences of each unique 'Gender' value
    gender_counts = data["Gender"].value_counts(dropna=False)
    print(gender_counts)


# %% Read in data
# os.chdir('c:/Users/bront/Documents/CanadaPostdoc/MegaProject/TaskBatteryAnalysis')

# read in mdes lab data
lab_data = pd.read_csv("scratch/data/source/esq_replication.csv")

# read in demographics lab data
demo_data = pd.read_csv("scratch/data/source/demo_replication.csv")

##################################################################################
# wrangle mDES and check n of IDs and observations

# drop run time column
esq_df = lab_data.drop(["Runtime_mod"], axis=1)

# make task name values match original dataset plus -Rep for plots
replace_dict_tasks = {
    "Hard_Math_Task": "HardMath-Rep",
    "Memory_Task": "Memory-Rep",
    "GoNoGo_Task": "GoNoGo-Rep",
    "Movie_Task-bridge": "Documentary-Rep",
}
esq_df = esq_df.replace({"Task_name": replace_dict_tasks})

# check number of participants and number of probes per participant
print("Number of PS in mDES data:", len(esq_df["Id_number"].unique()))

# check observation counts
observation_counts = (
    esq_df.groupby("Id_number").size().reset_index(name="Observation_count")
)

# Check if every participant has exactly 11 observations
all_have_11_observations = all(observation_counts["Observation_count"] == 11)
print("All have 11 observations:", all_have_11_observations)

# false so...
# Find ID numbers with less than 11 observations
id_numbers_less_than_11_observations = observation_counts[
    observation_counts["Observation_count"] < 11
]

# Print ID numbers with their number of observations
print("ID numbers with less than 11 observations:")
for index, row in id_numbers_less_than_11_observations.iterrows():
    print(f"ID: {row['Id_number']}, Observations: {row['Observation_count']}")

# 086 is missing half their data; removed as they can't be included
# in comparative analyses between all 4 tasks
esq_df_clean = esq_df[esq_df["Id_number"] != "subject_86"].copy()

##################################################################################
# wrangle demographics and check how many are missing

# change col name of gender and age to match original dataset
demo_data.rename(columns={"gender": "Gender"}, inplace=True)
demo_data.rename(columns={"age": "Age"}, inplace=True)

# participant section before removing 086
# participant_section(demo_data)

# remove subject 086 from demographics for participant section
demo_data_clean = demo_data[demo_data["Id_number"] != 86].copy()

# participant section after removing 086
participant_section(demo_data_clean)

###################################################################################

# merge demographics
# Convert 'ID_number' column to strings and add 'subject'
demo_data_clean["Id_number"] = "subject_" + demo_data_clean["Id_number"].astype(str)
# make esq a string too
esq_df_clean["Id_number"] = esq_df_clean["Id_number"].astype(str)

# merge on ID
esq_demo = pd.merge(esq_df_clean, demo_data_clean, on="Id_number", how="left")

###################################################################################
# drop rows with nan values in either age or gender
specified_columns = ["Age", "Gender"]
esq_demo.dropna(subset=specified_columns, inplace=True)

# sort by task and ID so that the order of Task_name is the same for each ID
esq_demo = esq_demo.sort_values(by=["Id_number", "Task_name"])

# save
esq_demo.to_csv("scratch//data//esq_demo_replication.csv", index=False)

# Make task name the index for making long version of grad_data below
esq_demo.index = esq_demo["Task_name"]

###################################################################################
## read in grad data and make changes to index
grad_data = pd.read_csv(
    "scratch//data//source//gradscores_spearman_combinedmask_cortical_subcortical.csv"
)

# select 4 tasks
# List of desired values
desired_values = ["hardMath", "memory", "gonogo", "movieBridge"]

# Select rows where 'column_name' matches any value in the list of desired values
grad_data = grad_data[grad_data["Task_name"].isin(desired_values)]

# make it match ESQ
replace_dict_tasks = {
    "hardMath": "HardMath-Rep",
    "memory": "Memory-Rep",
    "gonogo": "GoNoGo-Rep",
    "movieBridge": "Documentary-Rep",
}
grad_data = grad_data.replace({"Task_name": replace_dict_tasks})

grad_data.index = grad_data["Task_name"]  # make task name the index

long_grad_df = grad_data.reindex(esq_demo.index)  # make long version to match esq
long_grad_df["Id_number"] = esq_demo["Id_number"]  # add id column to long version

# reset index
long_grad_df.reset_index(drop=True, inplace=True)

# Remove the common string from column names
new_column_names = [
    col.replace("_cortical_subcortical", "") for col in long_grad_df.columns
]

# Update the DataFrame with the new column names
long_grad_df.columns = new_column_names

# can create averaged version of long_grad_df now to read into prediction CCA analysis
# average long_grad_df by Id_number and Task_name, set Sort to False and keep Id_number and Task_name as columns
grad_avg = long_grad_df.groupby(
    ["Id_number", "Task_name"], sort=False, as_index=False
).mean()

# change column names back for prediction script
replace_dict_tasks = {
    "HardMath-Rep": "HardMath",
    "Memory-Rep": "Memory",
    "GoNoGo-Rep": "GoNoGo",
    "Documentary-Rep": "Documentary",
}
grad_avg = grad_avg.replace({"Task_name": replace_dict_tasks})

# save average version
grad_avg.to_csv(
    "scratch//data//gradscores_spearman_combinedmask_cortical_subcortical_forCCA_prediction.csv",
    index=False,
)
