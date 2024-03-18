# -*- coding: utf-8 -*-
"""
Created on Fri 15th Sep 2023.

@author: bronte mckeown

Wrangle gradient scores and ESQ output (original sample) for PCA and CCA analyses.

"""

# %% import libraries

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# %% Read in data
os.chdir('c:/Users/bront/Documents/CanadaPostdoc/MegaProject/TaskBatteryAnalysis')

# read in lab data
lab_data = pd.read_csv("scratch//data//source//esq_demo_synolology.csv")

# drop run time and note column
esq_df = lab_data.drop(["Runtime_mod", "Note"
                  ], axis=1)

# change col name of gender for merging
esq_df.rename(columns={'Gender Identity': 'Gender'}, inplace=True)

# make task names shorter for plotting
replace_dict_tasks = {'You_Task':'You',
                'Friend_Task':'Friend',
                'Two-Back_Task-faces':'2B-Face',
                'Two-Back_Task-scenes':'2B-Scene',
                'Easy_Math_Task':'EasyMath',
                'Hard_Math_Task':'HardMath',
                'Reading_Task':'Read',
                'Memory_Task':'Memory',
                'One-Back_Task':'1B',
                'Zero-Back_Task':'0B',
                'GoNoGo_Task':'GoNoGo',
                'Finger_Tapping_Task':'FingerTap',
                'Movie_Task-incept':'SciFi',
                'Movie_Task-bridge':'Documentary',
                }
esq_df = esq_df.replace({'Task_name': replace_dict_tasks})

###################################################################################

# check demographic cols
esq_df.columns = [col.rstrip(' ') for col in esq_df.columns] # remove any empty strings
print(esq_df['Gender'].unique())
print(esq_df['Country'].unique())
print(esq_df['Language'].unique())
print(esq_df['Age'].unique())

# change men to man in gender col
replace_dict_gender = {'Men': 'Man'}
esq_df = esq_df.replace({'Gender': replace_dict_gender})

# change canad to canada in country col
replace_dict_country = {'Canad': 'Canada',
                        'Cnada': 'Canada',
                        'Canada ': 'Canada'}
esq_df = esq_df.replace({'Country': replace_dict_country})

# replace error in language
replace_dict_lan = {'Engligh': 'English'}
esq_df = esq_df.replace({'Language': replace_dict_lan})

print(esq_df['Gender'].unique())
print(esq_df['Country'].unique())
print(esq_df['Language'].unique())
print(esq_df['Age'].unique())

# check number of participants and number of probes per participant
print('Number of PS:', len(esq_df['Id_number'].unique()))
observation_counts = esq_df.groupby('Id_number').size().reset_index(name='Observation_count')

# Check if every participant has exactly 38 observations
all_have_38_observations = all(observation_counts['Observation_count'] == 38)
print('PS all have 38 observations:', all_have_38_observations)

# Aggregate data for counts 
aggregated_data = esq_df.groupby('Id_number').first().reset_index()

# Average age and age range of wide format data
modes = aggregated_data[['Age', 'Gender', 'Country','Language']].mode().iloc[0]
average_age = aggregated_data['Age'].mean()
min_age = aggregated_data['Age'].min()
max_age = aggregated_data['Age'].max()
age_counts = aggregated_data['Age'].value_counts(dropna=False)

# Print the results
print("Average Age:", average_age)
print ("Mode Age:", modes['Age'])
print("Age Range:", min_age, "to", max_age)
print (age_counts)

# Gender Counts
print ("Mode Gender:", modes['Gender'])
# Count occurrences of each unique 'Gender' value
gender_counts = aggregated_data['Gender'].value_counts(dropna=False)
print(gender_counts)

# Country Counts
print ("Mode Country:", modes['Country'])
# Count occurrences of each unique 'Gender' value
country_counts = aggregated_data['Country'].value_counts(dropna = False)
print(country_counts)

# Language Counts
print ("Mode Language:", modes['Language'])
# Count occurrences of each unique 'Gender' value
lan_counts = aggregated_data['Language'].value_counts(dropna=False)
print(lan_counts)

# drop rows with nan values in either age or gender
# this removes 4 participants, leaving 190
specified_columns = ['Age', 'Gender']
esq_df.dropna(subset=specified_columns, inplace=True)

# sort by task and ID so that the order of Task_name is the same for each ID
esq_df = esq_df.sort_values(by=['Id_number', 'Task_name'])

# Make task name the index for making long version of grad_data below
esq_df.index = esq_df["Task_name"]

# print number of participants after removing nans
print('Number of PS after removing those with no demographics:', len(esq_df['Id_number'].unique()))

# save to data folder
esq_df.to_csv("scratch//data//esq_demo_forPCA.csv", index = False)

###################################################################################

## read in grad data and make changes to index
grad_data = pd.read_csv("scratch//data//source//gradscores_spearman_combinedmask_cortical_subcortical.csv")

# make task names match esq dataframe
replace_dict_tasks = {'you': 'You',
                'friend': 'Friend',
                'twoBackFaces':'2B-Face',
                'twoBackScenes':'2B-Scene',
                'easyMath':'EasyMath',
                'hardMath':'HardMath',
                'reading':'Read',
                'memory':'Memory',
                'oneBack':'1B',
                'zeroBack':'0B',
                'gonogo':'GoNoGo',
                'fingerTapping':'FingerTap',
                'movieIncept':'SciFi',
                'movieBridge':'Documentary'
                }
grad_data = grad_data.replace({'Task_name': replace_dict_tasks})

grad_data.index = grad_data["Task_name"] # make task name the index

long_grad_df = grad_data.reindex(esq_df.index) # make long version to match esq
long_grad_df['Id_number'] = esq_df['Id_number'] # add id column to long version

# reset index
long_grad_df.reset_index(drop=True, inplace=True)

# Remove the common string from column names to tidy up
new_column_names = [col.replace('_cortical_subcortical', '') for col in long_grad_df.columns]

# Update the DataFrame with the new column names
long_grad_df.columns = new_column_names

# save per-trial version for CCA analyses
long_grad_df.to_csv("scratch//data//gradscores_spearman_combinedmask_cortical_subcortical_forCCA.csv", index = False)

print ("end")