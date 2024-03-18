# this script wrangles task performance data

import pandas as pd
import re

# read in data (manually cleaned a bit already (column names etc))
wide_df = pd.read_csv('scratch/data/source/task_perf_correctRT_edited.csv')

# average go/no-go columns
column1 = 'Go_acc'
column2 = 'NoGo_acc'

# Calculate the average and store it in a new column
wide_df['GoNoGo_acc'] = wide_df[[column1, column2]].mean(axis=1)
wide_df = wide_df.drop([column1, column2], axis=1)

# Melt the DataFrame and extract task, metric, and value columns
melted_df = pd.melt(wide_df, id_vars='Id_number', var_name='Task_Info', value_name='Value')

# Split the 'Task_Info' column into separate columns for 'Task' and 'Metric'
melted_df[['Task_name', 'Metric']] = melted_df['Task_Info'].str.split('_', 1, expand=True)

# Pivot the DataFrame to arrange 'Metric' values as columns
final_long_df = melted_df.pivot_table(index=['Id_number', 'Task_name'], columns='Metric', values='Value', aggfunc='first').reset_index()

# Function to transform subject strings
def transform_subject_string(input_str):
    numeric_part = re.search(r'\d+', input_str).group()
    formatted_numeric_part = '{:03d}'.format(int(numeric_part))
    non_numeric_part = re.sub(r'\d+', '', input_str)
    result_str = non_numeric_part.rstrip('_') + formatted_numeric_part
    return result_str

# Apply the transformation to the 'Subject' column
final_long_df['Id_number'] = final_long_df['Id_number'].apply(transform_subject_string)

# merge with demo and mdES
# pca_data = pd.read_csv('scratch/data/esq_demo_forPCA.csv')

# final_long_df['Id_number'] = final_long_df['Id_number'].astype(str)
# pca_data['Id_number'] = pca_data['Id_number'].astype(str)

# merge on sub id and task name
# merged_df = pd.merge(final_long_df, pca_data, on=['Id_number', 'Task_name'])

# average by task and person
# result_df = final_long_df.groupby(['Task_name', 'Id_number'], as_index=False).mean()

# save
final_long_df.to_csv('scratch/data/task_perf_correctRT.csv', index=False)

print ("end")
