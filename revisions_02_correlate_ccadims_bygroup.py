import scipy.stats as stats
import pandas as pd
import csv

results = {}
nonagg = pd.read_csv(
    "scratch/results/cca/allTasks/grad_weights_noRot_5ncomp.csv",
    index_col=0,
    header=0,
).stack()

agg = pd.read_csv(
    "scratch/results/cca/old/grad_weights_noRot_5ncomp.csv",
    index_col=0,
    header=0,
).stack()


out = stats.pearsonr(nonagg.values, agg.values)

print(out)
results["grad_agg_corr"] = {"corr": out[0], "p-val": out[1]}


for x in agg.index.levels[0]:
    data_agg = agg.loc[x]
    data_nonagg = nonagg.loc[x]
    out = stats.pearsonr(data_nonagg.values, data_agg.values)
    results[f"grad_agg_corr_{x}"] = {"corr": out[0], "p-val": out[1]}
nonagg = pd.read_csv(
    "scratch/results/cca/allTasks/pca_weights_noRot_5ncomp.csv",
    index_col=0,
    header=0,
).stack()

agg = pd.read_csv(
    "scratch/results/cca/groupedTasks/pca_weights_noRot_5ncomp.csv",
    index_col=0,
    header=0,
).stack()


out = stats.pearsonr(nonagg.values, agg.values)
results["pca_agg_corr"] = {"corr": out[0], "p-val": out[1]}


print(out)
for x in agg.index.levels[0]:
    data_agg = agg.loc[x]
    data_nonagg = nonagg.loc[x]
    out = stats.pearsonr(data_nonagg.values, data_agg.values)
    results[f"pca_agg_corr_{x}"] = {"corr": out[0], "p-val": out[1]}

# Flatten the nested dictionary
flattened_data = []
for key, value in results.items():
    flattened_entry = {"key": key}
    flattened_entry.update(value)
    flattened_data.append(flattened_entry)

# Define the CSV file name
csv_file = "scratch/results/cca/allTasks/pca_grad_weights_correlation.csv"

# Get the headers from the first dictionary entry
headers = flattened_data[0].keys()

# Write to CSV
with open(csv_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()
    writer.writerows(flattened_data)
