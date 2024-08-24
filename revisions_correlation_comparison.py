import pandas as pd
from scipy.stats import spearmanr, pearsonr

cca_new = pd.read_csv("scratch//results//cca//allTasks//allTask_variates_averaged.csv",index_col="Task_name")
cca_old = pd.read_csv("scratch//results//cca//allTasks//allTask_variates_average_old.csv",index_col="Task_name")

ccs = list(cca_new.axes[1])
cca_old = cca_old.reindex(cca_new.index)


# cca_new[ccs[1]] *= -1
# cca_new[ccs[2]] *= -1



spearman = spearmanr(cca_new.values.flatten(), cca_old.values.flatten())

pearson = pearsonr(cca_new.values.flatten(), cca_old.values.flatten())


print(f'Pearson: stat={pearson.correlation}, pvalue={pearson.pvalue}')
print(f'Spearman: stat={spearman.correlation}, pvalue={spearman.pvalue}')
