import pandas as pd
from ThoughtSpace.rhom import splithalf, dir_proj

## Load the dataframes containing the mDES probe data for the original and replication samples.
mdes_original = pd.read_csv('scratch/data/esq_demo_forPCA.csv')
mdes_replication = pd.read_csv('scratch/data/esq_demo_replication.csv')

## Include the columns relevant to the mDES probes.
pca_original = mdes_original.filter(regex="_response")

## Conduct a split-half reliability analysis on the 5-component solution for the original set of mDES probes.
orig_splithalf = splithalf(pca_original,
                           npc = 5,
                           rotation = "none",
                           file_prefix = "revisions")

## Prepare the original and replication datasets for component-similarity analysis (i.e., add a column designating a grouping variable).
pca_original['dataset'] = 'Original'

pca_replication = mdes_replication.filter(regex="_response")
pca_replication['dataset'] = "Replication"

mdes_merge = pd.concat([pca_original, pca_replication], axis = 0)

## Conduct a direct-projection reproducibility analysis on the merged datasets using 'dataset' (i.e., original vs. replication) as a grouping variable.
## This assesses the component similarity of the 5-component solution for the original set compared to the same solution for the replication set.
mergedset_dirproj = dir_proj(mdes_merge,
                             "dataset",
                             npc = 5,
                             rotation = "none",
                             file_prefix = "revisions")
