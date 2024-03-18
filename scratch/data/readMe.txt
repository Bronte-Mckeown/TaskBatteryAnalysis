'esq_demo_forPCA': mDES + demographics for the original N = 194 sample ready for applying PCA.

'gradscores_spearman_combinedmask_cortical_subcortical_forCCA.csv': gradscores, calculated via Spearmans rank using cortical+subcortical maps and the 'combined' mask,
ready for all-tasks CCA.

'gradscores_spearman_combinedmask_cortical_subcortical_forCCA_prediction.csv': gradscores, calculated via Spearmans rank using cortical+subcortical maps and the 'combined' mask,
ready for the CCA prediction script (averaged, 4 replication tasks only).

'esq_demo_replication': wrangled replication sample with mdes and demographics.

Note: in both final datasets, PS with missing age and gender dropped (4 in og sample, 5 in replication, leaving 190 PS in og, 96 in replication minus sub 086 = 95).

'task_perf_correctRT.csv' contains task performance data (response time calculated on correct trials only)

'replication_demo_n95.csv' contains replication sample's demographic data (age, sex, gender)

'og_demo_n190.csv' contains original sample's demographic data (age, sex, gender)

In gradients folder:
gradients folder contains gradients from StateSpace package (november 2023)

In taskmaps_masked:
contains nifti files of task maps with combined mask applied.
