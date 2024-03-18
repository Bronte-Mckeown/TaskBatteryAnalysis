# -*- coding: utf-8 -*-

'''
Correlate fMRI task maps with Gradient connectivity maps to produce state-space coordinates for each 
group-level task map in the task battery (14 maps).

Uses the 'combined mask' which excludes any regions which are not included in ANY
task map.

Uses cortical + subcortical gradient maps.

Spearman correlation.

'''
import os
from StateSpace import CorrelateTasksWithGradients

# get path to output dir 
outputdir = os.path.join(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0], 'TaskBatteryAnalysis/scratch/data')

# if os.path.exists(outputdir) == False: #If scratch doesn't exist, create.
#     os.mkdir(outputdir)

# Combined mask, cortical and subcortical with Spearman correlation
CorrelateTasksWithGradients.corrGroup('combinedmask_cortical_subcortical',
                                      'all',
                                      outputdir, corr_method = 'spearman',
                                      saveMaskedimgs = True,
                                      verbose = 1)



        







        



