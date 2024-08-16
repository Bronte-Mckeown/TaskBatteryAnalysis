####################### Load libraries #########################################
library(patchwork) # putting plots together in one figure
library(ggplot2)
library(dplyr)# data manipulation
library(tidyr)
library(ggrepel)
library(plotly)
library(plot3D)

########################## Read in data ########################################
#set current working directory
# setwd("C:\\Users\\bront\\Documents\\CanadaPostdoc\\MegaProject\\TaskBatteryAnalysis\\scratch\\data\\source")
file_path <- file.path(getwd(), "scratch/data/gradscores_spearman_combinedmask_cortical_subcortical.csv")
print(file_path)
#read in csv file
df1 <- read.csv(file_path, na.strings=c(""," ","NA", "nan"))

# Define the mapping between task names
task_name_mapping <- c(
  'you'           = 'You',
  'friend'        = 'Friend',
  'twoBackFaces'  = '2B-Face',
  'twoBackScenes' = '2B-Scene',
  'easyMath'      = 'EasyMath',
  'hardMath'      = 'HardMath',
  'reading'       = 'Read\n',
  'memory'        = 'Memory',
  'oneBack'       = '1B',
  'zeroBack'      = '0B',
  'gonogo'        = 'GoNoGo',
  'fingerTapping' = 'FingerTap',
  'movieIncept'   = 'SciFi',
  'movieBridge'   = 'Doc'
)

# Apply the mapping
df1 <- df1 %>%
  mutate(Task_name = task_name_mapping[Task_name])

# change directory to results
setwd(file.path(getwd(),"scratch\\results"))

# save out
png("grad_scatterplot3d_plot.png",  width     = 6.2,
    height    = 8,
    units     = "cm",
    res       = 1200,
    pointsize = 7)

with(df1, text3D(gradient1_cortical_subcortical,gradient2_cortical_subcortical,gradient3_cortical_subcortical, 
                 labels = Task_name,
                 theta = 70, phi = 10,
                 xlab = "", ylab = "", zlab = "", 
                 xlim = c(-.4, .4), ylim = c(-.4, .4), zlim = c(-.4, .4),
                 main = "", cex = 1, 
                 bty = "g", d = 2, 
                 adj = 0.5))


dev.off()

