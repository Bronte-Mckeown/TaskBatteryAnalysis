"
1. 3d plot of:
  1. real tasks in gradient space
  2. predicted tasks in gradient space
  
BUT this is the supplementary analysis which trained on 10 tasks.
"
####################### Load libraries #########################################

library(data.table) # data manipulation
library(ggthemes) # formatting graphs
library(ggpubr) # formatting graphs
library(patchwork) # putting plots together in one figure
library(sjPlot) # creating supplementary tables
library(dplyr)# data manipulation
library(plotly)
library(ggrepel)
library("plot3D")

########################## Read in data ########################################
#set current working directory
# setwd("C:\\Users\\bront\\Documents\\CanadaPostdoc\\MegaProject\\TaskBatteryAnalysis\\scratch\\results\\cca\\allTasks")
file_path_1 = file.path(getwd(), "scratch/results/cca/allTasks/reptasks_cca_predictgrads_10tasks.csv")
file_path_2 = file.path(getwd(), "scratch/results/cca/allTasks/reptasks_cca_realgrads.csv")
#read in csv file
df_predicted <- read.csv(file_path_1, na.strings=c(""," ","NA", "nan"))
df_actual <- read.csv(file_path_2, na.strings=c(""," ","NA", "nan"))

################################ 3d plot #######################################
# drop cols
df_actual <- df_actual[, -c(1, 3:7)]

# group averages
means_actual <- aggregate(df_actual[, 2:4], list(df_actual$Task_name), mean)

means_actual <- setNames(means_actual,  
                   c("Task_name","Gradient 1", "Gradient 2", "Gradient 3"))

means_pred <- aggregate(df_predicted[, 2:4], list(df_predicted$Task_name), mean)

means_pred <- setNames(means_pred,  
                         c("Task_name","Gradient 1", "Gradient 2", "Gradient 3"))


# Add the "suffix" column to means_pred
means_pred$suffix <- "Predicted"

# Add the "suffix" column to means_actual
means_actual$suffix <- "True"


stacked_data <- rbind(means_pred, means_actual)

################################################################################
setwd(file.path(getwd(), "scratch\\results\\cca"))

png("scatterplot3d_predict_10tasks.png",  width     = 2,
    height    = 2,
    units     = "in",
    res       = 1200,
    pointsize = 4)  # Adjust width and height as needed

with(stacked_data, text3D(`Gradient 1`,`Gradient 2`, `Gradient 3`, 
                          col=c(rep("red", 4), rep("black", 4)),
                          xlim = c(-2.5, 2.5), ylim = c(-2.5, 2.5), zlim = c(-2.5, 2.5),
                 labels = Task_name,
                 theta = 30, phi = 10,
                 xlab = "", ylab = "", zlab = "", 
                 main = "", cex = 1.4, 
                 bty = "g",  d = 2,
                 colkey = TRUE,
                 adj = 0.5))

dev.off()

