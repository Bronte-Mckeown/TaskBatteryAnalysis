"
x5 mixed LMs to compare PCAs between tasks in task battery data.

Using held-out data to avoid circularity.

"
####################### Load libraries #########################################

library(emmeans) # saving predicted values for graphs and post-hoc contrasts
library(data.table) # data manipulation
library(ggthemes) # formatting graphs
library(ggpubr) # formatting graphs
library(patchwork) # putting plots together in one figure
library(report) # use to get written summary of results if required
library(sjPlot) # creating nice tables
library(dplyr)# data manipulation
library(interactions)# easy-view of interactions
library(lme4) # lmer
library(rstatix) # for neatening tables
library(stringr) # for matching string pattern

########################## Read in data ########################################
# Set the parent directory containing the held-out folders
parent_folder <- "C:/Users/bront/Documents/CanadaPostdoc/MegaProject/TaskBatteryAnalysis/scratch/results/pca/holdOut"

# Get the list of subfolders
subfolder_list <- list.dirs(path = parent_folder, full.names = FALSE, recursive = FALSE)

# Specify the pattern
pattern <- "_20112023_"

# Initialize an empty list to store the file paths
file_list <- c()

# Iterate over the subfolders and retrieve the file paths
for (subfolder in subfolder_list) {
  file_path <- file.path(parent_folder, subfolder, "csvdata", "projected_pca_scores.csv")
  if (grepl(pattern, subfolder) && file.exists(file_path)) {
    file_list <- c(file_list, file_path)
  }
}

# Print the resulting file list
print(file_list)

# Create an empty data frame to store the concatenated data
df1 <- data.frame()

# Iterate over the file list and read each file
for (file in file_list) {
  # Read the file
  data <- read.csv(file)
  
  # select rows that match substring
  # Extract the part of the string before "_20112023"
  task <- basename(str_extract(file, ".*(?=_20112023_)"))
  
  print (task) # print task name
  
  # subset data to select rows matching current task
  subset_data <- subset(data, Task_name == task)
  
  # multiply flipped components by -1
  # Friend PCA_1 is flipped
  if (task == 'Friend') {
    subset_data$PCA_1 <- subset_data$PCA_1 * -1
    print("Friend PCA_1 is flipped")
  }
  
  # Memory PCA_1 is flipped
  if (task == 'Memory') {
    subset_data$PCA_1 <- subset_data$PCA_1 * -1
    print("Memory PCA_1 is flipped")
  }
  
  # Documentary PCA_1 is flipped
  if (task == 'Documentary') {
    subset_data$PCA_1 <- subset_data$PCA_1 * -1
    print("Documentary PCA_1 is flipped")
  }
  
  # SciFi PCA_1 is flipped
  if (task == 'SciFi') {
    subset_data$PCA_1 <- subset_data$PCA_1 * -1
    print("SciFi PCA_1 is flipped")
  }
  
  # 2B-Face PCA_1 is flipped
  if (task == '2B-Face') {
    subset_data$PCA_1 <- subset_data$PCA_1 * -1
    print("2B-Face PCA_1 is flipped")
  }
  
  # You PCA_1 is flipped
  if (task == 'You') {
    subset_data$PCA_1 <- subset_data$PCA_1 * -1
    print("You PCA_1 is flipped")
  }
    
  # You PCA_4 is flipped
    if (task == 'You') {
      subset_data$PCA_4 <- subset_data$PCA_4 * -1
      print("You PCA_4 is flipped")
    }
  
  #Concatenate the data vertically
  df1 <- rbind(df1, subset_data)
}


######################### Data prep ############################################
## Set fixed factors: Task_name 
df1$Task_name <- factor(df1$Task_name,levels = c("EasyMath", "HardMath",
                                                 "FingerTap", "GoNoGo",
                                                 "Friend", "You",
                                                 "Documentary", "SciFi",
                                                 "1B", "0B",
                                                 "Read", "Memory",
                                                 "2B-Face", "2B-Scene"),
                         ordered = FALSE) #set order to False.
#to check it's worked, run this command and see print out.
levels(df1$Task_name)

# Gender
df1$Gender <- as.factor(df1$Gender)
levels(df1$Gender)

# Id_number (participant)
df1$Id_number <- as.factor(df1$Id_number)

##################### Setting up for linear mixed models #######################
# set contrasts to contr.sum

options(contrasts = c("contr.sum","contr.poly"))
options("contrasts")

########################## Saving results ######################################
# set file name for lmer text output
fp = "LMM_pca_by_task_holdout.txt"

# set current directory to results folder
setwd("C:/Users/bront/Documents/CanadaPostdoc/MegaProject/MegaProject/scratch/results/pca/lmm")

results_dir <- "C:/Users/bront/Documents/CanadaPostdoc/MegaProject/MegaProject/scratch/results/pca/lmm"

############################# Models ###########################################
# set up list of dependent variables
dv <- c("PCA_0",
        "PCA_1",
        "PCA_2",
        "PCA_3",
        "PCA_4"
        )

# first run all 3 models without lmertest
for(i in 1:length(dv)){
   model <- paste("model",i, sep="") # create model name

   # run model
   m <- lmer(as.formula(paste(dv[i],"~ Task_name + Age + Gender +
                              (1|Id_number)")), data=df1)
   assign(model,m) # assign name to model
 }

###########################  Summary tables ####################################
# # combine all normal model's parameter estimates into one table
# # done before using lmertest to prevent conflict with satterthwaite
myfile2 <- file.path(results_dir,"LMM_pcascores_holdout_summary")
tab_model(model1,model2,model3, model4,model5, file = myfile2, df.method = "satterthwaite",
          p.val = "satterthwaite",show.df = FALSE,
          show.r2 = FALSE,
          show.stat = TRUE, show.icc = FALSE,
          show.re.var = TRUE)

##################### re-run models with lmertest ##############################
library(lmerTest) # for regular p-values for LMMs (f-tests)
# globally set lmertest.limit calculate df, also set to sattherthwaite
emm_options(lmerTest.limit = 7220, lmer.df = "satterthwaite")

# run all 3 models using each DV from list using loop
for(i in 1:length(dv)){
  model <- paste("model",i, sep="") # create names of models
  summ <- paste("summary",i, sep = "") # create names of summaries
  an <- paste("anova",i, sep = "") #  create names of anovas
  emmean <- paste("emmean",i, sep = "") # create name of emmeans
  
  # run model
  m <- lmer(as.formula(paste(dv[i],"~ Task_name + Age + Gender + (1|Id_number)")),
            data=df1) 
  
  s <- summary(m) # create summary
  a <- anova(m) # create anova
  # create emmeans
  e <- emmeans(m, ~ Task_name, adjust = 'bonferroni', type = "response")
  
  
  # format anova tables
  a = subset(a, select = -c(`Sum Sq`,`Mean Sq`) )
  a = as.data.frame(a)
  
  a[, c(2)] <- round(a[, c(2)], 0)
  
  a[, c(3)] <- round(a[, c(3)], 2)
  
  colnames(a)[4]  <- "p" 
  
  a <- p_format(a, digits = 2, leading.zero = FALSE, trailing.zero = TRUE, accuracy = .001)
  
  assign(model,m) # assign model to model name
  assign(summ,s) # assign summary to summary name
  assign(an, a) # assign anova to anova name
  assign(emmean, e) #assign emmean to emmean name
  
  #save outputs to txt file
  capture.output(s,file = fp, append = TRUE)
  cat("\n\n\n", file = fp, append = TRUE)
  capture.output(a,file = fp, append = TRUE)
  cat("\n\n\n", file = fp, append = TRUE)
  capture.output(e,file = fp, append = TRUE)
  cat("\n\n\n", file = fp, append = TRUE)
  
} 

############################ ANOVA tables ######################################
# save anova tables
anova_list <- list(anova1, anova2, anova3, anova4, anova5)
myfile3 <- file.path(results_dir,"LMM_pcascores_holdout_anovaAll")
tab_dfs(anova_list, file = myfile3,show.rownames = TRUE)

############################ Save EMMEANS as csv for word clouds ###############
emmean1_df <- as.data.frame(emmean1)
colnames(emmean1_df) <- c("Task_name", "emmean1")

emmean2_df <- as.data.frame(emmean2)
colnames(emmean2_df) <- c("Task_name", "emmean2")

emmean3_df <- as.data.frame(emmean3)
colnames(emmean3_df) <- c("Task_name", "emmean3")

emmean4_df <- as.data.frame(emmean4)
colnames(emmean4_df) <- c("Task_name", "emmean4")

emmean5_df <- as.data.frame(emmean5)
colnames(emmean5_df) <- c("Task_name", "emmean5")

# Select only the 'Task_name' and 'emmean' columns
emmean1_df <- select(emmean1_df, Task_name, emmean1)
emmean2_df <- select(emmean2_df, Task_name, emmean2)
emmean3_df <- select(emmean3_df, Task_name, emmean3)
emmean4_df <- select(emmean4_df, Task_name, emmean4)
emmean5_df <- select(emmean5_df, Task_name, emmean5)

# Merge the emmeans dataframes on 'Task_name'
merged_df <- merge(emmean1_df, emmean2_df, by = "Task_name", all = TRUE)
merged_df <- merge(merged_df, emmean3_df, by = "Task_name", all = TRUE)
merged_df <- merge(merged_df, emmean4_df, by = "Task_name", all = TRUE)
merged_df <- merge(merged_df, emmean5_df, by = "Task_name", all = TRUE)

names(merged_df) <- NULL

# save as csv
write.csv(merged_df, file = "pcascores_emmeans.csv", row.names = FALSE)

######################## Probe main effect of 'Task_name' ######################
# # create new txt file for post hoc comparisons
# fp2 <- "LMM_pca_by_task_holdout_posthoc.txt"
# 
# Task_name1.contrasts <- pairs(emmean1, adjust = "bonferroni", infer = TRUE)
# Task_name1.contrasts <- as.data.frame(Task_name1.contrasts)
# Task_name1.contrasts <- Task_name1.contrasts[order(Task_name1.contrasts$t.ratio), ]
# Task_name1.contrasts
# 
# Task_name2.contrasts <- pairs(emmean2, adjust = "bonferroni", infer = TRUE)
# Task_name2.contrasts <- as.data.frame(Task_name2.contrasts)
# Task_name2.contrasts <- Task_name2.contrasts[order(Task_name2.contrasts$t.ratio), ]
# Task_name2.contrasts
# 
# Task_name3.contrasts <- pairs(emmean3, adjust = "bonferroni", infer = TRUE)
# Task_name3.contrasts <- as.data.frame(Task_name3.contrasts)
# Task_name3.contrasts <- Task_name3.contrasts[order(Task_name3.contrasts$t.ratio), ]
# Task_name3.contrasts
# 
# Task_name4.contrasts <- pairs(emmean4, adjust = "bonferroni", infer = TRUE)
# Task_name4.contrasts <- as.data.frame(Task_name4.contrasts)
# Task_name4.contrasts <- Task_name4.contrasts[order(Task_name4.contrasts$t.ratio), ]
# Task_name4.contrasts
# 
# Task_name5.contrasts <- pairs(emmean5, adjust = "bonferroni", infer = TRUE)
# Task_name5.contrasts <- as.data.frame(Task_name5.contrasts)
# Task_name5.contrasts <- Task_name5.contrasts[order(Task_name5.contrasts$t.ratio), ]
# Task_name5.contrasts
# 
# # save to txt file
# cat("Comparing PCA1:\n", file = fp2, append = TRUE)
# capture.output(Task_name1.contrasts,file = fp2, append = TRUE)
# cat("\n\n\n", file = fp2, append = TRUE)
# 
# cat("Comparing PCA2:\n", file = fp2, append = TRUE)
# capture.output(Task_name2.contrasts,file = fp2, append = TRUE)
# cat("\n\n\n", file = fp2, append = TRUE)
# 
# cat("Comparing PCA3:\n", file = fp2, append = TRUE)
# capture.output(Task_name3.contrasts,file = fp2, append = TRUE)
# cat("\n\n\n", file = fp2, append = TRUE)
# 
# cat("Comparing PCA4:\n", file = fp2, append = TRUE)
# capture.output(Task_name4.contrasts,file = fp2, append = TRUE)
# cat("\n\n\n", file = fp2, append = TRUE)
# 
# cat("Comparing PCA5:\n", file = fp2, append = TRUE)
# capture.output(Task_name5.contrasts,file = fp2, append = TRUE)
# cat("\n\n\n", file = fp2, append = TRUE)

########################## Bar charts ##########################################

# set up list with names of emmeans (predicted means)
list <- c("emmean1", "emmean2","emmean3", "emmean4","emmean5")

# set font size
fontsize = 6

# function for making plots
myplot2 <- function(data, title){
  # x axis = Task_name, y axis = emmean, bars = gradient
  ggplot(summary(data), aes(x = emmean, y = Task_name)) +
    theme_light() +
    geom_bar(stat="identity",width = 2/.pt, position="dodge", color = "black" ,fill = "grey", size = 0.5/.pt) +
    xlim(-2.5, 2.5) +
    theme(axis.text.y = element_blank(),
          axis.text.x=element_text(size = fontsize,color = "black"),
          axis.title.x=element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.x = element_blank(),
          axis.ticks.y = element_blank(),
          plot.margin = margin(0, 0, 0, 0, "cm")) +
    # add error bars
    geom_errorbar(position=position_dodge(.6/.pt),width=0.25/.pt,size = 0.5/.pt, 
                  aes(xmax=upper.CL, xmin=lower.CL),alpha=1)
}

# call function for list of emmeans set above and store each one (bar1, bar2 etc)
for(i in seq_along(list)){
  bar <- paste("bar",i, sep="")
  b <- myplot2(get(list[i]), titles[i])
  assign(bar, b)
}

# add text to bar1
bar1 <- bar1 + theme(axis.text.y=element_text(size = fontsize,color = "black"))

# put together
all_plots <- ((bar1)|(bar2)|(bar3)|(bar4)|(bar5))&
  geom_vline(xintercept = 0, size = 0.2/.pt)
all_plots

# save out
ggsave(
  "LMM_pcascores_Holdout_barchart.tiff",
  all_plots, units = "cm",
  width = 8.7,
  height = 4,
  dpi = 1000)

############################# Assumptions ######################################
models = c(model1, model2, model3, model4, model5)
#QQ plots
for (i in seq_along(models)) {
  jpeg(paste("qq_plot", i, ".png", sep = ""))
  qq <- qqnorm(resid(models[[i]]))
  dev.off()
}

#histograms
for (i in seq_along(models)) {
  jpeg(paste("hist_plot", i, ".png", sep = ""))
  hist <- hist(resid(models[[i]]))
  dev.off()
}

#residual plots
for (i in seq_along(models)) {
  jpeg(paste("fitted_residual_plot", i, ".png", sep = ""))
  fitted.resid <- plot(fitted(models[[i]]),resid(models[[i]]),xlim=c(-3,3), ylim=c(-3,3))
  dev.off()
}
