"
x1 mixed LM to predict task RT (correct trials only) by PCAs.

"
####################### Load libraries #########################################

library(emmeans) # saving predicted values for graphs and post-hoc contrasts
library(data.table) # data manipulation
library(ggthemes) # formatting graphs
library(ggpubr) # formatting graphs
library(patchwork) # putting plots together in one figure
library(report) # use to get written summary of results if required
library(sjPlot) # creating supplementary tables
library(dplyr)# data manipulation
library(interactions)# easy-view of interactions
library(lme4) # lmer
library(rstatix) # for neatening tables
library(stringr)

########################## Read in data ########################################
# Set the parent directory containing the folders
parent_folder <- "C:/Users/bront/Documents/CanadaPostdoc/MegaProject/TaskBatteryAnalysis/scratch/results/pca/holdOut"

# Get the list of subfolders
subfolder_list <- list.dirs(path = parent_folder, full.names = FALSE, recursive = FALSE)

# Specify the pattern for the changing part of the file path
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
df <- data.frame()

# Iterate over the file list and read each file
for (file in file_list) {
  # Read the file
  data <- read.csv(file)
  
  # select rows that match substring
  # Extract the part of the string before "_20112023"
  task <- basename(str_extract(file, ".*(?=_20112023_)"))
  
  print (task)
  
  subset_data <- subset(data, Task_name == task)
  
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
  df <- rbind(df, subset_data)
}

# merge with task performance data
perf_df = read.csv("C:/Users/bront/Documents/CanadaPostdoc/MegaProject/TaskBatteryAnalysis/scratch/data/task_perf_correctRT.csv")

df$Id_number <- as.character(df$Id_number)
df$Task_name <- as.character(df$Task_name)

perf_df$Id_number <- as.character(perf_df$Id_number)
perf_df$Task_name <- as.character(perf_df$Task_name)

merged <-  merge(x=df, y=perf_df , by.x=c('Id_number','Task_name'), by.y=c('Id_number','Task_name'))

# average by person and task
df1 <- merged %>%
  group_by(Id_number, Task_name, Gender, Language, Country) %>%
  summarise_all(mean, na.rm = TRUE)

######################### Data prep ############################################
## Set fixed factors: Task_name 
df1$Task_name <- factor(df1$Task_name,levels = c("EasyMath", "HardMath",
                                                 "FingerTap", "GoNoGo",
                                                 "1B", "0B",
                                                 "2B-Face", "2B-Scene"),
                        ordered = FALSE) #set order to False.
#to check it's worked, run this command and see print out.
levels(df1$Task_name)


# Gender
df1$Gender <- as.factor(df1$Gender)
levels(df1$Gender)

# Id_number (participant)
df1$Id_number <- as.factor(df1$Id_number)

#check number of missing values for rt (8)
sum(is.na(df1$RT))

############################# Removing outliers ################################
# zscore RT
df1$rt_z_score <- ave(df1$RT, df1$Task_name, FUN=scale)

# Identify cases with z-score above or below 2.5
df1$rt_outlier <- ifelse(df1$rt_z_score > -2.5 & df1$rt_z_score < 2.5, "Not Outlier", "Outlier")

# set these cases to zero
df1$Z_rt_outliers <- ifelse(df1$rt_outlier == "Outlier", 0, df1$rt_z_score)

##################### Setting up for linear mixed models #######################
# set contrasts to contr.sum

options(contrasts = c("contr.sum","contr.poly"))
options("contrasts")

########################## Saving results ######################################
# set file name for lmer text output
fp = "LMM_taskrt_PCA_holdout.txt"

# set current directory to results folder
setwd("C:/Users/bront/Documents/CanadaPostdoc/MegaProject/TaskBatteryAnalysis/scratch/results/pca/lmm/taskperf/rt")

results_dir <- "C:/Users/bront/Documents/CanadaPostdoc/MegaProject/TaskBatteryAnalysis/scratch/results/pca/lmm/taskperf/rt"

############################# Models ###########################################
# set up list of dependent variables
dv <- c("Z_rt_outliers")

# first run all 3 models without lmertest to get accurate summary table below
for(i in 1:length(dv)){
  model <- paste("model",i, sep="") # create model name

  # run model
  m <- lmer(as.formula(paste(dv[i],"~ Task_name + PCA_0 + PCA_1 + PCA_2 + PCA_3 + PCA_4 +
  Task_name:PCA_0 + Task_name:PCA_1 + Task_name:PCA_2 + Task_name:PCA_3 + Task_name:PCA_4 + 
   Age + Gender + (1|Id_number)")),
            data=df1)
  assign(model,m) # assign name to model

}

###########################  Summary tables ####################################
# combine all model's parameter estimates into one table
# done before using lmertest to prevent conflict with satterthwaite
myfile2 <- file.path(results_dir,"LMM_rt_pca_holdout_summary")
tab_model(model1, file = myfile2, df.method = "satterthwaite",
          p.val = "satterthwaite",show.df = FALSE,
          show.r2 = FALSE,
          show.stat = TRUE, show.icc = FALSE,
          show.re.var = TRUE)

##################### re-run models with lmertest ##############################
library(lmerTest) # for regular p-values for LMMs (f-tests)
# globally set lmertest.limit to calculate df
# also set to sattherthwaite
emm_options(lmerTest.limit = 1520, lmer.df = "satterthwaite")

# run all 3 models using each DV from list using loop
for(i in 1:length(dv)){
  model <- paste("model",i, sep="") # create names of models
  summ <- paste("summary",i, sep = "") # create names of summaries
  an <- paste("anova",i, sep = "") #  create names of anovas
  emmean <- paste("emmean",i, sep = "") # create name of emmeans
  
  # run model
  m <- lmer(as.formula(paste(dv[i],"~ Task_name + PCA_0 + PCA_1 + PCA_2 + PCA_3 + PCA_4 +
  Task_name:PCA_0 + Task_name:PCA_1 + Task_name:PCA_2 + Task_name:PCA_3 + Task_name:PCA_4 + 
   Age + Gender + (1|Id_number)")),
            data=df1) 
  
  s <- summary(m) # create summary
  a <- anova(m) # create anova
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
  #capture.output(s,file = fp, append = TRUE)
  #cat("\n\n\n", file = fp, append = TRUE)
  #capture.output(a,file = fp, append = TRUE)
  #cat("\n\n\n", file = fp, append = TRUE)
  #capture.output(e,file = fp, append = TRUE)
  #cat("\n\n\n", file = fp, append = TRUE)
  
} 

############################ ANOVA tables ######################################
# save anova tables
anova_list <- list(anova1)
myfile3 <- file.path(results_dir,"LMM_rt_pca_holdout_anovaAll")
tab_dfs(anova_list, file = myfile3,show.rownames = TRUE)

################################################################################
# get CIs for coefficients for main effects
# cis <- confint(model1) 

######################## Probe Interactions  ###################################
# create new txt file for post hoc comparisons
fp2 <- "LMM_rt_pca_holdout_posthoc.txt"

## interaction between task and PCA_1
# simple slopes
task_pc1.slopes <- emtrends(model1, ~ Task_name, var="PCA_1", adjust = 'bonferroni', infer = TRUE)
# contrast slopes
task_pc1.contrasts <- emtrends(model1, pairwise ~ Task_name, var="PCA_1", adjust = 'bonferroni', infer = TRUE)
# save to txt file
cat("Probing two-way interaction between task and intrusive distraction:\n", file = fp2, append = TRUE)
capture.output(task_pc1.slopes, file = fp2, append = TRUE)
cat("\n\n\n", file = fp2, append = TRUE)
capture.output(task_pc1.contrasts,file = fp2, append = TRUE)
cat("\n\n\n", file = fp2, append = TRUE)

########################### Interaction plots ##################################
fontsize = 6

# create function for interaction plots
interaction_plots <- function(data, x, x_label, y_label, x_raw, y_raw){
  ggplot(data = data, aes(x = x, y = yvar)) +
    geom_line() +
    facet_wrap(~tvar, ncol = 2)+
    geom_ribbon(aes(ymax = UCL, ymin = LCL), alpha = 0.4) +
    geom_point(data = df1, aes(x = x_raw, y = y_raw),alpha =0.2, size = 0.05) +
    facet_wrap(~Task_name, ncol = 2)+
    labs(x = x_label, y = y_label) +
    theme_light() +
    theme(axis.text.y=element_text(size = fontsize, color = "black"),
          axis.text.x=element_text(size = fontsize, color = "black"),
          axis.title.x=element_text(size = fontsize, color = "black"),
          axis.title.y=element_text(size = fontsize, color = "black"),
          strip.text = element_text(size = fontsize, color = "black"))
}
# interaction between task and PCA_1
(mylist <- list(PCA_1 = seq(-3.5, 3.5, by = 0.1), Task_name = c("EasyMath","HardMath" ,"FingerTap","GoNoGo",
                                                            "1B","0B" ,
                                                            "2B-Face","2B-Scene")))


PCA_1.task.emmips <- emmip(model1, Task_name ~ PCA_1, at = mylist, CIs = TRUE, plotit = FALSE)


# call interaction plot function for list of emmips set above and store each one
PCA_1plot <- interaction_plots(PCA_1.task.emmips, PCA_1.task.emmips[, 2],
                               "Intrusive Distraction", "Response Time (z-scored)",
                               df1$PCA_1, df1$Z_rt_outliers)
PCA_1plot

# save plots as tiff
ggsave(
  "LMM_rt_by_pca_Holdout_interactions_rawpoint.tiff",
  PCA_1plot, units = "cm",
  width = 6,
  height = 10,
  dpi = 1000, 
)

############################# Assumptions ######################################
models = c(model1)
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
  fitted.resid <- plot(fitted(models[[i]]),resid(models[[i]]),xlim=c(-1,1), ylim=c(-1,1))
  dev.off()
}
