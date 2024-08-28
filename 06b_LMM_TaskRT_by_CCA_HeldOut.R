"
x1 mixed LM to predict task response time by held-out CCAs.

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
# Set the path to the folder containing the files
folder_path <- "scratch/results/cca/holdOut"

# Get the list of files with the desired pattern
file_pattern <- "*_variates.csv"  # Change this to your desired file pattern
file_list <- list.files(path = folder_path, pattern = file_pattern, full.names = TRUE)

# Create an empty data frame to store the concatenated data
df <- data.frame()

# 1B sum_CCAloading_3 is flipped
# 2B-Face sum_CCAloading_3 is flipped
# Documentary sum_CCAloading_3 is flipped
# Friend sum_CCAloading_2 is flipped
# Friend sum_CCAloading_3 is flipped
# GoNoGo sum_CCAloading_4 is flipped
# HardMath sum_CCAloading_3 is flipped
# HardMath sum_CCAloading_4 is flipped
# SciFi sum_CCAloading_3 is flipped
# You sum_CCAloading_3 is flipped
# You sum_CCAloading_4 is flipped

# Iterate over the file list and read each file
for (file in file_list) {
  # Read the file
  data <- read.csv(file)
  
  task <- basename(str_extract(file, ".*(?=_variates)"))
  
  subset_data <- subset(data, Task_name == task)

  # 1B sum_CCAloading_3 is flipped
  if (task == '1B') {
    subset_data$sum_CCAloading_3 <- subset_data$sum_CCAloading_3 * -1
    print("1B sum_CCAloading_3 is flipped")
  }
  
  # 2B-Face sum_CCAloading_3 is flipped
  if (task == '2B-Face') {
    subset_data$sum_CCAloading_3 <- subset_data$sum_CCAloading_3 * -1
    print("2B-Face sum_CCAloading_3 is flipped")
  }
  
  # Documentary sum_CCAloading_3 is flipped
  if (task == 'Documentary') {
    subset_data$sum_CCAloading_3 <- subset_data$sum_CCAloading_3 * -1
    print("Documentary sum_CCAloading_3 is flipped")
  }
  
  # Friend sum_CCAloading_2 is flipped
  if (task == 'Friend') {
    subset_data$sum_CCAloading_2 <- subset_data$sum_CCAloading_2 * -1
    print("Friend sum_CCAloading_2 is flipped")
  }
  
  # Friend sum_CCAloading_3 is flipped
  if (task == 'Friend') {
    subset_data$sum_CCAloading_3 <- subset_data$sum_CCAloading_3 * -1
    print("Friend sum_CCAloading_3 is flipped")
  }
  
  # GoNoGo sum_CCAloading_4 is flipped
  if (task == 'GoNoGo') {
    subset_data$sum_CCAloading_4 <- subset_data$sum_CCAloading_4 * -1
    print("GoNoGo sum_CCAloading_4 is flipped")
  }
  
  # HardMath sum_CCAloading_3 is flipped
  if (task == 'HardMath') {
    subset_data$sum_CCAloading_3 <- subset_data$sum_CCAloading_3 * -1
    print("HardMath sum_CCAloading_3 is flipped")
  }
  
  # HardMath sum_CCAloading_4 is flipped
  if (task == 'HardMath') {
    subset_data$sum_CCAloading_4 <- subset_data$sum_CCAloading_4 * -1
    print("HardMath sum_CCAloading_4 is flipped")
  }
  
  # SciFi sum_CCAloading_3 is flipped
  if (task == 'SciFi') {
    subset_data$sum_CCAloading_3 <- subset_data$sum_CCAloading_3 * -1
    print("SciFi sum_CCAloading_3 is flipped")
  }
  
  # You sum_CCAloading_3 is flipped
  if (task == 'You') {
    subset_data$sum_CCAloading_3 <- subset_data$sum_CCAloading_3 * -1
    print("You sum_CCAloading_3 is flipped")
  }
  
  # You sum_CCAloading_4 is flipped
  if (task == 'You') {
    subset_data$sum_CCAloading_4 <- subset_data$sum_CCAloading_4 * -1
    print("You sum_CCAloading_4 is flipped")
  }
  
  # Concatenate the data vertically
  df <- rbind(df, subset_data)
}

# merge with task performance data
file_path <- file.path(getwd(), "scratch/data/task_perf_correctRT.csv")
perf_df = read.csv(file_path)

df$Id_number <- as.character(df$Id_number)
df$Task_name <- as.character(df$Task_name)

perf_df$Id_number <- as.character(perf_df$Id_number)
perf_df$Task_name <- as.character(perf_df$Task_name)

merged <-  merge(x=df, y=perf_df , by.x=c('Id_number','Task_name'), by.y=c('Id_number','Task_name'))

# average by person and task
df1 <- merged %>%
  group_by(Id_number, Task_name, Gender) %>%
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

############################# Removing outliers ################################
# zscore Response Time
df1$RT_z_score <- ave(df1$RT, df1$Task_name, FUN=scale)

# Identify cases with z-score above 2.5
df1$RT_outlier <- ifelse(df1$RT_z_score < 2.5, "Not Outlier", "Outlier")

# set these cases to zero
df1$Z_RT_outliers <- ifelse(df1$RT_outlier == "Outlier", 0, df1$RT_z_score)

##################### Setting up for linear mixed models #######################
# set contrasts to contr.sum

options(contrasts = c("contr.sum","contr.poly"))
options("contrasts")

########################## Saving results ######################################
# set file name for lmer text output
fp = "LMM_taskRT_CCA_holdout.txt"
# set current directory to results folder

# set current directory to results folder


results_dir <- file.path(getwd(), "scratch/results/cca/lmm/taskperf/rt")

file_path <- file.path(getwd(), "scratch/results/pca/lmm/taskperf/rt")
setwd(file_path)
############################# Models ###########################################
# set up list of dependent variables
dv <- c("Z_RT_outliers")

# first run all 3 models without lmertest
for(i in 1:length(dv)){
  model <- paste("model",i, sep="") # create model name

  # run model
  m <- lmer(as.formula(paste(dv[i],"~ Task_name + sum_CCAloading_1 + sum_CCAloading_2 +sum_CCAloading_3 + sum_CCAloading_4 +
  Task_name:sum_CCAloading_1 + Task_name:sum_CCAloading_2 + Task_name:sum_CCAloading_3 + Task_name:sum_CCAloading_4 +
   Age + Gender + (1|Id_number)")),
            data=df1)
  assign(model,m) # assign name to model

}

# ###########################  Summary tables ####################################
# # combine all normal model's parameter estimates into one table
# done before using lmertest to prevent conflict with satterthwaite
myfile2 <- file.path(results_dir,"LMM_RT_by_CCA_holdout_summary")
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
  m <- lmer(as.formula(paste(dv[i],"~ Task_name + sum_CCAloading_1 + sum_CCAloading_2 +sum_CCAloading_3 + sum_CCAloading_4 +
  Task_name:sum_CCAloading_1 + Task_name:sum_CCAloading_2 + Task_name:sum_CCAloading_3 + Task_name:sum_CCAloading_4 +
   Age + Gender + (1|Id_number)")),
            data=df1) 
  
  s <- summary(m) # create summary
  a <- anova(m) # create anova
  e <- emmeans(m, ~ Task_name,adjust = 'bonferroni', type = "response")
  
  
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
anova_list <- list(anova1)
myfile3 <- file.path(results_dir,"LMM_RT_sumVariates_holdout_anova")
tab_dfs(anova_list, file = myfile3,show.rownames = TRUE)

################################################################################
# confidence intervals for coefficients
# cis <- confint(model1)

############################ Probe Interactions ################################
# create new txt file for post hoc comparisons
fp2 <- "LMM_RT_cca_heldout_posthoc.txt"

## interaction between task and sum_CCAloading_4
# simple slopes
sum_CCAloading_4.slopes <- emtrends(model1, ~ Task_name, var="sum_CCAloading_4", adjust = 'bonferroni', infer = TRUE)
# contrast slopes
sum_CCAloading_4.contrasts <- emtrends(model1, pairwise ~ Task_name, var="sum_CCAloading_4", adjust = 'tukey', infer = TRUE)
# save to txt file
cat("Probing two-way interaction between task and positive engagement:\n", file = fp2, append = TRUE)
capture.output(sum_CCAloading_4.slopes, file = fp2, append = TRUE)
cat("\n\n\n", file = fp2, append = TRUE)
capture.output(sum_CCAloading_4.contrasts,file = fp2, append = TRUE)
cat("\n\n\n", file = fp2, append = TRUE)

################################################################################
# create function for interaction plots
fontsize = 6
interaction_plots <- function(data, x, x_label, y_label,x_raw, y_raw){
  ggplot(data = data, aes(x = x, y = yvar)) +
    geom_line() +
    facet_wrap(~tvar)+
    geom_ribbon(aes(ymax = UCL, ymin = LCL), alpha = 0.4) +
    geom_point(data = df1, aes(x = x_raw, y = y_raw),alpha =0.2, size = 0.05) +
    facet_wrap(~Task_name)+
    labs(x = x_label, y = y_label) +
    theme_light() +
    theme(axis.text.y=element_text(size = fontsize, color = "black"),
          axis.text.x=element_text(size = fontsize, color = "black"),
          axis.title.x=element_text(size = fontsize, color = "black"),
          axis.title.y=element_text(size = fontsize, color = "black"),
          strip.text = element_text(size = fontsize, color = "black"))
}

# interaction between task and sum_CCAloading_4
(mylist <- list(sum_CCAloading_4 = seq(-5.2, 6.4, by = 0.1), Task_name = c("EasyMath","HardMath" ,"FingerTap","GoNoGo",
                                                                       "1B","0B" ,
                                                                       "2B-Face","2B-Scene")))

sum_CCAloading_4.task.emmips <- emmip(model1, Task_name ~ sum_CCAloading_4, at = mylist, CIs = TRUE, plotit = FALSE)


# call interaction plot function for list of emmips set above and store each one
sum_CCAloading_4plot <- interaction_plots(sum_CCAloading_4.task.emmips, sum_CCAloading_4.task.emmips[, 2],
                                          "Positive Engagement", "Response Time (z-scored)",
                                          df1$sum_CCAloading_4, df1$Z_RT_outliers)
sum_CCAloading_4plot

ggsave(
  "LMM_RT_by_cca_heldout_interaction_rawpoints.tiff",
  sum_CCAloading_4plot, units = "cm",
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
  fitted.resid <- plot(fitted(models[[i]]),resid(models[[i]]),xlim=c(-2,2), ylim=c(-2,2))
  dev.off()
}
