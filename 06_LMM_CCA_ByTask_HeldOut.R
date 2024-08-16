"
CCA dimensions by task.

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
df1 <- data.frame()

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
  df1 <- rbind(df1, subset_data)
}


######################### Data prep ############################################
## Set fixed factors: Task_name
df1$Task_name <- factor(df1$Task_name,
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
fp = "LMM_summedvariates_holdout.txt"

if (!dir.exists(file.path(getwd(), "scratch/results/cca/lmm"))) {
  # Create the directory if it doesn't exist
  dir.create(file.path(getwd(), "scratch/results/cca/lmm"), recursive = TRUE)
}

results_dir <- file.path(getwd(), "scratch/results/cca/lmm")
# set current directory to results folder
setwd(file.path(getwd(), "scratch/results/cca/lmm"))



############################# Models ###########################################
# set up list of dependent variables
dv <- c("sum_CCAloading_1",
        "sum_CCAloading_2",
        "sum_CCAloading_3",
        "sum_CCAloading_4"
        )

# first run all 3 models without lmertest
for(i in 1:length(dv)){
  model <- paste("model",i, sep="") # create model name

  # run model
  m <- lmer(as.formula(paste(dv[i],"~ Task_name + Age + Gender +
                             (1|Id_number)")),
            data=df1)
  assign(model,m) # assign name to model
}

# ###########################  Summary tables ####################################
# # combine all normal model's parameter estimates into one table
# # done before using lmertest to prevent conflict with satterthwaite
myfile2 <- file.path(results_dir,"LMM_CCAvariates_holdout_summary")
tab_model(model1,model2,model3, model4, file = myfile2, df.method = "satterthwaite",
          p.val = "satterthwaite",show.df = FALSE,
          show.r2 = FALSE,
          show.stat = TRUE, show.icc = FALSE,
          show.re.var = TRUE)

##################### re-run models with lmertest ##############################
library(lmerTest) # for regular p-values for LMMs (f-tests)
# globally set lmertest.limit to calculate df
# also set to sattherthwaite
emm_options(lmerTest.limit = 7220, lmer.df = "satterthwaite")

# run all 3 models using each DV from list using loop
for(i in 1:length(dv)){
  model <- paste("model",i, sep="") # create names of models
  summ <- paste("summary",i, sep = "") # create names of summaries
  an <- paste("anova",i, sep = "") #  create names of anovas
  emmean <- paste("emmean",i, sep = "") # create name of emmeans
  
  # run model
  m <- lmer(as.formula(paste(dv[i],"~ Task_name + Age + Gender +
                             (1|Id_number)")),
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
  #capture.output(s,file = fp, append = TRUE)
  #cat("\n\n\n", file = fp, append = TRUE)
  #capture.output(a,file = fp, append = TRUE)
  #cat("\n\n\n", file = fp, append = TRUE)
  #capture.output(e,file = fp, append = TRUE)
  #cat("\n\n\n", file = fp, append = TRUE)
  
} 

############################ ANOVA tables ######################################
# save anova tables
anova_list <- list(anova1, anova2, anova3, anova4)
myfile3 <- file.path(results_dir,"LMM_sumVariates_holdout_anovaAll")
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

# Select only the 'Task_name' and 'emmean' columns
emmean1_df <- select(emmean1_df, Task_name, emmean1)
emmean2_df <- select(emmean2_df, Task_name, emmean2)
emmean3_df <- select(emmean3_df, Task_name, emmean3)
emmean4_df <- select(emmean4_df, Task_name, emmean4)

# Merge the emmeans dataframes on 'Task_name'
merged_df <- merge(emmean1_df, emmean2_df, by = "Task_name", all = TRUE)
merged_df <- merge(merged_df, emmean3_df, by = "Task_name", all = TRUE)
merged_df <- merge(merged_df, emmean4_df, by = "Task_name", all = TRUE)

names(merged_df) <- NULL

write.csv(merged_df, file = "summed_cca_emmeans.csv", row.names = FALSE)

######################## Probe main effect of 'Task_name' ###########################
# # create new txt file for post hoc comparisons
# fp2 <- "LMM_Xvariates_holdout_posthoc.txt"
# 
# # compare Task_names along each gradient (target + non-target maps)
# Task_name1.contrasts <- pairs(emmean1, adjust = "bonferroni", infer = TRUE)
# Task_name1.contrasts
# Task_name2.contrasts <- pairs(emmean2, adjust = "bonferroni", infer = TRUE)
# Task_name2.contrasts
# Task_name3.contrasts <- pairs(emmean3, adjust = "bonferroni", infer = TRUE)
# Task_name3.contrasts
# Task_name4.contrasts <- pairs(emmean4, adjust = "bonferroni", infer = TRUE)
# Task_name4.contrasts
# 
# # save to txt file
# cat("Comparing CCA dimension 1 variates:\n", file = fp2, append = TRUE)
# capture.output(Task_name1.contrasts,file = fp2, append = TRUE)
# cat("\n\n\n", file = fp2, append = TRUE)
# 
# cat("Comparing CCA dimension 2 variates:\n", file = fp2, append = TRUE)
# capture.output(Task_name2.contrasts,file = fp2, append = TRUE)
# cat("\n\n\n", file = fp2, append = TRUE)
# 
# cat("Comparing CCA dimension 3 variates:\n", file = fp2, append = TRUE)
# capture.output(Task_name3.contrasts,file = fp2, append = TRUE)
# cat("\n\n\n", file = fp2, append = TRUE)
# 
# cat("Comparing CCA dimension 4 variates:\n", file = fp2, append = TRUE)
# capture.output(Task_name4.contrasts,file = fp2, append = TRUE)
# cat("\n\n\n", file = fp2, append = TRUE)

########################## Bar charts ##########################################
# set up list with names of emmeans (predicted means)
# Set up list with names of emmeans (predicted means)
list <- c("emmean1", "emmean2", "emmean3", "emmean4")
task_order <- c("EasyMath", "HardMath",
                                                 "FingerTap", "GoNoGo",
                                                 "Friend", "You",
                                                 "Documentary", "SciFi",
                                                 "1B", "0B",
                                                 "Read", "Memory",
                                                 "2B-Face", "2B-Scene")  # Replace with your actual task names
task_order <- rev(task_order)
fontsize = 6

# Function for making plots
myplot <- function(data, title, order) {
  # Extract data from emmGrid object
  data_summary <- summary(data)
  
  # Access Task_name column
  task_names <- data_summary$Task_name
  
  # Convert Task_name to a factor with the specified order
  task_names <- factor(task_names, levels = order)
  
  # Update the data frame with the ordered Task_name
  data_summary$Task_name <- task_names
  
  # x axis = Task_name, y axis = emmean, bars = gradient
  ggplot(data_summary, aes(x = emmean, y = Task_name)) +
    theme_light() +
    geom_bar(stat = "identity", width = 2/.pt, position = "dodge", color = "black", fill = "grey", size = 0.5/.pt) +
    xlim(-7, 7) +
    theme(axis.text.y = element_blank(),
          axis.text.x = element_text(size = fontsize, color = "black"),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.x = element_blank(),
          plot.margin = margin(0, 0, 0, 0, "cm")) +
    # Add error bars
    geom_errorbar(position = position_dodge(.6/.pt), width = 0.25/.pt, size = 0.5/.pt, 
                  aes(xmax = upper.CL, xmin = lower.CL), alpha = 1)
}

# Call function for list of emmeans set above and store each one (bar1, bar2 etc)
for (i in seq_along(list)) {
  bar <- paste("bar", i, sep = "")
  b <- myplot(get(list[i]), titles[i], task_order)
  assign(bar, b)
}

bar1 <- bar1 + theme(axis.text.y = element_text(size = fontsize, color = "black"))

# Put together
all_plots <- ((bar1) | (bar2) | (bar3) | (bar4)) &
  geom_vline(xintercept = 0, size = 0.2/.pt)
all_plots

# Save plots as tiff
ggsave(
  "LMM_sumVariates_Holdout_barchart.tiff",
  all_plots, units = "cm",
  width = 8.7,
  height = 4,
  dpi = 1000
)


# box plots
box1 <- ggplot(df1, aes(x=sum_CCAloading_1, y=Task_name)) + 
  geom_boxplot(outlier.size = 0.01)&theme(axis.text.y=element_text(size = fontsize,color = "black"))
box2 <- ggplot(df1, aes(x=sum_CCAloading_2, y=Task_name)) + 
  geom_boxplot(outlier.size = 0.01)+theme(axis.text.y = element_blank())
box3 <- ggplot(df1, aes(x=sum_CCAloading_1, y=Task_name)) + 
  geom_boxplot(outlier.size = 0.01)+theme(axis.text.y = element_blank())
box4 <- ggplot(df1, aes(x=sum_CCAloading_1, y=Task_name)) + 
  geom_boxplot(outlier.size = 0.01)+theme(axis.text.y = element_blank())

allbox <- ((box1)|(box2)|(box3)|(box4))& theme(axis.text.x=element_text(size = fontsize,color = "black"),
                                                      axis.title.y = element_blank(),
                                                      axis.title.x = element_blank())
allbox

ggsave(
  "sumvariates_bytask_boxplot.tiff",
  allbox, units = "cm",
  width = 20,
  height = 7,
  dpi = 1000)


############################# Assumptions ######################################
models = c(model1, model2, model3, model4)
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
