# Load required libraries
library(dplyr)
library(ggplot2)
library(plotly)
library(patchwork)

############################### Read in data ###################################
# set directory
setwd("C:\\Users\\bront\\Documents\\CanadaPostdoc\\MegaProject\\TaskBatteryAnalysis\\")

# Read the first CSV file of omnibus PCA
first_csv <- read.csv("scratch/results/pca/allTasks/allTasksNoRotation5_20112023_11-40-53/csvdata/projected_pca_scores.csv")

# Sort first_csv by 'Id_number' and 'Task_name'
first_csv_sorted <- first_csv %>%
  arrange(Id_number, Task_name) 

# Use list.files to select multiple data paths matching a pattern to read in held out projected scores
data_paths <- list.files("scratch/results/pca/holdOut", pattern = "projected_pca_scores.csv", recursive = TRUE, full.names = TRUE)

############################# each separately ##################################
# number of tasks
num_tasks = 14

# Desired confidence level
confidence_level <- 0.95

# Bonferroni-adjusted confidence level
adjusted_confidence_level <- 1 - (1 - confidence_level) / num_tasks
adjusted_confidence_level

# Create a list to store results
correlation_list <- list()

# Extract unique tasks from all data paths
all_tasks <- unique(sapply(data_paths, function(path) strsplit(strsplit(path, "/")[[1]][5], "_")[[1]][1]))

# Create a list to store scatterplots for checking linearity assumption
scatterplots_list <- list()

# Loop over data paths and read in held-out CSV files
for (path in data_paths) {
  # identify task string
  task <- strsplit(strsplit(path, "/")[[1]][5], "_")[[1]][1]
  
  # Read in the CSV file
  csv_data <- read.csv(path)
  
  # Sort csv_data by 'Id_number' and 'Task_name'
  csv_data_sorted <- csv_data %>%
    arrange(Id_number, Task_name)
  
  # Filter data for the current task in both dataframes
  first_task_data <- first_csv_sorted[first_csv_sorted$Task_name == task, ]
  csv_task_data <- csv_data_sorted[csv_data_sorted$Task_name == task, ]
  
  # Group by Id_number and calculate averages for PCA columns
  first_task_data_avg <- first_task_data %>%
    group_by(Id_number) %>%
    summarise(across(starts_with("PCA_"), ~ mean(., na.rm = TRUE)))
  
  csv_task_data_avg <- csv_task_data %>%
    group_by(Id_number) %>%
    summarise(across(starts_with("PCA_"), ~ mean(., na.rm = TRUE)))
  
  # Select PCA column names
  pca_cols <- grep("^PCA_", names(first_task_data_avg), value = TRUE)
  
  # Loop over PCA column names
  for (column in pca_cols) {
    
    # Create scatterplot
    scatterplot <- ggplot(data = bind_cols(first_task_data_avg, csv_task_data_avg), aes(x = first_task_data_avg[[column]], y = csv_task_data_avg[[column]])) +
      geom_point() +
      geom_smooth(method = "lm", se = FALSE) +
      labs(x = paste0("PCA_", gsub("PCA_", "", column)), y = paste0("PCA_", gsub("PCA_", "", column))) +
      ggtitle(paste("Scatterplot for", column, "in Task", task))
    
    # Store scatterplot in the list to check after
    scatterplots_list[[paste("Task", task, "_", column, sep = "")]] <- scatterplot
    
    # Remove outliers from each dataframe separately (done to check)
    # first_task_data_avg[[column]][abs(scale(first_task_data_avg[[column]], center = TRUE, scale = TRUE)) > 2.5] <- NA
    # csv_task_data_avg[[column]][abs(scale(csv_task_data_avg[[column]], center = TRUE, scale = TRUE)) > 2.5] <- NA
    
    # Correlate each column with the respective column in the merged data
    cor_test_result <- cor.test(first_task_data_avg[[column]], csv_task_data_avg[[column]], conf.level = adjusted_confidence_level,
                                alternative = "two.sided",
                                na.action = "na.omit")
    
    # Take the absolute value of the correlation and CIs for plotting
    abs_correlation <- abs(cor_test_result$estimate)
    abs_lower_ci <- abs(cor_test_result$conf.int[1])
    abs_upper_ci <- abs(cor_test_result$conf.int[2])
    
    # Create a DataFrame for the current correlation if it doesn't exist
    if (is.null(correlation_list[[column]])) {
      correlation_list[[column]] <- data.frame(Task = all_tasks, Correlation = rep(NA, length(all_tasks)),
                                               LowerCI = rep(NA, length(all_tasks)), UpperCI = rep(NA, length(all_tasks)))
    }
    
    # Find the index of the current task in the dataframe
    task_index <- which(all_tasks == task)
    
    # Fill in the values in the dataframe
    correlation_list[[column]]$Correlation[task_index] <- abs_correlation
    correlation_list[[column]]$LowerCI[task_index] <- abs_lower_ci
    correlation_list[[column]]$UpperCI[task_index] <- abs_upper_ci
  }
}

# Concatenate correlations for each component into one dataframe (horizontal stack)
result_df <- bind_cols(correlation_list)

#save as csv
setwd("C:\\Users\\bront\\Documents\\CanadaPostdoc\\MegaProject\\TaskBatteryAnalysis\\scratch\\results\\pca")
write.csv(result_df, "heldoutpca_sim.csv", row.names = FALSE)

# Access each dataframe from the list by PCA correlation
# For example, to access the dataframe for PCA correlation "PCA_1":
# pca1_dataframe <- result_df[, c("Task", "Correlation", "LowerCI", "UpperCI")]

# Now 'result_df' contains the concatenated correlations for each component

# Loop over PCA components 1 to 5
for (i in 0:4) {
  # Calculate the minimum correlation for the current PCA component
  min_correlation <- round(min(correlation_list[[paste0("PCA_", i)]]$Correlation), digits = 3)
  
  # Calculate the maximum correlation for the current PCA component
  max_correlation <- round(max(correlation_list[[paste0("PCA_", i)]]$Correlation), digits = 3)
  
  # Print the results for the current PCA component
  cat("PCA Component", i, ": Min =", min_correlation, ", Max =", max_correlation, "\n")
}


################################ Greyscale bar #################################
fontsize = 6
# Define task order
task_order <- c("EasyMath", "HardMath", "FingerTap", "GoNoGo",
                "Friend", "You", "Documentary", "SciFi",
                "1B", "0B", "Read", "Memory",
                "2B-Face", "2B-Scene")

# Function for making plots
myplot2 <- function(data, title) {
  ggplot(data, aes(x = Correlation, y = factor(Task, levels = task_order, ordered = FALSE))) +
 coord_cartesian(xlim = c(.8, 1))+
    theme_light() +
    geom_bar(stat="identity",width = 2/.pt, position="dodge", color = "black" ,fill = "grey", size = 0.5/.pt) +
    theme(axis.text.y = element_blank(),
          axis.text.x=element_text(size = fontsize,color = "black"),
          axis.title.x=element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.x = element_blank(),
          axis.ticks.y = element_blank(),
        plot.margin = margin(0, 0, 0, 0, "cm"))   +
    # Add error bars
    geom_errorbar(position=position_dodge(.6/.pt),width=0.25/.pt,size = 0.5/.pt, 
                  aes(xmax = UpperCI, xmin = LowerCI), alpha = 1)+
    scale_x_continuous(breaks = seq(.8, 1, by = 0.1),labels = c(".8",".9","1"))
}

# Loop over the correlation_list to create plots
for (i in seq_along(correlation_list)) {
  pca_corr <- correlation_list[[i]]
  title <- paste("PCA Correlation ", i, sep = "")
  plot_name <- paste("pca_plot_", i, sep = "")
  assign(plot_name, myplot2(pca_corr, title))
}

pca_plot_1 <- pca_plot_1 + theme(axis.text.y=element_text(size = fontsize,color = "black"))

# put together
all_plots <- ((pca_plot_1)|(pca_plot_2)|(pca_plot_3)|(pca_plot_4)|(pca_plot_5))&
  geom_vline(xintercept = 0, size = 0.2/.pt)
all_plots

setwd("C:/Users/bront/Documents/CanadaPostdoc/MegaProject/TaskBatteryAnalysis/scratch/results/pca")

ggsave(
  "Holdout_similarity_barchart.tiff",
  all_plots, units = "cm",
  width = 8.7,
  height = 4,
  dpi = 1000)


