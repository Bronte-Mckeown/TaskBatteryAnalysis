# script plots distribution of distances as histograms from CCA prediction
# (14 tasks)

# Load required libraries
library(ggplot2)
library(patchwork)
library(plyr)

# set working directory
setwd("C:\\Users\\bront\\Documents\\CanadaPostdoc\\MegaProject\\TaskBatteryAnalysis\\scratch\\results\\cca\\allTasks")

# Pattern to match files
file_pattern <- "^(.*)_dists_for_bargraphs\\.csv$"

# Create a list to store dataframes and plots
dfs <- list()
results <- list()
plot_list <- list()

# Read CSV files into dataframes using pattern matching
matching_files <- list.files(pattern = file_pattern)
for (filename in matching_files) {
  task_name <- gsub(file_pattern, "\\1", filename)
  dfs[[task_name]] <- read.csv(filename)
}

# calculate means and SE
for (task_name in names(dfs)) {
  df<- dfs[[task_name]]
  result <- df %>%
    group_by(task) %>%
    dplyr::summarise(
      Mean = mean(distance),
      SE = sd(distance) / sqrt(n())
    )
  results[[task_name]] <- result 
}

fontsize = 5

# histograms
for (task_name in names(results)) {
  result <- results[[task_name]]
  df <- dfs[[task_name]]
  
  print (result$Mean)
  
  # Interleaved histograms
  p <- ggplot(df, aes(x=distance, color=task)) +
    geom_histogram(aes(fill = task), position="identity", alpha=0.1)+
    geom_vline(data=result, aes(xintercept=Mean, color=task),
               linetype="dashed")+
    theme(
      legend.title=element_blank())+
    labs(title = paste(task_name),  y = "Count", x = 'Distance')
  p
  
  plot_list[[task_name]] <- p
}

allplots <- ((plot_list[[1]] + plot_list[[2]]) / (plot_list[[3]] + plot_list[[4]])) +
  plot_layout(guides = "collect")& 
  theme(legend.position = 'bottom', plot.margin = margin(0.01, 0.01, 0.01, 0.01, "cm"))
allplots

ggsave(
  "distances_histograms_rawpoints.tiff",
  allplots, units = "cm",
  width = 4.38,
  height = 3.9,
  dpi = 1000)