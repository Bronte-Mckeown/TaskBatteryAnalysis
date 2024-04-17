# plots mean distances as bar graphs for supplementary figure

# Load required libraries
library(ggplot2)
library(patchwork)
library(plyr)

# set working directory
setwd("C:\\Users\\bront\\Documents\\CanadaPostdoc\\MegaProject\\TaskBatteryAnalysis\\scratch\\results\\cca\\allTasks")

# Pattern to match files
file_pattern <- "^(.*)_dists_for_bargraphs_10tasks\\.csv$"

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
# Plotting using ggplot2 and save in a list
for (task_name in names(results)) {
  result <- results[[task_name]]
  df <- dfs[[task_name]]
  
  p <- ggplot(result, aes(y = reorder(task, -Mean), x = Mean)) +
    geom_bar(stat = "identity", position = "dodge", fill = "grey") +
    geom_point(data = df, aes(x = distance, y = task), color = "blue", alpha = 0.1,
               size = 0.0001)+
    geom_errorbar(
      aes(xmin = Mean - SE, xmax = Mean + SE),
      width = 0.3,size = 0.2,
      position = position_dodge(0.9)) +
    labs(title = paste(task_name),  y = "Mean Distance") + theme_light() +
    theme(axis.text.y=element_text(size = fontsize, color = "black"),
          axis.text.x=element_text(size = fontsize, color = "black", margin = margin(r=0)),
          title = element_text(size = fontsize, color = "black", face = "bold"),
          axis.title.x=element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.x = element_blank(),
          axis.ticks.y = element_blank(),
          axis.ticks.margin = unit(0, 'cm'),
          plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm"),
          plot.title.position = "plot")+
    scale_x_continuous(limits = c(0,11))
  
  plot_list[[task_name]] <- p
}

allplots <- (plot_list[[1]] + plot_list[[2]]) / (plot_list[[3]] + plot_list[[4]]) & theme(plot.margin = margin(0.01, 0.01, 0.01, 0.01, "cm"))
allplots

# Save the plot
ggsave(
  "distances_bargraphs_10tasks_rawpoints.tiff",
  allplots, units = "cm",
  width = 4.38,
  height = 3.9,
  dpi = 1000)

