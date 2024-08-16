# Load required libraries
library(dplyr)
library(ggplot2)
library(plotly)
library("plot3D")

############################### Read in data ###################################

# setwd("C:\\Users\\bront\\Documents\\CanadaPostdoc\\MegaProject\\TaskBatteryAnalysis\\")
file_path <- file.path(getwd(),"scratch/results/pca/allTasks/allTasksNoRotation5_20112023_11-40-53/csvdata/projected_pca_scores.csv")
# Read the first CSV file of omnibus PCA
first_csv <- read.csv(file_path, na.strings=c(""," ","NA", "nan"))

# Sort first_csv by 'Id_number' and 'Task_name'
first_csv_sorted <- first_csv %>%
  arrange(Id_number, Task_name) 

############################## 3d plot labels ##################################
df2 <- first_csv %>%
  group_by(Task_name) %>%
  summarise_at(vars(c(PCA_0, PCA_1, PCA_2, PCA_3, PCA_4)), list(mean = mean))

setwd(file.path(getwd(), "scratch\\results\\pca"))

write.csv(df2, "pca_taskaverages.csv", row.names = FALSE)


# jitter labels to prevent text label overlap
df2 <- df2 %>%
  mutate(Task_name = case_when(
    Task_name == 'Documentary' ~ 'Doc',
    Task_name == 'SciFi' ~ '          SciFi',
    Task_name == 'FingerTap' ~ '  FingerTap',
    Task_name == 'GoNoGo' ~ '  GoNoGo\n',
    Task_name == '1B' ~ '  1B\n',
    TRUE ~ Task_name
  ))

# save plot
png("pca_scatterplot3d_plot.png",  width     = 6.2,
    height    = 8,
    units     = "cm",
    res       = 1200,
    pointsize = 6.7)

with(df2, text3D(PCA_0_mean,PCA_1_mean, PCA_2_mean, 
                 labels = Task_name,
                 theta = 70, phi = 10, xlim = c(-1.5, 1.5), ylim = c(-1.5, 1.5), zlim = c(-1.5, 1.5),
                 xlab = "", ylab = "", zlab = "", 
                 main = "", cex = 1, 
                 bty = "g", d = 2, 
                 adj = 0.5))


dev.off()

