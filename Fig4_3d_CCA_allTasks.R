# script plots CCA summed variates for each task in 3d scatterplot.

# Load required libraries
library(dplyr)
library(ggplot2)
library(plotly)
library(patchwork)
library("plot3D")

# set working directory
setwd("C:\\Users\\bront\\Documents\\CanadaPostdoc\\MegaProject\\TaskBatteryAnalysis\\")

# Read the first CSV file of omnibus CCA
first_csv <- read.csv("scratch\\results\\cca\\allTasks\\allTask_variates.csv")

# Sort first_csv by 'Id_number' and 'Task_name'
first_csv_sorted <- first_csv %>%
  arrange(Id_number, Task_name) 

############################## 3d plot labels ##################################
# text labels

df2 <- first_csv %>%
  group_by(Task_name) %>%
  summarise_at(vars(c(sum_CCAloading_1, sum_CCAloading_2, sum_CCAloading_3, 
                      sum_CCAloading_4)), list(mean = mean))    

# quick look using plotly
# write.csv(df2, "scratch\\results\\cca\\allTasks\\allTask_variates_average.csv")
# 
# fig <- plot_ly(df2, x = ~sum_CCAloading_1_mean,
#                y = ~sum_CCAloading_2_mean,
#                z = ~sum_CCAloading_3_mean,  text = ~Task_name,
#                textfont = list(color = "black", size = 12),
#                marker = list(size = 6, color = "black"), opacity = 0.8 )
# fig <- fig %>% layout(scene = list(xaxis = list(title = 'Dimension 1', titlefont = list(size = 18)),
#                                    yaxis = list(title = 'Dimension 2', titlefont = list(size = 18)),
#                                    zaxis = list(title = 'Dimension 3', titlefont = list(size = 18))))
# fig <- fig%>% add_text(textposition = c("bottom center","top center",
#                                         "top center",
#                                         "bottom center",
#                                         "bottom center","bottom center",
#                                         "bottom center",
#                                         "bottom center","bottom center",
#                                         "bottom center","bottom center",
#                                         "bottom center","top right",
#                                         "bottom center"))
# fig

##########################
# save proper one out
setwd("C:\\Users\\bront\\Documents\\CanadaPostdoc\\MegaProject\\TaskBatteryAnalysis\\scratch\\results\\cca")

png("scatterplot3d_plot.png",  width     = 2,
    height    = 2,
    units     = "in",
    res       = 1200,
    pointsize = 4)  # Adjust width and height as needed

with(df2, text3D(sum_CCAloading_1_mean,sum_CCAloading_2_mean, sum_CCAloading_3_mean, 
                 labels = Task_name, zlim = c(-2.5,2),
                 theta = 30, phi = 10,
                 xlab = "", ylab = "", zlab = "", 
                 main = "", cex = 1.4, 
                 bty = "g",  d = 2,
                 adj = 0.5))

dev.off()

