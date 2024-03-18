"
1. 3d plot of:
  1. real tasks in gradient space
  2. predicted tasks in gradient space
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
setwd("C:\\Users\\bront\\Documents\\CanadaPostdoc\\MegaProject\\TaskBatteryAnalysis\\scratch\\results\\cca\\allTasks")

#read in csv file
df_predicted <- read.csv("reptasks_cca_predictgrads.csv", na.strings=c(""," ","NA", "nan"))
df_actual <- read.csv("reptasks_cca_realgrads.csv", na.strings=c(""," ","NA", "nan"))

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
setwd("C:\\Users\\bront\\Documents\\CanadaPostdoc\\MegaProject\\TaskBatteryAnalysis\\scratch\\results\\cca")

png("scatterplot3d_predict.png",  width     = 2,
    height    = 2,
    units     = "in",
    res       = 1200,
    pointsize = 4)  # Adjust width and height as needed

with(stacked_data, text3D(`Gradient 1`,`Gradient 2`, `Gradient 3`, 
                          col=c(rep("red", 4), rep("black", 4)),
                          xlim = c(-1.5,2.3),
                 labels = Task_name,
                 theta = 30, phi = 10,
                 xlab = "", ylab = "", zlab = "", 
                 main = "", cex = 1.4, 
                 bty = "g",  d = 2,
                 colkey = TRUE,
                 adj = 0.5))

dev.off()


################################################################################
# take a look two dimensions at a time for checking
scatter_plot <- function(data, x, y, label) {
  
  ggplot(data, aes({{x}}, {{y}}, label = {{label}}, color = Task_name)) +
    geom_point()+geom_text_repel()+
    theme_light()+ 
    geom_hline(yintercept = 0)+ geom_vline(xintercept = 0)
}


grad12 <- scatter_plot(stacked_data, `Gradient 1`, 
                      `Gradient 2`, suffix)+labs(x = "Gradient 1", y = "Gradient 2")
grad12

grad13 <- scatter_plot(stacked_data, `Gradient 1`, 
                       `Gradient 3`, suffix)+labs(x = "Gradient 1", y = "Gradient 3")
grad13

grad23 <- scatter_plot(stacked_data, `Gradient 2`, 
                       `Gradient 3`, suffix)+labs(x = "Gradient 2", y = "Gradient 3")
grad23

all <- grad12+ grad13+grad23
all

fig <- plot_ly(stacked_data, x = ~`Gradient 1`, y = ~`Gradient 2`, z = ~`Gradient 3`,  
               color = ~Task_name, symbol = ~suffix, symbols = c("circle", "circle-open"),
               marker = list(size = 10), opacity = 0.4)
fig <- fig %>% add_markers()
fig <- fig %>% layout(showlegend = TRUE,
                      scene = list(xaxis = list(title = 'Gradient 1',
                                                range=c(-3,3),
                                                
                                                
                                                tickfont=list(size=12)),
                                   yaxis = list(title = 'Gradient 2',
                                                range=c(-3,3),
                                                
                                                tickfont=list(size=12)),
                                   zaxis = list(title = 'Gradient 3', 
                                                range=c(-3,3),
                                               
                            
                                                tickfont=list(size=12))))
fig


