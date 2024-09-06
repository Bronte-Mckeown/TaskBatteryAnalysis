devtools::install_github("ricardo-bion/ggradar", 
                         dependencies = TRUE)

library(ggplot2)
library(ggradar)
suppressPackageStartupMessages(library(dplyr))
library(scales)
library(tidyr)


########################## Read in data ########################################
#set current working directory
setwd("scratch\\results\\cca\\allTasks\\projectedBrainMaps")

#read in csv file
d <- read.csv("ccaDim_yeo_avgs.csv", na.strings=c(""," ","NA", "nan"))


d$yeo_network <- factor(d$yeo_network, levels = c("VisCent",
                                                  "VisPeri",
                                                  "SM-a",
                                                  "SM-b",
                                                  "DAN-a",
                                                  "DAN-b",
                                                  "VAN-a",
                                                  "VAN-b",
                                                  "LMB-b",
                                                  "LMB-a",
                                                  "FPN-a",
                                                  "FPN-b",
                                                  "FPN-c",
                                                  "DMN-a",
                                                  "DMN-b",
                                                  "DMN-c",
                                                  "TempPar"),
                   ordered = FALSE)


d$ccaDim <- sub("cca1", "CCA 1", d$ccaDim)
d$ccaDim <- sub("cca2", "CCA 2", d$ccaDim)
d$ccaDim <- sub("cca3", "CCA 3", d$ccaDim)
d$ccaDim <- sub("cca4", "CCA 4", d$ccaDim)


d$ccaDim <- factor(d$ccaDim , levels = c('CCA 1', 'CCA 2','CCA 3',
                                              'CCA 4'))
levels(d$ccaDim )

d <- d[, c(1:4)]

########################## Radar plots ########################################

# Pivot the dataframe to wide format
df <- pivot_wider(d, names_from = yeo_network, values_from = mean_value)

subset_df <- subset(df, (ccaDim %in% c('CCA 1', 'CCA 2','CCA 3', 'CCA 4')))
subset_df

radar <- ggradar(subset_df, grid.min = -20, grid.max = 20,grid.mid = 0,
                    values.radar = c("Low", "0", "High"),
                    legend.position = "bottom", fill = FALSE,base.size = 10,
                    group.line.width = 0.8,
                    group.point.size = 5,
                    grid.line.width = 0.2)
radar

# save
setwd("scratch\\results\\cca\\allTasks\\projectedBrainMaps")
ggsave(
  "ccaDims_radar.tiff",
  radar, units = "cm",
  width = 20,
  height = 20,
  dpi = 1000, 
)

