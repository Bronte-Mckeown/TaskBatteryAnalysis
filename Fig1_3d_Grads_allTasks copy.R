####################### Load libraries #########################################
library(rgl)
library(patchwork) # putting plots together in one figure
library(ggplot2)
library(dplyr) # data manipulation
library(tidyr)
library(ggrepel)
library(plotly)

library(plotrix)
library("plot3D")

########################## Read in data ########################################
#set current working directory
# setwd("C:\\Users\\bront\\Documents\\CanadaPostdoc\\MegaProject\\TaskBatteryAnalysis\\scratch\\data\\source")
file_path <- file.path(getwd(), "scratch/data/gradscores_spearman_combinedmask_cortical_subcortical.csv")
print(file_path)
#read in csv file
df1 <- read.csv(file_path, na.strings=c(""," ","NA", "nan"))

# Define the mapping between task names
task_name_mapping <- c(
  'you'           = 'You',
  'friend'        = 'Friend',
  'twoBackFaces'  = '2B-Face',
  'twoBackScenes' = '2B-Scene',
  'easyMath'      = 'EasyMath',
  'hardMath'      = 'HardMath',
  'reading'       = 'Read\n',
  'memory'        = 'Memory',
  'oneBack'       = '1B',
  'zeroBack'      = '0B',
  'gonogo'        = 'GoNoGo',
  'fingerTapping' = 'FingerTap',
  'movieIncept'   = 'SciFi',
  'movieBridge'   = 'Doc'
)

# Apply the mapping
df1 <- df1 %>%
  mutate(Task_name = task_name_mapping[Task_name])

# Define the asEuclidean function
asEuclidean <- function(homogeneous_coords) {
  # Divide by the last row to convert from homogeneous to Euclidean coordinates
  euclidean_coords <- homogeneous_coords[, 1:3] / homogeneous_coords[, 4]
  return(euclidean_coords)
}

# Define the thigmophobe.text3d function
thigmophobe.text3d <- function(x, y = NULL, z = NULL, texts, ...) {
  xyz <- xyz.coords(x, y, z)
  # Define a function to sanitize text labels
  sanitize_text <- function(text) {
    # Remove or replace unsupported characters
    gsub("[^[:alnum:] ]", "", text)
  }

# Apply the sanitize_text function to the texts vector
texts <- sapply(texts, sanitize_text)
  # Get the coordinates as columns in a matrix in
  # homogeneous coordinates
  pts3d <- rbind(xyz$x, xyz$y, xyz$z, 1)

  # Apply the viewing transformations and convert 
  # back to Euclidean
  pts2d <- asEuclidean(t(par3d("projMatrix") %*% 
                         par3d("modelMatrix") %*% 
                         pts3d))

  # Find directions so that the projections don't overlap
  pos <- plotrix::thigmophobe(pts2d)

  # Set adjustments for the 4 possible directions
  adjs <- matrix(c(0.5, 1.2,   
                   1.2, 0.5,  
                   0.5, -0.2,  
                  -0.2, 0.5), 
                 4, 2, byrow = TRUE)

  # Plot labels one at a time in appropriate directions.
  for (i in seq_along(xyz$x)) 
    text3d(x = pts3d[1, i], y = pts3d[2, i], z = pts3d[3, i], texts = texts[i], 
           adj = adjs[pos[i],])
}

# Change directory to results
setwd(file.path(getwd(),"scratch\\results"))

# Open a new 3D plotting device
open3d()

# Plot the 3D points
plot3d(df1$gradient1_cortical_subcortical, df1$gradient2_cortical_subcortical, df1$gradient3_cortical_subcortical, 
       xlim = c(-.4, .4), ylim = c(-.4, .4), zlim = c(-.4, .4),
       xlab = "", ylab = "", zlab = "", 
       main = "", size = 1, 
       type = "s", col = "blue")
       
par3d(userMatrix = rotationMatrix(pi/2, 0, 1, 0) )

thigmophobe.text3d(df1$gradient1_cortical_subcortical, df1$gradient2_cortical_subcortical, df1$gradient3_cortical_subcortical, 
                             texts = df1$Task_name,
                             theta = 70, phi = 10,
                             xlab = "", ylab = "", zlab = "", 
                             xlim = c(-.4, .4), ylim = c(-.4, .4), zlim = c(-.4, .4),
                             main = "", cex = 1, 
                             bty = "g", d = 2)

# Remove default axes
axes3d(edges = "bbox")


grid3d(c("x", "y", "z"))

rgl.snapshot("pca_scatterplot3d_plot.png", fmt = "png")

