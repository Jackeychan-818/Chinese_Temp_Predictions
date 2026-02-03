###############################################################################
#                            R SCRIPT TO PLOT CHINESE CITIES
###############################################################################
# Description:
#  - Reads an Excel file containing city data for China (lat/lon, city name, etc.)
#  - Plots the locations of these cities on a map of China.
#
# Before Running:
#  1. Install required libraries if you haven't: 
#       install.packages(c("readxl", "ggplot2", "dplyr", "maps"))
#  2. Adjust file paths, column names, and any aesthetic settings as needed.
###############################################################################

# ------------------------ LOAD LIBRARIES ------------------------ #
library(readxl)    # For reading Excel files
library(ggplot2)   # For plotting in R
library(dplyr)     # For data manipulation
library(maps)      # For map data (basic world maps)


# ------------------------ READ YOUR DATA ------------------------ #
# Adjust the path to your Excel file if needed.
my_data <- read_excel("/Users/jackey/Desktop/Chinese_Temp_Predictions/Chinese_Weather_Differences.xlsx")

# Check the column names to confirm how latitude and longitude are labeled.
print(names(my_data))

# Suppose the columns for lat/long are "lat" and "lon". Rename them to "Latitude" & "Longitude".
# If they're already named something else, adjust as needed.
names(my_data)[names(my_data) == "lat"] <- "Latitude"
names(my_data)[names(my_data) == "lon"] <- "Longitude"

# Optional: If your Excel file has a city name column named differently, rename it to "City":
# names(my_data)[names(my_data) == "oldCityName"] <- "City"

# Let's see how data looks after renaming:
head(my_data)


# ------------------------ GET MAP DATA ------------------------ #
# The 'maps' package includes a world map. We subset it to get the China polygon.
world_map <- map_data("world")
china_map <- subset(world_map, region == "China")


# ------------------------ PLOT ------------------------ #
# 1) Draw the polygon for China
# 2) Overlay the points (cities) from your Excel data
# 3) (Optional) add labels for city names
ggplot() +
  # Draw the China map polygon
  geom_polygon(
    data = china_map,
    aes(x = long, y = lat, group = group),
    fill = "gray90",   # Fill color for China's land
    color = "black"    # Outline color
  ) +
  
  # Plot your city locations
  geom_point(
    data = my_data,
    aes(x = longitude, y = latitude),
    color = "blue",
    size = 2
  ) +
  
  # (Optional) Add city labels if you have a "City" column
  # geom_text(
  #   data = my_data,
  #   aes(x = Longitude, y = Latitude, label = City),
  #   vjust = -1, 
  #   color = "red",
  #   size = 3
  # ) +
  
  # Customize labels and theme
  labs(
    title = "Chinese Cities from Excel Sheet",
    x = "longitude",
    y = "latitude"
  ) +
  
  # Ensure a proper aspect ratio for maps
  coord_quickmap() +
  
  # Apply a minimal theme
  theme_minimal()

###############################################################################
#                           END OF R SCRIPT
###############################################################################
