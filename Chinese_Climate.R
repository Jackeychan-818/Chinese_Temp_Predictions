# Load necessary libraries
library(sp)
library(gstat)
library(sf)
library(ggplot2)
library(viridis)
library(readxl)
library(dplyr)
library(gridExtra)  # For combining multiple plots

# Define the folder path for saving plots (your Downloads folder)
output_folder <- "/Users/jackey/Downloads/"

# Load datasets
my_data <- read_excel("/Users/jackey/Desktop/Kriging/Corrected_City-Specific_Weekly_Temperature_Differences_with_Location.xlsx")
my_grid <- read_excel("/Users/jackey/Desktop/Kriging/china_grid2.xlsx")

# Initialize a data frame to collect metrics for all weeks
all_metrics <- data.frame(
  Week = integer(),
  RMSE = numeric(),
  MAE = numeric(),
  R_squared = numeric(),
  stringsAsFactors = FALSE
)

# Set fixed scales for all weeks
# For variogram plot (set manually or dynamically adjust based on full data range)
variogram_y_max <- 5
variogram_x_max <- 500  # Distance in km

# For actual vs predicted plot
predicted_y_max <- 5
predicted_x_max <- 5

# Loop through each week number (22 to 34)
for (week in 22:34) {
  # Filter data for the current week
  filtered_data <- my_data %>% filter(week_number == as.character(week))
  
  # Convert data to spatial data frames (sf objects)
  my_data_sf <- st_as_sf(filtered_data, coords = c("longitude", "latitude"), crs = 4326)
  china_grid <- st_as_sf(my_grid, coords = c("longitude", "latitude"), crs = 4326)
  
  # Reproject data to EPSG:3857 for distances in meters
  my_data_sf <- st_transform(my_data_sf, crs = 3857)
  china_grid <- st_transform(china_grid, crs = 3857)
  
  # Compute experimental variogram and fit a variogram model
  v <- variogram(temp_diff ~ 1, data = my_data_sf)
  vinitial <- vgm(psill = 10, model = "Sph", range = 500000, nugget = 1)  # Updated range in meters
  vfit <- fit.variogram(v, vinitial)
  
  # Perform leave-one-out cross-validation (LOOCV)
  cross_val_results <- list()
  for (i in 1:nrow(my_data_sf)) {
    training_data <- my_data_sf[-i, ]
    test_point <- my_data_sf[i, ]
    v <- variogram(temp_diff ~ 1, data = training_data)
    vinitial <- vgm(psill = 10, model = "Sph", range = 500000, nugget = 1)  # Updated range in meters
    vfit <- fit.variogram(v, vinitial)
    kriging_model <- gstat(formula = temp_diff ~ 1, data = training_data, model = vfit)
    prediction <- predict(kriging_model, as(test_point, "Spatial"))
    cross_val_results[[i]] <- data.frame(
      longitude = st_coordinates(test_point)[1],
      latitude = st_coordinates(test_point)[2],
      observed = test_point$temp_diff,
      predicted = prediction@data$var1.pred,
      residuals = test_point$temp_diff - prediction@data$var1.pred
    )
  }
  
  cv_results_df <- do.call(rbind, cross_val_results)
  
  # Calculate accuracy metrics
  rmse <- sqrt(mean((cv_results_df$predicted - cv_results_df$observed)^2))
  mae <- mean(abs(cv_results_df$predicted - cv_results_df$observed))
  r_squared <- 1 - sum((cv_results_df$observed - cv_results_df$predicted)^2) /
    sum((cv_results_df$observed - mean(cv_results_df$observed))^2)
  
  # Collect metrics for the current week
  all_metrics <- rbind(
    all_metrics,
    data.frame(Week = week, RMSE = rmse, MAE = mae, R_squared = r_squared)
  )
  
  # Compute theoretical variogram line
  dist_seq <- seq(0, max(v$dist), length.out = 100)
  theoretical_variogram <- variogramLine(vfit, dist_vector = dist_seq, covariance = FALSE)
  
  # Create variogram plot with fixed scales
  variogram_plot <- ggplot() +
    geom_point(data = as.data.frame(v), aes(x = dist / 1000, y = gamma), color = "blue") +  # Convert distance to km
    geom_line(data = as.data.frame(theoretical_variogram), aes(x = dist / 1000, y = gamma), color = "red") +
    annotate("text", x = variogram_x_max * 5, y = variogram_y_max * 0.1, hjust = 1, vjust = 0, 
             label = paste0(
               "RMSE: ", round(rmse, 2), "\n",
               "MAE: ", round(mae, 2), "\n",
               "R-squared: ", round(r_squared, 2)
             )) +
    labs(
      title = paste("Fitted Variogram - Week", week),
      x = "Distance (km)",  # Updated axis label
      y = "Semivariance"
    ) +
    theme_minimal() +
    coord_cartesian(xlim = c(0, 2500), ylim = c(0, variogram_y_max))
  
  # Kriging interpolation
  kriging_model <- gstat(formula = temp_diff ~ 1, data = my_data_sf, model = vfit)
  kpred <- predict(kriging_model, as(china_grid, "Spatial"))
  kpred_sf <- st_as_sf(kpred)
  
  # Create kriging prediction plot with fixed white at 0
  prediction_plot <- ggplot() +
    geom_sf(data = kpred_sf, aes(color = var1.pred)) +
    scale_color_gradientn(
      name = "Predicted",
      colors = c("blue", "white", "red"),
      values = scales::rescale(c(min(kpred_sf$var1.pred, na.rm = TRUE), 0, max(kpred_sf$var1.pred, na.rm = TRUE))),
      limits = range(kpred_sf$var1.pred, na.rm = TRUE),
      breaks = pretty(range(kpred_sf$var1.pred, na.rm = TRUE), n = 5),
      labels = scales::label_number()
    ) +
    labs(
      title = paste("Predicted Difference - Week", week),
    ) +
    theme_minimal()
  
  # Create actual vs. predicted plot with fixed scales
  actual_vs_predicted_plot <- ggplot(cv_results_df, aes(x = observed, y = predicted)) +
    geom_point(color = "blue", alpha = 0.6) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    labs(
      title = "Actual vs Predicted Values",
      x = "Observed",
      y = "Predicted"
    ) +
    theme_minimal() +
    coord_cartesian(xlim = c(-predicted_x_max, predicted_x_max), ylim = c(-predicted_y_max, predicted_y_max))
  
  # Create variance map
  variance_plot <- ggplot() +
    geom_sf(data = kpred_sf, aes(color = var1.var)) +
    scale_color_gradientn(
      name = "Variance",
      colors = c("white", "red"),
      values = scales::rescale(c(0, max(kpred_sf$var1.var, na.rm = TRUE))),
      limits = c(0, max(kpred_sf$var1.var, na.rm = TRUE)),  # Ensure the scale is consistent
      breaks = pretty(c(0, max(kpred_sf$var1.var, na.rm = TRUE)), n = 5),
      labels = scales::label_number()
    ) +
    labs(
      title = paste("Variance Map - Week", week)
    ) +
    theme_minimal()
  
  # Combine plots into 2x2 layout
  combined_plot <- grid.arrange(
    variogram_plot,
    prediction_plot,
    actual_vs_predicted_plot,
    variance_plot,
    ncol = 2
  )
  
  # Save the combined plot
  ggsave(
    filename = paste0(output_folder, "Combined_Plot_Week_", week, ".png"),
    plot = combined_plot,
    width = 16, height = 12
  )
}

# Save metrics to a CSV file
write.csv(all_metrics, file = paste0(output_folder, "All_Metrics.csv"), row.names = FALSE)
print(all_metrics)
