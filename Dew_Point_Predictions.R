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
my_data <- read_excel("/Users/jackey/Desktop/Chinese_Temp_Predictions/Chinese_Weather_Differences.xlsx")
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
variogram_y_max <- 10  # for semivariance plot
variogram_x_max <- 500 # distance in km

# Loop through each week number (0 to 12)
for (week in 0:12) {
  # Filter data for the current week
  filtered_data <- my_data %>% 
    filter(week_number == as.character(week))
  
  # Convert to sf objects
  my_data_sf <- st_as_sf(filtered_data, coords = c("longitude", "latitude"), crs = 4326)
  china_grid <- st_as_sf(my_grid, coords = c("longitude", "latitude"), crs = 4326)
  
  # Reproject data to EPSG:3857 (meters)
  my_data_sf <- st_transform(my_data_sf, crs = 3857)
  china_grid <- st_transform(china_grid, crs = 3857)
  
  # Experimental variogram & fit
  v <- variogram(dew_point_diff ~ 1, data = my_data_sf)
  vinitial <- vgm(psill = 10, model = "Sph", range = 500000, nugget = 0)
  vfit <- fit.variogram(v, vinitial)
  
  # LOOCV
  cross_val_results <- list()
  for (i in 1:nrow(my_data_sf)) {
    training_data <- my_data_sf[-i, ]
    test_point    <- my_data_sf[i, ]
    
    v_cv        <- variogram(dew_point_diff ~ 1, data = training_data)
    vinitial_cv <- vgm(psill = 10, model = "Sph", range = 500000, nugget = 0)
    vfit_cv     <- fit.variogram(v_cv, vinitial_cv)
    
    kriging_model <- gstat(formula = dew_point_diff ~ 1, data = training_data, model = vfit_cv)
    prediction    <- predict(kriging_model, as(test_point, "Spatial"))
    
    cross_val_results[[i]] <- data.frame(
      longitude = st_coordinates(test_point)[1],
      latitude  = st_coordinates(test_point)[2],
      observed  = test_point$dew_point_diff,
      predicted = prediction@data$var1.pred,
      residuals = test_point$dew_point_diff - prediction@data$var1.pred
    )
  }
  
  # Combine CV results
  cv_results_df <- do.call(rbind, cross_val_results)
  
  # Optionally cap the observed/predicted in CV results to [-6, 10]
  cv_results_df <- cv_results_df %>%
    mutate(
      observed  = pmin(pmax(observed, -6), 10),
      predicted = pmin(pmax(predicted, -6), 10)
    )
  
  # Accuracy metrics
  rmse <- sqrt(mean((cv_results_df$predicted - cv_results_df$observed)^2))
  mae  <- mean(abs(cv_results_df$predicted - cv_results_df$observed))
  r_squared <- 1 - sum((cv_results_df$observed - cv_results_df$predicted)^2) /
    sum((cv_results_df$observed - mean(cv_results_df$observed))^2)
  
  # Store metrics
  all_metrics <- rbind(
    all_metrics,
    data.frame(Week = week, RMSE = rmse, MAE = mae, R_squared = r_squared)
  )
  
  # Theoretical variogram line
  dist_seq <- seq(0, max(v$dist), length.out = 100)
  theoretical_variogram <- variogramLine(vfit, dist_vector = dist_seq, covariance = FALSE)
  
  # ---- Variogram Plot ----
  variogram_plot <- ggplot() +
    geom_point(data = as.data.frame(v), aes(x = dist / 1000, y = gamma), color = "blue") +
    geom_line(data = as.data.frame(theoretical_variogram), aes(x = dist / 1000, y = gamma), color = "red") +
    annotate("text", 
             x = variogram_x_max * 3.8, 
             y = variogram_y_max * 0.1, 
             hjust = 1, 
             vjust = 0, 
             label = paste0(
               "RMSE: ", round(rmse, 2), "\n",
               "MAE: ", round(mae, 2), "\n",
               "R-squared: ", round(r_squared, 2)
             )) +
    labs(
      title = paste("Fitted Variogram - Week", week),
      x = "Distance (km)",
      y = "Semivariance"
    ) +
    theme_minimal() +
    coord_cartesian(xlim = c(0, 2500), ylim = c(0, variogram_y_max))
  
  # ---- Kriging Prediction ----
  kriging_model <- gstat(formula = dew_point_diff ~ 1, data = my_data_sf, model = vfit)
  kpred <- predict(kriging_model, as(china_grid, "Spatial"))
  kpred_sf <- st_as_sf(kpred) %>%
    # Cap predicted values at [-6, 10]
    mutate(
      var1.pred = pmin(pmax(var1.pred, -6), 10)
    )
  
  # ---- Prediction Plot ----
  prediction_plot <- ggplot() +
    # size=0.5 for smaller dots
    geom_sf(data = kpred_sf, aes(color = var1.pred), size = 0.5) +
    scale_color_gradientn(
      name = "Predicted",
      colors = c("blue", "white", "red"),
      values = scales::rescale(c(-6, 0, 10)),  # Make sure -6 -> blue, 0 -> white, 10 -> red
      limits = c(-6, 10),
      breaks = seq(-6, 10, by = 2),
      labels = scales::label_number()
    ) +
    labs(
      title = paste("Predicted Dewpoint Difference - Week", week)
    ) +
    theme_minimal() +
    theme(
      axis.title = element_blank(),
      axis.text  = element_blank(),
      axis.ticks = element_blank()
    )
  
  # ---- Actual vs. Predicted ----
  actual_vs_predicted_plot <- ggplot(cv_results_df, aes(x = observed, y = predicted)) +
    geom_point(color = "blue", alpha = 0.6) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    labs(
      title = "LOOCV Results",
      x = "Observed",
      y = "Predicted"
    ) +
    theme_minimal() +
    # Force axes from -6 to 10
    coord_cartesian(xlim = c(-6, 10), ylim = c(-6, 10))
  
  # ---- Variance Plot ----
  variance_plot <- ggplot() +
    # size=0.5 for smaller dots
    geom_sf(data = kpred_sf, aes(color = var1.var), size = 0.5) +
    scale_color_gradientn(
      name = "Variance",
      colors = c("white", "red"),
      values = scales::rescale(c(0, max(kpred_sf$var1.var, na.rm = TRUE))),
      limits = c(0, max(kpred_sf$var1.var, na.rm = TRUE)),
      breaks = pretty(c(0, max(kpred_sf$var1.var, na.rm = TRUE)), n = 5),
      labels = scales::label_number()
    ) +
    labs(
      title = paste("Variance Map - Week", week)
    ) +
    theme_minimal() +
    theme(
      axis.title = element_blank(),
      axis.text  = element_blank(),
      axis.ticks = element_blank()
    )
  
  # ---- Combine 4 plots in 2x2 ----
  combined_plot <- grid.arrange(
    variogram_plot,
    prediction_plot,
    actual_vs_predicted_plot,
    variance_plot,
    ncol = 2
  )
  
  # Save the combined plot
  ggsave(
    filename = paste0(output_folder, "Combined_Plot_Week_Dewpoint", week, ".png"),
    plot = combined_plot,
    width = 16,
    height = 12
  )
}

# Save metrics to a CSV file
write.csv(all_metrics, file = paste0(output_folder, "All_Metrics.csv"), row.names = FALSE)
print(all_metrics)
