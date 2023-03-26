# A function for calculating an optimal 'power' value for
# the inverse distance weighted (IDW) interpolation,
# based of {sf} object, using {spatstat}.
# Instructions from https://rpubs.com/Dr_Gurpreet/interpolation_idw_R

# Lesser idwpower - more 'grainy' interpolation, less the distance in which 
# nearby points affect new values.

# Input:
# points with values as an {sf} object, with single numerical (response) variable;
# vector map of extent as an {sf} object
# Output: numerical value of optimal 'power' for IDW interpolation

idw_optimalpower = function(data_sf, extent){
  library(sf)
  library(sp)
  library(spatstat)
  library(Metrics)
  
  # Creation of observation window.
  extent_sp <- as_Spatial(extent) # converting boundary to {sp} format
  obs_window <- owin(extent_sp@bbox[1,], extent_sp@bbox[2,])
  data_sp <- as(data_sf, "Spatial") # switch point data from {sf} to {sp}
  
  # Creation of point pattern object.
  ppp_data <- ppp(
    data_sp@coords[, 1],
    data_sp@coords[, 2],
    marks = data_sp@data[1],
    window = obs_window
  )
  
  # Calculation of mean squared error.
  # Mean squared error (MSE) is a measure of accuracy of the interpolation method.
  # Determination of optimal power for idw by cross validation.
  # Cross validating results to obtain lowest error.
  powers <- seq(from = 0.001, to = 2, by = 0.1)
  mse_result <- NULL
  for (power in powers) {
    CV_idw <- idw(ppp_data, power = power, at = "points")
    mse_result <- c(mse_result,
                    Metrics::mse(ppp_data$marks,CV_idw))
  }
  optimal_power <- powers[which.min(mse_result)]
  print(optimal_power)
}