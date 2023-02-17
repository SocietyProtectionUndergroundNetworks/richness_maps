library(data.table)
library(spatialRF)
library(sf)
library(geoR)
library(tidyverse)
library(geosphere)
library(patchwork)

# Summary of k-fold CV R2 values per model
df <- fread('/Users/johanvandenhoogen/ETH/Projects/collembola/model_details_Biomass_dry_mcg_m2_mean_final.csv') %>% 
  mutate(classProperty = 'Biomass_dry') %>% 
  rbind(., fread('/Users/johanvandenhoogen/ETH/Projects/collembola/model_details_Biomass_fresh_mcg_m2_mean_final.csv') %>% 
          mutate(classProperty = 'Biomass_fresh')) %>% 
  rbind(., fread('/Users/johanvandenhoogen/ETH/Projects/collembola/model_details_CommunityMetabolism_Jh_m2_mean_final.csv') %>% 
          mutate(classProperty = 'CommunityMetabolism')) %>% 
  rbind(fread('/Users/johanvandenhoogen/ETH/Projects/collembola/model_details_Density_m2_mean_final.csv') %>% 
          mutate(classProperty = 'Density')) %>% 
  rbind(., fread('/Users/johanvandenhoogen/ETH/Projects/collembola/model_details_Rarefied_final.csv') %>% 
          mutate(classProperty = 'Rarefied'))

df %>% group_by(classProperty) %>% 
  summarise_each(mean)

# Full dataset
df <- fread('/Users/johanvandenhoogen/ETH/Projects/collembola/data/20210527_collembola_sampled.csv')

# Variables included in models
covariateList = c(
  'CGIAR_Aridity_Index',
  'CHELSA_BIO_Annual_Mean_Temperature',
  'CHELSA_BIO_Annual_Precipitation',
  'CHELSA_BIO_Precipitation_Seasonality',
  'CHELSA_BIO_Precipitation_of_Driest_Quarter',
  'CHELSA_BIO_Temperature_Annual_Range',
  'CHELSA_BIO_Temperature_Seasonality',
  'ConsensusLandCoverClass_Barren',
  'ConsensusLandCoverClass_Cultivated_and_Managed_Vegetation',
  'ConsensusLandCoverClass_Herbaceous_Vegetation',
  'ConsensusLandCoverClass_Shrubs',
  'EarthEnvTopoMed_Elevation',
  'EarthEnvTopoMed_Roughness',
  'GPWv4_Population_Density',
  'GlobBiomass_AboveGroundBiomass',
  'HansenEtAl_TreeCover_Year2010',
  'MODIS_NPP',
  'SG_Bulk_density_015cm',
  'SG_Clay_Content_015cm',
  'SG_Coarse_fragments_015cm',
  'SG_SOC_Content_015cm',
  'SG_Sand_Content_015cm',
  'SG_Soil_pH_H2O_015cm')

# List of response variables to model
varList = c(
  'Biomass_dry_mcg_m2_mean',
  'Biomass_fresh_mcg_m2_mean',
  'CommunityMetabolism_Jh_m2_mean',
  # 'CommunityMetabolism_Jh_m2_mean_soilT',
  'Density_m2_mean',
  'Rarefied')

# Range of distances to test Moran's I 
distance.thresholds <- c(10, 50, 100, 150, 200, 300, 400, 500, 750, 1000)

# For each variable, train a non-spatial RF and retrieve Moran's I
for (var in varList){
  df <- fread(paste0('/Users/johanvandenhoogen/ETH/Projects/collembola/data/training_data/',var,'/',var,'CV_Fold_Collection.csv'))
  
  training_data <- df %>% select(all_of(covariateList), all_of(var), Pixel_Lat, Pixel_Long) 
  xy <- training_data[, c("Pixel_Long", "Pixel_Lat")]
  
  model.non.spatial <- spatialRF::rf(
    data = training_data,
    dependent.variable.name = var,
    predictor.variable.names = covariateList,
    distance.matrix = distm(xy)/1000,
    distance.thresholds = distance.thresholds,
    xy = xy, 
    seed = 123,
    verbose = FALSE
  )
  
  moransI <- model.non.spatial$residuals$autocorrelation$per.distance %>% 
    mutate(classProperty = var)
  
  assign(paste0(var, '_moran'), moransI)
}

# Create composite plot
plot_moran <- Biomass_dry_mcg_m2_mean_moran %>% 
  rbind(., Biomass_fresh_mcg_m2_mean_moran) %>% 
  rbind(., CommunityMetabolism_Jh_m2_mean_moran) %>% 
  # rbind(., CommunityMetabolism_Jh_m2_mean_soilT_moran) %>% 
  rbind(., Density_m2_mean_moran) %>% 
  rbind(., Rarefied_moran) %>% 
  mutate(p.value.binary = case_when(p.value >= 0.05 ~ "p >= 0.05",
                                    p.value < 0.05 ~ "p < 0.05")) %>% 
  ggplot(aes(x = distance.threshold, y = moran.i)) +
  geom_line() +
  geom_point(aes(color = p.value.binary)) +
  scale_color_manual(
    breaks = c("p < 0.05", "p >= 0.05"),
    values = c("red", "black")
  ) +
  facet_wrap(vars(classProperty), nrow = 1) +
  ylim(c(-0.05, 0.1)) +
  geom_hline(yintercept = 0, linetype = 'dashed') +
  theme_bw() +
  xlab('Distance (km)') +
  ylab("Moran's I") +
  theme(legend.title=element_blank())


results <- list()
# For each variable, train a non-spatial RF and retrieve Moran's I
for (var in varList){
  df <- fread(paste0('/Users/johanvandenhoogen/ETH/Projects/collembola/output/',var,'_pred_obs.csv')) %>% 
    mutate(predicted := !!as.name(paste0(var,'_EnsembleMean'))) %>% 
    mutate(observed := exp(!!as.name(var))) %>% 
    mutate(resid = abs(predicted-observed)) %>% 
    select(Pixel_Long, Pixel_Lat, resid) %>% 
    st_as_sf(coords = c('Pixel_Long', 'Pixel_Lat')) %>% 
    st_set_crs("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0") %>% 
    st_transform("+proj=merc +lon_0=0 +lat_ts=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")
  datM_df <- data.frame(st_coordinates(df)[,1],st_coordinates(df)[,2],df$resid)
  data_geoR = as.geodata(datM_df, coords.col = 1:2, data.col = 3)
  
  empVario <- variog(data_geoR, uvec = distance.thresholds*1000)
  envelope <- variog.mc.env(data_geoR, obj.variog = empVario)
  # plot(empVario, envelope = envelope, pch=21, bg="grey", xlab="Distance (m)", ylab="Semivariance", main = paste0('Semivariogram on ', var))
  
  # Get the range of autocorrelation
  # empVarioDF <- data.frame(empVario$u, empVario$v, envelope$v.lower)
  # range <- empVarioDF %>% filter(empVario.v > envelope.v.lower) %>% select(empVario.u) %>% min()
  results[[var]][['v']] <- empVario$v
  results[[var]][['v.lower']] <- envelope$v.lower
  results[[var]][['v.upper']] <- envelope$v.upper
  results[[var]][['var']] <- var
  results[[var]][['distance']] <- empVario$uvec/1000
}

plot_semivario <- results %>%
  map(as_tibble) %>%
  reduce(bind_rows) %>% 
  # mutate(distance = distance.thresholds) %>%
  # pivot_longer(-distance) %>% 
  ggplot() +
  geom_point(aes(x = distance, y = v)) +
  geom_line(aes(x = distance, y = v)) +
  geom_line(aes(x = distance, y = v.lower), linetype = 2) +
  geom_line(aes(x = distance, y = v.upper), linetype = 2) +
  facet_wrap(vars(var), scales = 'free', nrow = 1) +
  theme_bw() +
  ylab('Semivariance') + xlab('Distance (km)') 
  # theme(axis.text.y = element_blank())
plot_semivario


# Fetch SLOO-CV results
df <- fread('/Users/johanvandenhoogen/ETH/Projects/collembola/output/Biomass_dry_mcg_m2_mean_sloo_cv_results_woExtrapolation.csv') %>% 
  mutate('var' = 'Biomass_dry_mcg_m2') %>% 
  rbind(fread('/Users/johanvandenhoogen/ETH/Projects/collembola/output/Biomass_fresh_mcg_m2_mean_sloo_cv_results_woExtrapolation.csv') %>% 
          mutate('var' = 'Biomass_fresh_mcg_m2')) %>% 
  rbind(fread('/Users/johanvandenhoogen/ETH/Projects/collembola/output/CommunityMetabolism_Jh_m2_mean_sloo_cv_results_woExtrapolation.csv') %>% 
          mutate('var' = 'CommunityMetabolism_Jh_m2')) %>% 
  rbind(fread('/Users/johanvandenhoogen/ETH/Projects/collembola/output/Density_m2_mean_sloo_cv_results_woExtrapolation.csv') %>% 
          mutate('var' = 'Density_m2')) %>% 
  rbind(fread('/Users/johanvandenhoogen/ETH/Projects/collembola/output/Rarefied_sloo_cv_results_woExtrapolation.csv') %>% 
          mutate('var' = 'Rarefied')) %>% 
  mutate(buffer_size = buffer_size/1000) %>% 
  group_by(var, buffer_size) %>% 
  summarise(R2_val = mean(R2_val)) %>% 
  ungroup()

# Create composite plot
plot_coefdef <- df %>% 
  ggplot(aes(x = buffer_size, y = R2_val)) +
  geom_point() +
  geom_line() +
  scale_color_discrete() +
  theme_bw() +
  xlab("Buffer size (km)") + ylab("Coefficient of Determination R2") +
  labs(colour = "Variable") + 
  geom_hline(yintercept = 0, linetype = 'dashed') +
  facet_wrap(vars(var), nrow = 1)


# Final plot
final_plot <- plot_moran + 
  # plot_semivario +
  plot_coefdef +
  plot_layout(ncol = 1) + plot_annotation(tag_levels = 'A')
final_plot

ggsave('/Users/johanvandenhoogen/ETH/Projects/collembola/spatial_stats.pdf', final_plot)
