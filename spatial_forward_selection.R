list <- list()
for(num in seq(0,20)){
  df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20230217_predObs_forwardselected_spatialpredictors.csv') %>% filter(number_of_spatialpredictors == num)
  xy <- df[, c("Pixel_Long", "Pixel_Lat")]
  distance.matrix = distm(xy)/1000
  
  # Range of distances to test Moran's I 
  distance.thresholds <- c(10, 50, 100, 150, 200, 300, 400, 500, 750, 1000)
  
  out <- spatialRF::moran_multithreshold(
    x = df$residuals,
    distance.matrix = distance.matrix,
    distance.threshold = distance.thresholds,
    verbose = F
  )
  
  moransI <- out$per.distance %>% 
    mutate(number_of_spatialpredictors = num)
  
  list[[num + 1]] <- moransI
 }

df <- as.data.frame(do.call(rbind, list))


df %>% mutate(p.value.binary = case_when(p.value >= 0.05 ~ "p >= 0.05",
                                           p.value < 0.05 ~ "p < 0.05")) %>% 
  ggplot(aes(x = distance.threshold, y = moran.i)) +
  geom_line() +
  geom_point(aes(color = p.value.binary)) +
  scale_color_manual(
    breaks = c("p < 0.05", "p >= 0.05"),
    values = c("red", "black")
  ) +
  facet_wrap(vars(number_of_spatialpredictors)) +
  # ylim(c(-0.05, 0.1)) +
  geom_hline(yintercept = 0, linetype = 'dashed') +
  theme_bw() +
  xlab('Distance (km)') +
  ylab("Moran's I") +
  theme(legend.title=element_blank()) +
  ggtitle('Number of Spatial Predictors')








env = c(
  'CGIAR_PET',
  'CHELSA_BIO_Annual_Mean_Temperature',
  'CHELSA_BIO_Annual_Precipitation',
  'CHELSA_BIO_Max_Temperature_of_Warmest_Month',
  'CHELSA_BIO_Precipitation_Seasonality',
  'ConsensusLandCover_Human_Development_Percentage',
  'ConsensusLandCoverClass_Barren',
  'ConsensusLandCoverClass_Deciduous_Broadleaf_Trees',
  'ConsensusLandCoverClass_Evergreen_Broadleaf_Trees',
  'ConsensusLandCoverClass_Evergreen_Deciduous_Needleleaf_Trees',
  'ConsensusLandCoverClass_Herbaceous_Vegetation',
  'ConsensusLandCoverClass_Mixed_Other_Trees',
  'ConsensusLandCoverClass_Shrubs',
  'EarthEnvTexture_CoOfVar_EVI',
  'EarthEnvTexture_Correlation_EVI',
  'EarthEnvTexture_Homogeneity_EVI',
  'EarthEnvTopoMed_AspectCosine',
  'EarthEnvTopoMed_AspectSine',
  'EarthEnvTopoMed_Elevation',
  'EarthEnvTopoMed_Slope',
  'EarthEnvTopoMed_TopoPositionIndex',
  'EsaCci_BurntAreasProbability',
  'GHS_Population_Density',
  'GlobBiomass_AboveGroundBiomass',
  'GlobPermafrost_PermafrostExtent',
  'MODIS_NPP',
  'PelletierEtAl_SoilAndSedimentaryDepositThicknesses',
  'SG_Depth_to_bedrock',
  'SG_Sand_Content_005cm',
  'SG_SOC_Content_005cm',
  'SG_Soil_pH_H2O_005cm',
  # 'target_marker',
  'sequencing_platform',
  'sample_type',
  'primers'
)

spatial_preds = c('MEM1', 'MEM10', 'MEM11', 'MEM13', 'MEM18', 'MEM19', 'MEM20', 'MEM30', 'MEM35', 'MEM37', 'MEM4', 'MEM45', 'MEM51', 'MEM52', 'MEM58', 'MEM6', 'MEM7', 'MEM8', 'MEM81', 'MEM9')
list <- list()

df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/arbuscular_mycorrhizal_richness_training_data_wMEMs.csv') %>% na.omit()
xy <- df[, c("Pixel_Long", "Pixel_Lat")]
distance.matrix <- distm(xy)/1000

# For each variable, train a non-spatial RF and retrieve Moran's I
for(num in seq(0,20)){
  df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/arbuscular_mycorrhizal_richness_training_data_wMEMs.csv') %>% na.omit()
  
  covariateList <- c(env, spatial_preds[0:num])
  training_data <- df %>% select(all_of(covariateList),arbuscular_mycorrhizal_richness, Pixel_Lat, Pixel_Long) 
  xy <- training_data[, c("Pixel_Long", "Pixel_Lat")]
  
  model.non.spatial <- spatialRF::rf(
    data = training_data,
    dependent.variable.name = 'arbuscular_mycorrhizal_richness',
    predictor.variable.names = covariateList,
    distance.matrix = distance.matrix,
    distance.thresholds = distance.thresholds,
    xy = xy, 
    seed = 123,
    verbose = FALSE
  )
  
  moransI <- model.non.spatial$residuals$autocorrelation$per.distance %>% 
    mutate(number_of_spatialpredictors = num)
  
  list[[num + 1]] <- moransI
}

df <- as.data.frame(do.call(rbind, list))

df %>% mutate(p.value.binary = case_when(p.value >= 0.05 ~ "p >= 0.05",
                                         p.value < 0.05 ~ "p < 0.05")) %>% 
  ggplot(aes(x = distance.threshold, y = moran.i)) +
  geom_line() +
  geom_point(aes(color = p.value.binary)) +
  scale_color_manual(
    breaks = c("p < 0.05", "p >= 0.05"),
    values = c("red", "black")
  ) +
  facet_wrap(vars(number_of_spatialpredictors)) +
  # ylim(c(-0.05, 0.1)) +
  geom_hline(yintercept = 0, linetype = 'dashed') +
  theme_bw() +
  xlab('Distance (km)') +
  ylab("Moran's I") +
  theme(legend.title=element_blank()) +
  ggtitle('Number of Spatial Predictors')






