library(data.table)
library(h2o)
library(tidyverse)

# Load smapled dataset
rawRegressionMatrix <- fread("/Users/johanvandenhoogen/SPUN/richness_maps/data/20211026_ECM_diversity_data_sampled.csv")

bandNames <- c(
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
  'SG_Soil_pH_H2O_005cm'
)

# Name of the dependent variable
varToModel <- 'myco_diversity'

# Select the bands / covariates from the regression matrix, in addition to the dependent variable of interest
regressionMatrix <- rawRegressionMatrix %>% 
  select(varToModel, all_of(bandNames), Pixel_Lat, Pixel_Long) %>% 
  group_by(Pixel_Lat, Pixel_Long) %>% 
  summarise_all(median) %>% 
  ungroup() %>% 
  filter(varToModel != 0) %>% 
  mutate(myco_diversity = myco_diversity + 1) %>% 
  mutate(myco_diversity = log(myco_diversity)) %>% 
  select(-Pixel_Lat, -Pixel_Long)

# Initiate the H2O cluster
localH2O <- h2o.init(nthreads = 7, max_mem_size = '500g', ignore_config = TRUE)

# Import the regression matrix
regMatrixH2O <- as.h2o(regressionMatrix, destination_frame = "regMatrixH2O")

# Simple RF model, no gridsearch
rf_model <- h2o.randomForest(
  y = varToModel,
  training_frame = regMatrixH2O,
  ntrees = 500, 
  sample_rate = 0.632,
  mtries = 7,
  min_rows = 3,
  # nfolds = 10,
  seed = 123)

h2o.r2(rf_model)

###################################################
### Predicted - observed plot
FullPrediction <- as.data.frame(exp(h2o.predict(rf_model, regMatrixH2O)) - 1 )

TrainAndPredicted <- as.data.frame(FullPrediction) %>% rename(predict = 1 )
TrainAndPredicted$train <- exp(regressionMatrix[[varToModel]]) - 1

# Plot predicted vs observed values
ggplot(TrainAndPredicted, aes(x = train, y = predict)) +
  geom_point() +
  labs(x = "Predicted",
       y = "Observed") +
  geom_abline() +
  stat_smooth(method = 'lm', formula = 'y ~ x') +
  scale_x_log10() + scale_y_log10()+
  theme_bw()




# Load smapled dataset
rawRegressionMatrix <- fread("/Users/johanvandenhoogen/SPUN/richness_maps/data/20220805_ECM_diversity_log_pixelAggMedian_wSpatialPreds.csv")

bandNames <- c(
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
  # "dist1",
  # "dist2",
  # "dist3",
  # "dist4",
  # "dist5",
  # "obs1",
  # "obs2",
  # "obs3",
  # "obs4",
  # "obs5"
)

# Name of the dependent variable
varToModel <- 'ECM_diversity'

# Select the bands / covariates from the regression matrix, in addition to the dependent variable of interest
regressionMatrix <- rawRegressionMatrix %>% 
  select(varToModel, all_of(bandNames), Pixel_Lat, Pixel_Long) %>% 
  group_by(Pixel_Lat, Pixel_Long) %>% 
  summarise_all(median) %>% 
  ungroup() %>% 
  filter(varToModel != 0) %>% 
  # mutate(ECM_diversity = ECM_diversity + 1) %>% 
  # mutate(ECM_diversity = log(ECM_diversity)) %>% 
  select(-Pixel_Lat, -Pixel_Long)

# Initiate the H2O cluster
localH2O <- h2o.init(nthreads = 7, max_mem_size = '500g', ignore_config = TRUE)

# Import the regression matrix
regMatrixH2O <- as.h2o(regressionMatrix, destination_frame = "regMatrixH2O")

# Simple RF model, no gridsearch
rf_model <- h2o.randomForest(
  y = varToModel,
  training_frame = regMatrixH2O,
  ntrees = 500, 
  sample_rate = 0.632,
  mtries = 7,
  min_rows = 3,
  # nfolds = 10,
  seed = 123)

h2o.r2(rf_model)
h2o.varimp_plot(rf_model)

###################################################
### Predicted - observed plot
FullPrediction <- as.data.frame(exp(h2o.predict(rf_model, regMatrixH2O)) - 1 )

TrainAndPredicted <- as.data.frame(FullPrediction) %>% rename(predict = 1 )
TrainAndPredicted$train <- exp(regressionMatrix[[varToModel]]) - 1

# Plot predicted vs observed values
ggplot(TrainAndPredicted, aes(x = train, y = predict)) +
  geom_point() +
  labs(x = "Predicted",
       y = "Observed") +
  geom_abline() +
  stat_smooth(method = 'lm', formula = 'y ~ x') +
  scale_x_log10() + scale_y_log10()+
  theme_bw()
















# Terminate H2O session
h2o.shutdown(prompt = FALSE)

