library(data.table)
library(caret)
library(janitor)
library(tidyverse)

covariateList <- c(
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
  'SG_Soil_pH_H2O_005cm')

projectVars <- c('sequencing_platform',
                 'sample_type',
                 'primers')

# Load data
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20230206_GFv4_AM_richness_rarefied_sampled.csv') %>% 
  select(all_of(covariateList), all_of(projectVars), sample_id, rarefied, Pixel_Lat, Pixel_Long, Resolve_Biome)

# Create dummy variables
dummy <- dummyVars(" ~ .", data = df %>% select(-sample_id))

# Onehot encoding and add sample_id columnn
df_onehot <- data.frame(predict(dummy, newdata = df)) %>% 
  left_join(., df %>% select(-projectVars))

# Fix column names
names(df_onehot) <- gsub('[.]', '_', names(df_onehot))

# Print variable names
cat(names(df_onehot),sep="\n")

# Print reference levels
df_onehot %>% filter(sample_id == 'S1002')

# Write to file
fwrite(df_onehot, '/Users/johanvandenhoogen/SPUN/richness_maps/data/20230206_GFv4_AM_richness_rarefied_sampled_oneHot.csv')
