library(tidymodels)
set.seed(123)
library(ranger)
library(data.table)
library(tidyverse)
library(h2o)


# Initiate the H2O cluster
localH2O <- h2o.init(nthreads = 7, max_mem_size = '500g', ignore_config = TRUE) 


covariateList = c(
  'CGIAR_PET',
  'CHELSA_BIO_Annual_Mean_Temperature',
  'CHELSA_BIO_Annual_Precipitation',
  'CHELSA_BIO_Max_Temperature_of_Warmest_Month',
  'CHELSA_BIO_Precipitation_Seasonality',
  'ConsensusLandCover_Human_Development_Percentage',
  # 'ConsensusLandCoverClass_Barren',
  # 'ConsensusLandCoverClass_Deciduous_Broadleaf_Trees',
  # 'ConsensusLandCoverClass_Evergreen_Broadleaf_Trees',
  # 'ConsensusLandCoverClass_Evergreen_Deciduous_Needleleaf_Trees',
  # 'ConsensusLandCoverClass_Herbaceous_Vegetation',
  # 'ConsensusLandCoverClass_Mixed_Other_Trees',
  # 'ConsensusLandCoverClass_Shrubs',
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
  # 'GlobPermafrost_PermafrostExtent',
  'MODIS_NPP',
  # 'PelletierEtAl_SoilAndSedimentaryDepositThicknesses',
  'SG_Depth_to_bedrock',
  'SG_Sand_Content_005cm',
  'SG_SOC_Content_005cm',
  'SG_Soil_pH_H2O_005cm',
  'Pixel_DistanceToNOCentroid',
  'Pixel_DistanceToNWCentroid',
  'Pixel_DistanceToSOCentroid',
  'Pixel_DistanceToSWCentroid',
  'sequencing_platform454Roche',
  'sequencing_platformIllumina',
  'sample_typerhizosphere_soil',
  'sample_typesoil',
  'sample_typetopsoil',
  'primersAML1_AML2_then_AMV4_5NF_AMDGR',
  'primersAML1_AML2_then_NS31_AM1',
  'primersAML1_AML2_then_nu_SSU_0595_5__nu_SSU_0948_3_',
  'primersAMV4_5F_AMDGR',
  'primersAMV4_5NF_AMDGR',
  'primersGeoA2_AML2_then_NS31_AMDGR',
  'primersGeoA2_NS4_then_NS31_AML2',
  'primersGlomerWT0_Glomer1536_then_NS31_AM1A_and_GlomerWT0_Glomer1536_then_NS31_AM1B',
  'primersGlomerWT0_Glomer1536_then_NS31_AM1A__GlomerWT0_Glomer1536_then_NS31_AM1B',
  'primersNS1_NS4_then_AML1_AML2',
  'primersNS1_NS4_then_AMV4_5NF_AMDGR',
  'primersNS1_NS4_then_NS31_AM1',
  'primersNS1_NS41_then_AML1_AML2',
  'primersNS31_AM1',
  'primersNS31_AML2',
  'primersWANDA_AML2')


folds <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/arbuscular_mycorrhizal_richness_training_data.csv')

regMatrixH2O <- as.h2o(folds %>% select(all_of(covariateList), 
                                        arbuscular_mycorrhizal_richness, 
                                        CV_Fold_Spatial))




h2o.randomForest(
  y = 'arbuscular_mycorrhizal_richness',
  training_frame = as.h2o(fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/arbuscular_mycorrhizal_richness_training_data.csv') %>% select(all_of(covariateList), 
                                           arbuscular_mycorrhizal_richness, 
                                           CV_Fold_Spatial)),
  ntrees = 250, 
  min_rows = 4, # Minimum leaf population
  mtries = 8, # Variables per split
  fold_column = "CV_Fold_Spatial",
  keep_cross_validation_predictions = F,
  seed = 42)@model$cross_validation_metrics_summary %>% rownames_to_column('metric') %>% filter(metric == 'r2') %>% pull(mean)






h2o.randomForest(
  y = 'arbuscular_mycorrhizal_richness',
  training_frame = regMatrixH2O,
  ntrees = 200, 
  min_rows = 10, # Minimum leaf population
  mtries = 10, # Variables per split
  nfolds = 10,
  fold_assignment = 'Random',
  keep_cross_validation_predictions = F,
  seed = 42)@model$cross_validation_metrics_summary %>% rownames_to_column('metric') %>% filter(metric == 'r2') %>% pull(mean)


