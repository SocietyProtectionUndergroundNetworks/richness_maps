suppressPackageStartupMessages(library(doParallel, quietly = T))
suppressPackageStartupMessages(library(data.table, quietly = T))
suppressPackageStartupMessages(library(sf, quietly = T))
suppressPackageStartupMessages(library(h3jsr, quietly = T))
suppressPackageStartupMessages(library(blockCV, quietly = T))

# setwd('/Users/johanvandenhoogen/SPUN/richness_maps')

# Main function
main <- function(k, type, inputPath, lonString, latString, crs, seed) {
  # Set default values 
  if(length(lonString) == 0){lonString = 'longitude'}
  if(length(latString) == 0){latString = 'latitude'}
  if(type == 'Rectangles'){type = 'Rectangle'}
  if(type == 'Hexagons'){type = 'Hexagon'}
  if(length(crs) == 0){crs = 'EPSG:4326'}
  
  print(paste0('Generating ',k,' folds using ',type,' shapes in the ',crs,' projection.'))
  generateFolds(k, type, inputPath, lonString, latString, crs, seed)
  
}

# Function to generate the folds 
generateFolds <- function(k,
                          type,
                          inputPath,
                          lonString,
                          latString,
                          crs='EPSG:4326',
                          seed) {
  
  # Load the data and transform into spatial features 
  sampleLocations <- fread(inputPath)
  sampleLocations_sf <- st_as_sf(sampleLocations, coords=c(lonString, latString), crs='EPSG:4326')
  
  ### Uber H3 
  # Hexagons with roughly equally sized area
  if(type == 'H3'){
    # Get the polygons
    listOfPolygons <- list()
    for (i in 0:4){
      listOfPolygons[[i+1]] <- cell_to_polygon(unlist(get_children(h3_address = get_res0(), res = i, simple = TRUE)))
    }
    
    # Generate folds
    # Make a cluster for parallel computation
    cl = makeCluster(min(detectCores()-1, 5))
    registerDoParallel(cl)
    foldIDs <- foreach(i=1:5, .packages=(.packages()), .errorhandling='remove') %dopar% {
      folds <- cv_spatial(x = sampleLocations_sf,
                          k = k,
                          user_blocks = listOfPolygons[[i]],
                          selection = "random",
                          iteration = 100,
                          progress = FALSE,
                          report = FALSE,
                          biomod2 = FALSE,
                          plot = FALSE,
                          seed = seed)
      foldIDs <- as.data.frame(folds$folds_ids)
      foldName <- paste0('foldID_H3res',i-1)
      colnames(foldIDs) <- foldName
      return(foldIDs)
    }
    stopCluster(cl)
    sampleLocations <- cbind(sampleLocations, data.frame(foldIDs))
  }
  
  ### Rectangles in different sizes
  # !! Be aware that some sizes won't work and that some samples might get excluded
  # !! Also, depending on the CRS, the hexagons are either 
  #     (i) not equally sized (EPSG:4326), which means that folds far away from the equator 
  #     are much smaller in size and thus get penalized, or
  #     (ii) not proper rectangles (EPSG:8857), which means that distances are not coherent across the globe
  if(type == 'Rectangle'){
    if(crs == 'EPSG:8857'){
      # Reload the data and transform into spatial features in equal area projection
      sampleLocations <- fread(inputPath)
      sampleLocations_sf <- st_as_sf(sampleLocations, coords=c(lonString, latString), crs='EPSG:4326')
      sampleLocations_sf <- st_transform(sampleLocations_sf, crs=crs)
      # Block sizes in m
      blockSizes <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20) * 100 * 1e3
      bS <- blockSizes / 1e3
      deg_to_metre <- 111325
      unit <- 'km'
    }
    else{
      # Define block sizes to loop over 
      blockSizes <- bS <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20)
      deg_to_metre <- 1
      unit <- 'deg'
    }
    
    # Make a cluster for parallel computation
    cl = makeCluster(detectCores()-1)
    registerDoParallel(cl)
    foldIDs <- foreach(i=1:length(blockSizes), .packages=(.packages()), .errorhandling='remove') %dopar% {
      # Generate folds
      folds <- cv_spatial(x = sampleLocations_sf,
                          k = k,
                          size = blockSizes[i],
                          deg_to_metre = deg_to_metre,
                          selection = "random",
                          progress = FALSE,
                          report = FALSE,
                          hexagon = FALSE,
                          iteration = 100,
                          biomod2 = FALSE,
                          plot = FALSE,
                          seed = seed)
      foldIDs <- as.data.frame(folds$folds_ids)
      foldName <- paste0('foldID_',bS[i],unit,'_',type)
      colnames(foldIDs) <- foldName
      return(foldIDs)
    }
    stopCluster(cl)
    sampleLocations <- cbind(sampleLocations, data.frame(foldIDs))
  }  
  
  ### Hexagons in different sizes
  # !! Be aware that some sizes won't work and that some samples might get excluded
  # !! Also, depending on the CRS, the hexagons are either 
  #     (i) not equally sized (EPSG:4326), which means that folds far away from the equator 
  #     are much smaller in size and thus get penalized, or
  #     (ii) not proper hexagons (EPSG:8857), which means that distances are not coherent across the globe
  if(type == 'Hexagon'){
    if(crs == 'EPSG:8857'){
      # Reload the data and transform into spatial features in equal area projection
      sampleLocations <- fread(inputPath)
      sampleLocations_sf <- st_as_sf(sampleLocations, coords=c(lonString, latString), crs='EPSG:4326')
      sampleLocations_sf <- st_transform(sampleLocations_sf, crs=crs)
      # Block sizes in m
      blockSizes <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20) * 100 * 1e3
      bS <- blockSizes / 1e3
      deg_to_metre <- 111325
      unit <- 'km'
    }
    else{
      # Define block sizes to loop over 
      blockSizes <- bS <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20)
      deg_to_metre <- 1
      unit <- 'deg'
    }
    
    # Make a cluster for parallel computation
    cl = makeCluster(detectCores()-1)
    registerDoParallel(cl)
    foldIDs <- foreach(i=1:length(blockSizes), .packages=(.packages()), .errorhandling='remove') %dopar% {
      # Generate folds
      folds <- cv_spatial(x = sampleLocations_sf,
                          k = k,
                          size = blockSizes[i],
                          deg_to_metre = deg_to_metre,
                          selection = "random",
                          progress = FALSE,
                          report = FALSE,
                          hexagon = TRUE,
                          iteration = 100,
                          biomod2 = FALSE,
                          plot = FALSE,
                          seed = seed)
      foldIDs <- as.data.frame(folds$folds_ids)
      foldName <- paste0('foldID_',bS[i],unit,'_',type)
      colnames(foldIDs) <- foldName
      return(foldIDs)
    }
    stopCluster(cl)
    sampleLocations <- cbind(sampleLocations, data.frame(foldIDs))
  }
  
  
  # Omit samples in case NAs got introduced
  sampleLocations <- na.omit(sampleLocations)
  
  # TEMP: skip writing to file
  # Save the generated folds
  # outputPath <- gsub(".csv", "", inputPath)
  # fwrite(sampleLocations, paste0(outputPath, "_wSpatialFolds_",type,".csv"))
  
  sampleLocations
}

folds <- main(k = 10,
     type = 'Rectangle',
     inputPath = 'data/arbuscular_mycorrhizal_richness_training_data.csv',
     lonString = 'Pixel_Long',
     latString = 'Pixel_Lat',
     crs = 'EPSG:4326',
     seed = 0)

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
  'ConsensusLandCoverClass_Shrubs',
  'EarthEnvTexture_CoOfVar_EVI',
  'EarthEnvTexture_Correlation_EVI',
  'EarthEnvTexture_Homogeneity_EVI',
  'EarthEnvTopoMed_AspectCosine',
  'EarthEnvTopoMed_AspectSine',
  'EarthEnvTopoMed_Elevation',
  'EarthEnvTopoMed_Slope',
  'EarthEnvTopoMed_TopoPositionIndex',
  # 'EsaCci_BurntAreasProbability',
  'GHS_Population_Density',
  'GlobBiomass_AboveGroundBiomass',
  # 'GlobPermafrost_PermafrostExtent',
  'MODIS_NPP',
  # 'PelletierEtAl_SoilAndSedimentaryDepositThicknesses',
  'SG_Depth_to_bedrock',
  'SG_Sand_Content_005cm',
  'SG_SOC_Content_005cm',
  'SG_Soil_pH_H2O_005cm',
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

spatial_preds = c('MEM1', 'MEM10', 'MEM11', 'MEM13', 'MEM18', 'MEM19', 'MEM20', 'MEM30', 'MEM35', 'MEM37', 'MEM4', 'MEM45', 'MEM51', 'MEM52', 'MEM58', 'MEM6', 'MEM7', 'MEM8', 'MEM81', 'MEM9')

library(h2o)

# Initiate the H2O cluster
localH2O <- h2o.init(nthreads = 7, max_mem_size = '500g', ignore_config = TRUE) 

# Import the regression matrix
regMatrixH2O <- as.h2o(folds %>% select(all_of(covariateList), arbuscular_mycorrhizal_richness, foldID_H3res1))

list <- list()
for(i in c(0,1,2,3,4)){
  regMatrixH2O <- as.h2o(folds %>% select(all_of(covariateList), arbuscular_mycorrhizal_richness, paste0('foldID_H3res',i)))
  
  list[[i + 1]] <- h2o.randomForest(
    y = 'arbuscular_mycorrhizal_richness',
    training_frame = regMatrixH2O,
    ntrees = 200, 
    min_rows = 10, # Minimum leaf population
    mtries = 10, # Variables per split
    fold_column = paste0('foldID_H3res',i),
    keep_cross_validation_predictions = F,
    seed = 42)@model$cross_validation_metrics_summary# %>% rownames_to_column('metric') %>% filter(metric == 'r2')
  
  
}

folds <- folds %>% left_join(fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20230206_GFv4_AM_richness_rarefied_sampled.csv') %>% select('sample_id', starts_with('Pixel_Distance')))
folds

regMatrixH2O <- as.h2o(folds %>% select(all_of(covariateList), starts_with('Pixel_Distance'), arbuscular_mycorrhizal_richness, foldID_5deg_Rectangle))


# cl = makeCluster(detectCores()-1)
# registerDoParallel(cl)
# gridSearch <- foreach(i = seq(2,20,2), .packages=(.packages())) %dopar% {
#   localH2O <- h2o.init(nthreads = 7, max_mem_size = '500g', ignore_config = TRUE) 
#   
#   summary <- h2o.randomForest(
#     y = 'arbuscular_mycorrhizal_richness',
#     training_frame = regMatrixH2O,
#     ntrees = 200, 
#     min_rows = 8, # Minimum leaf population
#     mtries = i, # Variables per split
#     fold_column = paste0('foldID_H3res1'),
#     keep_cross_validation_predictions = F,
#     seed = 42)@model$cross_validation_metrics_summary %>% rownames_to_column('metric')
#   
#   summary$mries = i
#   
#   return(summary)
# }
# # stopCluster(cl)
# gridSearch <- lapply(gridSearch, c)
# gridSearch %>% bind_rows() %>% select(metric, mean, sd, mries) %>% filter(metric == 'r2')

# # Spatial CV
# h2o.r2(h2o.randomForest(
#   y = 'arbuscular_mycorrhizal_richness',
#   training_frame = regMatrixH2O,
#   ntrees = 200, 
#   min_rows = 10, # Minimum leaf population
#   mtries = 10, # Variables per split
#   fold_column = "foldID_5deg_Rectangle",
#   keep_cross_validation_predictions = F,
#   seed = 42), xval = TRUE)
# 
# # Random CV
# h2o.r2(h2o.randomForest(
#   y = 'arbuscular_mycorrhizal_richness',
#   training_frame = regMatrixH2O,
#   ntrees = 200, 
#   min_rows = 5, # Minimum leaf population
#   mtries = 10, # Variables per split
#   nfolds = 10,
#   fold_assignment = 'Random',
#   keep_cross_validation_predictions = F,
#   seed = 123), xval = TRUE)
folds
h2o.randomForest(
  y = 'arbuscular_mycorrhizal_richness',
  training_frame = regMatrixH2O,
  ntrees = 250, 
  min_rows = 8, # Minimum leaf population
  mtries = 14, # Variables per split
  fold_column = "foldID_5deg_Rectangle",
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


# rf_model@model$cross_validation_metrics_summary %>% rownames_to_column('metric') %>% filter(metric == 'r2') %>% pull(mean)



search_criteria <- list(strategy = "RandomDiscrete", max_models = 100, seed = 123)

# Perform RF grid search across parameters
RF_params <- list(ntrees = 250,
                  # max_depth = seq(5, 30, 5),
                  mtries = seq(2,10,1), # Variables per split
                  min_rows = seq(2,10,2) # Minimum leaf population
)

RF_grid <- h2o.grid("randomForest",
                    y = 'arbuscular_mycorrhizal_richness',
                    grid_id = "RF_grid",
                    training_frame = regMatrixH2O,
                    seed = 123,
                    hyper_params = RF_params,
                    sample_rate = 0.632,
                    fold_column = "foldID_H3res1",
                    search_criteria = search_criteria
)

# Retrieve grid searched model performance, sort by RMSE
RF_gridperf <- h2o.getGrid(grid_id = "RF_grid",
                           sort_by = "R2",
                           decreasing = TRUE)

