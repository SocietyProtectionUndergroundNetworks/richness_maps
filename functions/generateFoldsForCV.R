#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(doParallel, quietly = T))
suppressPackageStartupMessages(library(data.table, quietly = T))
suppressPackageStartupMessages(library(sf, quietly = T))
suppressPackageStartupMessages(library(h3jsr, quietly = T))
suppressPackageStartupMessages(library(blockCV, quietly = T))

# Main function
main <- function(k, type, inputPath, lonString, latString, crs, seed) {
  
  # input formatting ------------
  args = commandArgs(trailingOnly=TRUE)
  
  k_pos <- which(args == '--k')
  type_pos <- which(args == '--type')
  path_pos <- which(args == '--path')
  lon_pos <- which(args == '--lon')
  lat_pos <- which(args == '--lat')
  crs_pos <- which(args == '--crs')
  seed_pos <- which(args == '--seed')
  
  k <- args[k_pos+1]
  type <- args[type_pos+1]
  inputPath <- args[path_pos+1]
  lonString <- args[lon_pos+1]
  latString <- args[lat_pos+1]
  crs <- args[crs_pos+1]
  seed <- args[seed_pos+1]
  
  if(type == 'H3'){print(paste0('Generating ',k,' folds using ',type,' shapes'))}
  if(type %in% c('Rectangle', 'Hexagon')){print(paste0('Generating ',k,' folds using ',type,' shapes in the ',crs,' projection.'))}  
  generateFolds(k, type, inputPath, lonString, latString, crs, seed)
}

# Function to generate the folds 
generateFolds <- function(k = 10,
                          type = 'Rectangle',
                          inputPath,
                          lonString = 'longitude',
                          latString = 'latitude',
                          crs = 'EPSG:4326',
                          seed = 123) {
  
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
  
  # Save the generated folds
  outputPath <- gsub(".csv", "", inputPath)
  fwrite(sampleLocations, inputPath)
  
}

main()
