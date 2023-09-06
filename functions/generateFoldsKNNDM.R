#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(data.table, quietly = T))
suppressPackageStartupMessages(library(sf, quietly = T))
suppressPackageStartupMessages(library(tidyverse, quietly = T))

# Main function
main <- function(tpoints, modeldomain = NULL, ppoints = NULL,
                 k = 10, maxp = 0.5,
                 clustering = "hierarchical", linkf = "ward.D2",
                 samplesize = 1000, sampling = "regular") {
  
  # input formatting ------------
  args = commandArgs(trailingOnly=TRUE)
  
  path_pos <- which(args == '--path_training')
  path_random_pos <- which(args == '--path_random')
  # tpoints_pos <- which(args == '--tpoints')
  modeldomain_pos <- which(args == '--modeldomain')
  ppoints_pos <- which(args == '--ppoints')
  k_pos <- which(args == '--k')
  maxp_pos <- which(args == '--maxp')
  clustering_pos <- which(args == '--clustering')
  linkf_pos <- which(args == '--linkf')
  samplesize_pos <- which(args == '--samplesize')
  sampling_pos <- which(args == '--sampling')  
  
  path <- args[path_pos+1]
  # path_random <- args[path_random_pos+1]
  # tpoints <- args[tpoints_pos+1]
  modeldomain <- args[modeldomain_pos+1]
  ppoints <- args[ppoints_pos+1]
  k <- as.numeric(args[k_pos+1])
  maxp <- as.numeric(args[maxp_pos+1])
  clustering <- args[clustering_pos+1]
  linkf <- args[linkf_pos+1]
  samplesize <- as.numeric(args[samplesize_pos+1])
  sampling <- args[sampling_pos+1]

  # Load data and convert to sf
  points <- fread(path) %>%
    st_as_sf(., coords = c("Pixel_Long", "Pixel_Lat"),
             crs= "+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0")

  # Load random data as proxy for model domain, convert to sf
  ppoints <- fread(ppoints) %>% 
    filter(!is.na(Pixel_Lat)) %>% 
    st_as_sf(., coords = c("Pixel_Long", "Pixel_Lat"),
             crs= "+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0")
            
  # Calculate fold assignment
  output <- knndmW(points, modeldomain, ppoints, k, maxp, clustering, linkf, samplesize, sampling)
  
  # Select fold assignment with lowest W statistic, add to data
  df <- points %>% 
    mutate(knndmw_CV_folds = as.factor(output$clusters[[which.min(output$W)]])) %>% 
    mutate(Pixel_Long = st_coordinates(geometry)[, "X"],
         Pixel_Lat = st_coordinates(geometry)[, "Y"])

  # Write to file
  fwrite(df, file = path)
}

# knndmW function
knndmW <- function(tpoints, modeldomain = NULL, ppoints = NULL,
                   k = 10, maxp = 0.5,
                   clustering = "hierarchical", linkf = "ward.D2",
                   samplesize = 1000, sampling = "regular"){
  
  # create sample points from modeldomain
  if(is.null(ppoints)&!is.null(modeldomain)){
    if(!identical(sf::st_crs(tpoints), sf::st_crs(modeldomain))){
      stop("tpoints and modeldomain must have the same CRS")
    }
    message(paste0(samplesize, " prediction points are sampled from the modeldomain"))
    ppoints <- sf::st_sample(x = modeldomain, size = samplesize, type = sampling)
    sf::st_crs(ppoints) <- sf::st_crs(modeldomain)
  }else if(!is.null(ppoints)){
    if(!identical(sf::st_crs(tpoints), sf::st_crs(ppoints))){
      stop("tpoints and ppoints must have the same CRS")
    }
  }
  
  # Prior checks
  if (!(clustering %in% c("kmeans", "hierarchical"))) {
    stop("clustering must be one of `kmeans` or `hierarchical`")
  }
  if (!(maxp < 1 & maxp > 1/k)) {
    stop("maxp must be strictly between 1/k and 1")
  }
  if (any(class(tpoints) %in% "sfc")) {
    tpoints <- sf::st_sf(geom = tpoints)
  }
  if (any(class(ppoints) %in% "sfc")) {
    ppoints <- sf::st_sf(geom = ppoints)
  }
  if(is.na(sf::st_crs(tpoints))){
    warning("Missing CRS in training or prediction points. Assuming projected CRS.")
    islonglat <- FALSE
  }else{
    islonglat <- sf::st_is_longlat(tpoints)
  }
  if(isTRUE(islonglat) & clustering == "kmeans"){
    stop("kmeans works in the Euclidean space and therefore can only handle
         projected coordinates. Please use hierarchical clustering or project your data.")
  }
  
  # Gj and Gij calculation
  tcoords <- sf::st_coordinates(tpoints)[,1:2]
  if(isTRUE(islonglat)){
    distmat <- sf::st_distance(tpoints)
    units(distmat) <- NULL
    diag(distmat) <- NA
    Gj <- apply(distmat, 1, function(x) min(x, na.rm=TRUE))
    Gij <- sf::st_distance(ppoints, tpoints)
    units(Gij) <- NULL
    Gij <- apply(Gij, 1, min)
  }else{
    Gj <- c(FNN::knn.dist(tcoords, k = 1))
    Gij <- c(FNN::knnx.dist(query = sf::st_coordinates(ppoints)[,1:2],
                            data = tcoords, k = 1))
  }
  if(clustering == "hierarchical"){
    # For hierarchical clustering we need to compute the full distance matrix,
    # but we can integrate geographical distances
    if(!isTRUE(islonglat)){
      distmat <- sf::st_distance(tpoints)
    }
    hc <- stats::hclust(d = stats::as.dist(distmat), method = linkf)
  }
  
  # Build grid of number of clusters to try - we sample low numbers more intensively
  clustgrid <- data.frame(nk = as.integer(round(exp(seq(log(k), log(nrow(tpoints)-2),
                                                        length.out = 100)))))
  clustgrid$W <- NA
  clustgrid <- clustgrid[!duplicated(clustgrid$nk),]
  clustgroups <- list()
  
  # Compute 1st PC for ordering clusters
  pcacoords <- stats::prcomp(tcoords, center = TRUE, scale. = FALSE, rank = 1)
  
  # We test each number of clusters
  for(nk in clustgrid$nk){
    
    # Create nk clusters
    if(clustering == "hierarchical"){
      clust_nk <- stats::cutree(hc, k=nk)
    }else if(clustering == "kmeans"){
      clust_nk <- stats::kmeans(tcoords, nk)$cluster
    }
    
    tabclust <- as.data.frame(table(clust_nk))
    tabclust$clust_k <- NA
    
    # compute cluster centroids and apply PC loadings to shuffle along the 1st dimension
    centr_tpoints <- sapply(tabclust$clust_nk, function(x){
      centrpca <- matrix(apply(tcoords[clust_nk %in% x, , drop=FALSE], 2, mean), nrow = 1)
      colnames(centrpca) <- colnames(tcoords)
      return(predict(pcacoords, centrpca))
    })
    tabclust$centrpca <- centr_tpoints
    tabclust <- tabclust[order(tabclust$centrpca),]
    
    # We don't merge big clusters
    clust_i <- 1
    for(i in 1:nrow(tabclust)){
      if(tabclust$Freq[i] >= nrow(tpoints)/k){
        tabclust$clust_k[i] <- clust_i
        clust_i <- clust_i + 1
      }
    }
    rm("clust_i")
    
    # And we merge the remaining into k groups
    clust_i <- setdiff(1:k, unique(tabclust$clust_k))
    tabclust$clust_k[is.na(tabclust$clust_k)] <- rep(clust_i, ceiling(nk/length(clust_i)))[1:sum(is.na(tabclust$clust_k))]
    tabclust2 <- data.frame(ID = 1:length(clust_nk), clust_nk = clust_nk)
    tabclust2 <- merge(tabclust2, tabclust, by = "clust_nk")
    tabclust2 <- tabclust2[order(tabclust2$ID),]
    clust_k <- tabclust2$clust_k
    
    # Compute W statistic if not exceeding maxp
    if(!any(table(clust_k)/length(clust_k)>maxp)){
      
      if(isTRUE(islonglat)){
        Gjstar_i <- distclust_geo(distmat, clust_k)
      }else{
        Gjstar_i <- distclust_proj(tcoords, clust_k)
      }
      clustgrid$W[clustgrid$nk==nk] <- twosamples::wass_stat(Gjstar_i, Gij)
      clustgroups[[paste0("nk", nk)]] <- clust_k
    }
  }
  
  # return every split
  W_final <- clustgrid[!is.na(clustgrid$W),"W"]
  clust <- clustgroups[!is.na(clustgrid$W)]
  
  # Output
  res <- list(clusters = clust, W=W_final)
  class(res) <- c("list")
  res
}


# Helper function: Compute out-of-fold NN distance (geographical)
distclust_geo <- function(distm, folds){
  alldist <- rep(NA, length(folds))
  for(f in unique(folds)){
    alldist[f == folds] <- apply(distm[f == folds, f != folds, drop=FALSE], 1, min)
  }
  alldist
}

# Helper function: Compute out-of-fold NN distance (projected)
distclust_proj <- function(tr_coords, folds){
  alldist <- rep(NA, length(folds))
  for(f in unique(folds)){
    alldist[f == folds] <- c(FNN::knnx.dist(query = tr_coords[f == folds,,drop=FALSE],
                                            data = tr_coords[f != folds,,drop=FALSE], k = 1))
  }
  alldist
}
main()
