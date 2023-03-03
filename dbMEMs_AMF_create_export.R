rm(list = ls()) 

# Load the required packages
library(tidyverse)
library(data.table)
library(sf)
library(sp)
library(ape)
library(spdep)
library(ade4)
library(adegraphics)
library(adespatial)
library(vegan)
# Source additional functions

source("quickMEM.R") 


# Arbuscular mycorrhizal fungi
fungi_sf <- fread("data/20230206_GFv4_AM_richness_rarefied_sampled_oneHot.csv") |>
  dplyr::select(sample_id, rarefied, Pixel_Lat, Pixel_Long) |>
  as.data.frame() |>
  drop_na() |>
  st_as_sf(coords = c("Pixel_Long", "Pixel_Lat"), crs = 4326) |>
  mutate(geom_text = as.character(geometry)) |>
  # cbind(st_coordinates(.)) |>
  rename(AMF_diversity = rarefied) |>
  select(sample_id, geom_text, AMF_diversity)

# Aggregate richness values to unique geometries (points)
fungi_sf_woDupl <- fungi_sf |>
  group_by(geom_text) |>
  summarise(AMF_diversity = median(AMF_diversity)) |>
  select(geom_text, AMF_diversity)

# Split w/o duplicates data by richness and coordinates
AMF_diversity <- fungi_sf_woDupl$AMF_diversity
amf.xy <- st_coordinates(fungi_sf_woDupl)

# Transform the data 
amf.s <- scale(AMF_diversity, center = TRUE, scale = FALSE)
amf.xy.c <- scale(amf.xy, center = TRUE, scale = FALSE)

amf.dbmem.quick <- quickMEM(amf.s, amf.xy)
summary(amf.dbmem.quick)
# Save eigenvectors and richness data into single object
amf.dbMEM.uniqueGeom.allData <- list(amf.dbmem.quick, fungi_sf, fungi_sf_woDupl)
save(amf.dbMEM.uniqueGeom.allData, file = "amf.dbMEM.uniqueGeom.allData.Rdata")

# Load MEMs back and add as new spatial covariate
load(file = "amf.dbMEM.uniqueGeom.allData.Rdata")
amf.dbmem.quick <- amf.dbMEM.uniqueGeom.allData[[1]]
fungi_sf <- amf.dbMEM.uniqueGeom.allData[[2]]
fungi_sf_woDupl <- amf.dbMEM.uniqueGeom.allData[[3]]

summary(amf.dbmem.quick)

# Eigenvalues
amf.dbmem.quick[[2]] # OR amf.dbmem.quick$eigenvalues

# Results of forward selection
amf.dbmem.quick[[3]] # OR amf.dbmem.quick$fwd.sel

# Selected eigenvectors
MEMs <- amf.dbmem.quick[[4]] # OR amf.dbmem.quick$dbMEM_red_model

# Create non-spatial data frame with MEMs for each unique geometry
df <- fungi_sf_woDupl |>
  cbind(MEMs) |>
  st_drop_geometry() |>
  select(-AMF_diversity)

# Merge source data with the MEMs
dbMEMs <- fungi_sf |>
  left_join(df) |>
  select(-geom_text, -AMF_diversity)

save(dbMEMs, file = "dbMEMs_AMF.Rdata")

# Add MEMs to the training data
# Arbuscular mycorrhizal fungi
fread("data/20230206_GFv4_AM_richness_rarefied_sampled_oneHot.csv") |>
  left_join(dbMEMs) |>
  select(-geometry) |>
  fwrite(file = "20230206_GFv4_AM_richness_rarefied_sampled_oneHot_wMEMs.csv")
