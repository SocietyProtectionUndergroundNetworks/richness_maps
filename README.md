# SPUN Fungal Richness maps
This repository contains code and data to reproduce the Arbuscular Mycorrhizal (AM) and Ectomycorrhizal (ECM) richness maps from Van Nuland et al. 202x doi: 10.xxx.xxx

## Rarefaction
AM and ECM rarefied richness are obtained from [GlobalFungi](https://globalfungi.com/). Richness values are rarefied using the R package `iNEXT`. Code and data to reproduce this are found in the folder `xxx`

## Geospatial modeling
The geospatial modeling approach is divided in several parts, with the respecitve scripts numbered accordingly.
* 1 Covariate sampling. Here, we extract per-pixel environmental covariate data.
* 2 Data filtering. Removing outliers per biome.
* 3 One-hot encoding. Transforming the project-specific variables from multilevel categorical to binary format.
* 4 Modeling pipeline. The actual modeling approach. 

