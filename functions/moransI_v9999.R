library(data.table)
library(geosphere)
library(spatialRF)
library(patchwork)
library(tidyverse)

# Set wd
setwd('/Users/johanvandenhoogen/SPUN/richness_maps')

# Distance matrix, calculate only once to speed things up
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230510_arbuscular_mycorrhizal_richness_pred_obs.csv') %>% 
  # filter(number_of_spatialpredictors == 0) %>% 
  group_by(Pixel_Lat, Pixel_Long) %>%
  filter(row_number()==1) %>%
  mutate(resid = arbuscular_mycorrhizal_richness - arbuscular_mycorrhizal_richness_Predicted)

xy <- df[, c("Pixel_Long", "Pixel_Lat")]
distance.matrix = distm(xy)/1000

# Range of distances to test Moran's I 
distance.thresholds <- c(0, 5, 10, 50, 100, 150, 200, 300, 400, 500, 750, 1000)

out <- spatialRF::moran_multithreshold(
  x = df$resid,
  distance.matrix = distance.matrix,
  distance.threshold = distance.thresholds,
  verbose = F
)

moransI <- out$per.distance 

moran.test <- spatialRF::moran(
  x = df$resid,
  distance.matrix = distance.matrix,
  verbose = FALSE
)
moran.test$plot

# Plot
moransI %>% bind_rows() %>% 
  mutate(p.value.binary = case_when(p.value >= 0.05 ~ "p >= 0.05",
                                    p.value < 0.05 ~ "p < 0.05")) %>% 
  ggplot(aes(x = distance.threshold, y = moran.i)) +
  geom_line() +
  geom_point(aes(color = p.value.binary)) +
  scale_color_manual(
    breaks = c("p < 0.05", "p >= 0.05"),
    values = c("red", "black")
  ) +
  geom_hline(yintercept = 0, linetype = 'dashed') +
  theme_bw() +
  xlab('Distance (km)') +
  ylab("Moran's I") +
  theme(legend.title=element_blank())


## Ecto

# Distance matrix, calculate only once to speed things up
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230501_ectomycorrhizal_richness_pred_obs.csv') %>% 
  # filter(number_of_spatialpredictors == 0) %>% 
  group_by(Pixel_Lat, Pixel_Long) %>%
  filter(row_number()==1) %>%
  mutate(resid = ectomycorrhizal_richness - ectomycorrhizal_richness_Predicted)

xy <- df[, c("Pixel_Long", "Pixel_Lat")]
distance.matrix = distm(xy)/1000

# Range of distances to test Moran's I 
distance.thresholds <- c(0, 5, 10, 50, 100, 150, 200, 300, 400, 500, 750, 1000)

out <- spatialRF::moran_multithreshold(
  x = df$resid,
  distance.matrix = distance.matrix,
  distance.threshold = distance.thresholds,
  verbose = F
)

moransI <- out$per.distance 

moran.test <- spatialRF::moran(
  x = df$resid,
  distance.matrix = distance.matrix,
  verbose = FALSE
)
moran_em <- moran.test$plot
moran_em

# Plot
moransI %>% bind_rows() %>% 
  mutate(p.value.binary = case_when(p.value >= 0.05 ~ "p >= 0.05",
                                    p.value < 0.05 ~ "p < 0.05")) %>% 
  ggplot(aes(x = distance.threshold, y = moran.i)) +
  geom_line() +
  geom_point(aes(color = p.value.binary)) +
  scale_color_manual(
    breaks = c("p < 0.05", "p >= 0.05"),
    values = c("red", "black")
  ) +
  geom_hline(yintercept = 0, linetype = 'dashed') +
  theme_bw() +
  xlab('Distance (km)') +
  ylab("Moran's I") +
  theme(legend.title=element_blank())






