list <- list()
for(num in seq(0,20)){
  df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20230217_predObs_forwardselected_spatialpredictors.csv') %>% filter(number_of_spatialpredictors == num)
  xy <- df[, c("Pixel_Long", "Pixel_Lat")]
  distance.matrix = distm(xy)/1000
  
  # Range of distances to test Moran's I 
  distance.thresholds <- c(1, 10, 25, 50, 100, 150, 250, 500, 1000)
  
  out <- spatialRF::moran_multithreshold(
    x = df$AbsResidual,
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

