library(data.table)
library(patchwork)
library(tidyverse)

predobs <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/spatial/20220815_ECM_diversity_pred_obs_spatial.csv') %>% 
  group_by(Pixel_Lat, Pixel_Long) %>% 
  summarise_all(mean) %>% 
  ungroup() %>% 
  select(contains('ECM'))

p1<- predobs %>% 
  mutate(ECM_diversity = exp(ECM_diversity) - 1) %>%
  mutate(ECM_diversity_Predicted = exp(ECM_diversity_Predicted) - 1) %>%
  ggplot(aes(x=ECM_diversity, y =ECM_diversity_Predicted))+
  geom_point()+
  coord_equal()+
  geom_abline()+
  geom_smooth(method = 'lm', formula = 'y ~ x') +
  scale_x_log10() + scale_y_log10() +
  ggtitle('Median per pixel, spatial predictors')


predobs_woSpatial <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/spatial/woSpatial/20220817_ECM_diversity_pred_obs_spatial_woSpatial.csv') %>% 
  group_by(Pixel_Lat, Pixel_Long) %>% 
  summarise_all(mean) %>%
  ungroup() %>% 
  select(contains('ECM'))

p2 <- predobs_woSpatial %>% 
  mutate(ECM_diversity = exp(ECM_diversity) - 1) %>%
  mutate(ECM_diversity_Predicted = exp(ECM_diversity_Predicted) - 1) %>%
  ggplot(aes(x=ECM_diversity, y =ECM_diversity_Predicted))+
  geom_point()+
  coord_equal()+
  geom_abline()+
  geom_smooth(method = 'lm', formula = 'y ~ x') +
  scale_x_log10() + scale_y_log10()+
  ggtitle('Median per pixel, no spatial predictors')




predobs_current <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220819_ECM_diversity_pred_obs_zeroInflated_distictObs_wProjectVars.csv') %>% 
  select(contains('ECM'), Pixel_Lat, Pixel_Long) %>% 
  group_by(Pixel_Lat, Pixel_Long) %>%
  summarise_all(mean) %>%
  ungroup() %>%
  select(contains('ECM'))

p3<- 
  predobs_current %>% 
  # mutate(ECM_diversity = exp(ECM_diversity) - 1) %>%
  # mutate(ECM_diversity_Predicted = exp(ECM_diversity_Predicted) - 1) %>%
  ggplot(aes(x=ECM_diversity, y =ECM_diversity_Predicted))+
  geom_point()+
  coord_equal()+
  geom_abline()+
  geom_smooth(method = 'lm', formula = 'y ~ x') +
  scale_x_log10() + scale_y_log10()+
  ggtitle('Current approach')


# p1 + p2 + p3


df <- predobs %>% 
  mutate(ECM_diversity = exp(ECM_diversity) - 1) %>%
  mutate(ECM_diversity_Predicted = exp(ECM_diversity_Predicted) - 1) %>% 
  mutate(approach = 'Pixel median\nWith spatial predictors\nx-val R2: 0.60') %>% 
  rbind(., predobs_woSpatial %>% 
          mutate(ECM_diversity = exp(ECM_diversity) - 1) %>%
          mutate(ECM_diversity_Predicted = exp(ECM_diversity_Predicted) - 1) %>%
          mutate(approach = 'Pixel median\n Without spatial predictors\nx-val R2: 0.39')) %>% 
  rbind(., predobs_current %>% 
          mutate(approach = 'Distinct observations\nMedian predicted-observed\nx-val R2: 0.65')) %>% 
  mutate(approach = factor(approach, levels = c("Distinct observations\nMedian predicted-observed\nx-val R2: 0.65",
                                              "Pixel median\nWith spatial predictors\nx-val R2: 0.60" ,
                                              "Pixel median\n Without spatial predictors\nx-val R2: 0.39")))

df %>% ggplot(aes(x=ECM_diversity, y =ECM_diversity_Predicted)) +
  geom_point(alpha = 0.1) +
  coord_equal() +
  geom_abline() +
  geom_smooth(method = 'lm', formula = 'y ~ x') +
  scale_x_log10() + scale_y_log10() +
  # coord_cartesian(expand = TRUE) +
  facet_wrap(vars(approach)) +
  xlab('Observed') + ylab('Predicted') +
  theme_bw()
gag