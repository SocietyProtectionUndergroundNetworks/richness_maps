library(data.table)
library(tidyverse)
setwd('/Users/johanvandenhoogen/SPUN/richness_maps/')
# Raw results have 10 predictions per point (pixel); one for each model in the ensemble. Process by taking the mean for each predicted point
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20220326_AMF_jackknife_results_envOnly.csv') %>% 
  mutate(ID = floor(V1/10)) %>% #add pseudo-ID per pixel
  group_by(ID) %>% 
  mutate(across(c(predicted, AMF_diversity), ~ mean(.x, na.rm = TRUE))) %>% #change to summarise if only one record per sample is needed
  mutate(abs_residual = abs(AMF_diversity - predicted))

# Quick plot
df %>% 
  ggplot(aes(x = AMF_diversity, y = predicted)) +
  geom_point() +
  geom_smooth(formula = 'y~x', method = 'lm') +
  geom_abline() +
  xlim(c(0,180)) + ylim(c(0,180)) +
  xlab("observed")

# EMF
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20220324_ECM_jackknife_results_envOnly.csv') %>% 
  mutate(ID = floor(V1/10)) %>% #add pseudo-ID per pixel
  group_by(ID) %>% 
  mutate(across(c(predicted, ECM_diversity), ~ mean(.x, na.rm = TRUE))) %>% #change to summarise if only one record per sample is needed
  mutate(abs_residual = abs(ECM_diversity - predicted))

# Quick plot
df %>% 
  ggplot(aes(x = ECM_diversity, y = predicted)) +
  geom_point() +
  geom_smooth(formula = 'y~x', method = 'lm') +
  geom_abline() +
  xlim(c(0,500)) + ylim(c(0,500)) +
  xlab('observed')
