library(data.table)
library(tidyverse)

# Raw results have 10 predictions per sample; one for each model in the ensemble. Process by taking the mean for each
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20220308_AMF_jackknife_results.csv') %>% 
  mutate(ID = floor(V1/10)) %>% 
  group_by(ID) %>% 
  mutate(across(c(predicted, AMF_diversity), ~ mean(.x, na.rm = TRUE))) %>% #change to summarise if only one record per sample is needed
  mutate(abs_residual = abs(AMF_diversity - predicted))
