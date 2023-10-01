rm(list=ls())
library(data.table)
library(caret)
library(janitor)
library(tidyverse)

projectVars <- c('sequencing_platform',
                 'sample_type',
                 'primers')

# Load data
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20231001_EM_sampled_outliersRemoved.csv') %>% 
  select(all_of(projectVars), sample_id)

# Create dummy variables
dummy <- dummyVars(" ~ .", data = df %>% select(-sample_id))

# Onehot encoding and add sample_id columnn
df_onehot <- data.frame(predict(dummy, newdata = df)) %>% 
  cbind(df %>% select(sample_id)) #%>% 
  # left_join(., fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/arbuscular_mycorrhizal_richness_training_data_wMEMs.csv') %>% select(sample_id, starts_with('MEM')))

# Fix column names
names(df_onehot) <- gsub('[.]', '_', names(df_onehot))

# Print variable names
cat(names(df_onehot),sep="\n")

# Print reference levels
df_onehot %>% filter(sample_id == 'FMS17564v2')

# Combine into one dataframe
df_final <- df_onehot %>% left_join(., fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20231001_EM_sampled_outliersRemoved.csv') %>% 
                                      select(-all_of(projectVars)), by = 'sample_id')

# Write to file
fwrite(df_final, '/Users/johanvandenhoogen/SPUN/richness_maps/data/20231001_EM_richness_rarefied_sampled_oneHot.csv')
