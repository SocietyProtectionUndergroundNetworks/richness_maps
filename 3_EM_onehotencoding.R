rm(list=ls())
library(data.table)
library(caret)
library(janitor)
library(tidyverse)

projectVars <- c('sequencing_platform',
                 'sample_type',
                 'primers')

# Load data
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20250117_ECM_sampled_outliersRemoved.csv') %>% 
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
df_final <- df_onehot %>% left_join(., fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20250117_ECM_sampled_outliersRemoved.csv') %>% 
                                      select(-all_of(projectVars)), by = 'sample_id') %>% 
  # replace "NA_" with NA 
  mutate(across(where(is.character), ~na_if(., 'NA_'))) %>% 
  # Replace ranges with midpoint
  mutate(extraction_dna_mass = ifelse(extraction_dna_mass == '0.05 to 0.2', 0.125, extraction_dna_mass)) %>%
  mutate(extraction_dna_mass = ifelse(extraction_dna_mass == '0.25 to 0.3', 0.275, extraction_dna_mass)) %>%
  mutate(extraction_dna_mass = ifelse(extraction_dna_mass == '0.25 to 0.28', 0.265, extraction_dna_mass)) %>%
  mutate(extraction_dna_mass = ifelse(extraction_dna_mass == '0.25 to 0.30', 0.275, extraction_dna_mass)) %>%
  mutate(extraction_dna_mass = ifelse(extraction_dna_mass == '0.25 to 0.5', 0.375, extraction_dna_mass)) %>%
  mutate(extraction_dna_mass = ifelse(extraction_dna_mass == '0.25 to 0.50', 0.375, extraction_dna_mass)) %>%
  mutate(extraction_dna_mass = ifelse(extraction_dna_mass == '0.3 to 0.75', 0.525, extraction_dna_mass)) %>%
  mutate(extraction_dna_mass = ifelse(extraction_dna_mass == '0.3 to 2', 1.15, extraction_dna_mass)) %>%
  mutate(extraction_dna_mass = ifelse(extraction_dna_mass == '0.4 to 1', 0.8, extraction_dna_mass))

# Write to file
fwrite(df_final, '/Users/johanvandenhoogen/SPUN/richness_maps/data/20250117_ECM_sampled_outliersRemoved_onehot.csv')
