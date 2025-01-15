library(data.table)
library(caret)
library(janitor)
library(tidyverse)

projectVars <- c('sequencing_platform',
                 'sample_type',
                 'primers')

# Load data
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20250115_AMF_sampled_outliersRemoved.csv') %>% 
  select(all_of(projectVars), sample_id)

# Create dummy variables
dummy <- dummyVars(" ~ .", data = df %>% select(-sample_id))

# Onehot encoding and add sample_id columnn
df_onehot <- data.frame(predict(dummy, newdata = df)) %>% 
  cbind(df %>% select(sample_id)) 

# Fix column names
names(df_onehot) <- gsub('[.]', '_', names(df_onehot))

# Print variable names
cat(names(df_onehot),sep="\n")

# Print reference levels
df_onehot %>% filter(sample_id == 'S1002')

# Merge with original data, fix NA_ values 
df_final <- df_onehot %>% left_join(., fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20250115_AMF_sampled_outliersRemoved.csv') %>% 
                                      select(-all_of(projectVars)), by = 'sample_id') %>% 
  # replace "NA_" with NA 
  mutate(across(where(is.character), ~na_if(., 'NA_'))) %>% 
  mutate(across(where(is.character), as.factor)) %>% 
  # replace "0.25 to 0.28" with "0.265" in extraction_dna_mass
  mutate(extraction_dna_mass = ifelse(extraction_dna_mass == '0.25 to 0.28', 0.265, extraction_dna_mass))

# Write to file
fwrite(df_final, '/Users/johanvandenhoogen/SPUN/richness_maps/data/20250115_AMF_sampled_outliersRemoved_onehot.csv')
