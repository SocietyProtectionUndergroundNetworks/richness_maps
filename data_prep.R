library(data.table)
library(tidyverse)

# Full dataset
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20211026_ECM_diversity_data_sampled.csv')

# primers to be removed
primers_to_remove <-  c("58A2F/ITS4",
                        "gITS7/ITS4ngs",
                        "ITS1F_KYO2/LR3 and ITS3_KYO2/LR_KYO1b",
                        "ITS1F/ITS2",
                        "ITS1F/ITS3",
                        "ITS1F/ITS4 and fITS7/ITS4",
                        "ITS1F/ITS4 and ITS3/ITS4",
                        "ITS3/ITS4-OF and ITS86F/ITS4",
                        "ITS4_Fun/5.8S_Fun",
                        "ITS5/ITS4",
                        "ITS7o/ITS4",
                        "ITS9/ITS4")

# Define custom operator
`%notin%` <- Negate(`%in%`)

# Outlier sample IDs
outliers <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/GF_outlier_sampleIDs.csv', header = T) %>% 
  pull(sample_ID)

# Clean data
cleaned_data <- df %>% 
  filter(target_marker != "ITS1") %>% 
  filter(primers %notin% primers_to_remove) %>%
  filter(sample_id %notin% outliers)

# Write to file
fwrite(cleaned_data, '/Users/johanvandenhoogen/SPUN/richness_maps/data/20220805_ECM_diversity_data_sampled_cleaned.csv')

# Sample.type: soil
# target_marker: Illumina
# target_marker: ITS2
# Primers: ITS3/ITS4

# For mapping pipeline: get sample ID of sample withreference categories
ref_ids <- cleaned_data %>% 
  filter(sample_type == 'soil') %>% 
  filter(sequencing_platform == 'Illumina') %>% 
  filter(target_marker == 'ITS2') %>% 
  filter(primers == 'ITS3/ITS4') %>% 
  pull(sample_id)

ref_ids
