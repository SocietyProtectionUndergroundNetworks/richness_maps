rm(list=ls())
library(RColorBrewer)
library(data.table)
library(tidyverse)

`%notin%` <- Negate(`%in%`)

# ECM

# Load data, rename biome names when writing per-biome summary file. Uncomment to retain integers (necessary for mapping)
df <- fread("/Users/johanvandenhoogen/SPUN/richness_maps/data/20231001_EM_richness_rarefied_sampled.csv") %>% 
  mutate(Resolve_Biome = as.integer(Resolve_Biome)) #%>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 1, "Tropical Moist Forests")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 2, "Tropical Dry Forests")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 3, "Tropical Coniferous Forests")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 4, "Temperate Broadleaf Forests")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 5, "Temperate Conifer Forests")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 6, "Boreal Forests")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 7, "Tropical Grasslands")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 8, "Temperate Grasslands")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 9, "Flooded Grasslands")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 10, "Montane Grasslands")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 11, "Tundra")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 12, "Mediterranean Forests")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 13, "Deserts")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 14, "Mangroves"))

metadata <- fread('/Users/johanvandenhoogen/SPUN/richness_pipeline/data/REL4_Colin_datasets_samples_metadata.csv')

# Filter data:
# 1. Get mean and sd per guild per biome, but exclude Bissett and Yan papers
# 2. Remove 

# Get mean & IQR values
summary_woBissetYan <- df %>% 
  left_join(metadata, by = c('sample_id' = 'sample_ID')) %>% 
  filter(paper_id %notin% c('Bissett_AAAA_2016', 'Yan_2018_A0B2')) %>% 
  group_by(Resolve_Biome) %>% 
  summarise(n = n(), median = median(rarefied), iqr = IQR(rarefied)) %>% 
  mutate(cutoff = median + 5 * iqr)

# Statistics
dropped_stats <- df %>%
  left_join(summary_woBissetYan, by = c("Resolve_Biome")) %>%
  mutate(dropped = rarefied > cutoff) %>%
  group_by(Resolve_Biome) %>%
  summarise(n = mean(n), median = mean(median), iqr = mean(iqr), cutoff = mean(cutoff), n_dropped = sum(dropped))

dropped_points <- df %>%
  left_join(summary_woBissetYan, by = c("Resolve_Biome")) %>%
  filter(rarefied > cutoff)

# Write to file
# fwrite(dropped_stats, '/Users/johanvandenhoogen/SPUN/richness_maps/output/2023101_EM_outlier_removal_stats.csv')

# primers/seq platforms/markers to remove
seq_platforms_toRemove = "DNBSEQ-G400"

primers_toRemove = c("5.8SR/ITS4", 
                     "gITS7/NLC2mod", 
                     "ITS1F_KYO2/LR3 then ITS3_KYO2/LR_KYO1b",
                     "ITS1F/ITS2", 
                     "ITS1F/ITS3", 
                     "ITS3_KYO2/ITS4_KYO3", 
                     "ITS3-Mix1 to 2/ITS4-cwmix1 + ITS4-cwmix2", 
                     "ITS3ngs mix/ITS4ngs",
                     "ITS3ngs1 to 5/ITS4ngs", 
                     "ITS4_Fun/5.8S_Fun", 
                     "ITS5/ITS2",  
                     "ITS5/ITS4", 
                     "ITS7o/ITS4", 
                     "ITS9/ITS4")

target_markers_toRemove = "ITS1"

# Filter data
filtered_data <- df %>% 
  left_join(summary_woBissetYan, by = c("Resolve_Biome")) %>%
  filter(rarefied <= cutoff) %>%
  mutate(Resolve_Biome = as.factor(Resolve_Biome)) %>% 
  filter(sequencing_platform %notin% seq_platforms_toRemove) %>% 
  filter(primers %notin% primers_toRemove) %>% 
  filter(target_gene %notin% target_markers_toRemove)

# Write to file
fwrite(filtered_data, '/Users/johanvandenhoogen/SPUN/richness_maps/data/20231001_EM_sampled_outliersRemoved.csv')

# Per biome boxplots
filtered_data %>% 
  ggplot(aes(x = Resolve_Biome, y = rarefied)) +
  facet_wrap(vars(Resolve_Biome), scales = "free") +
  geom_jitter(aes(color = Resolve_Biome), alpha = 0.25) +
  geom_boxplot(fill = NA,  outlier.shape = NA) + 
  theme(legend.position = 'none') 

# Display data on map
ggplot() +
  geom_polygon(data = map_data("world"), 
               aes(x = long, y = lat, group = group),
               fill = "#bababa",
               color = NA,
               size = 0.1) + 
  coord_fixed(1.1) +
  geom_point(data = filtered_data %>% 
               arrange(rarefied), 
             aes(x = Pixel_Long, y = Pixel_Lat, fill = rarefied),
             color = "black",
             pch = 21
  ) +
  scale_fill_gradientn(colors = brewer.pal(8, "YlOrRd"),
                       limits = c(0, 500),
                       oob = scales::squish,
                       name = "EM Richness Rarefied") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.box="horizontal",
        panel.grid = element_blank(),
        axis.title=element_blank(),
        axis.text=element_blank()) +
  guides(fill = guide_colorbar(title.position = "top"))

# Display only filtered locations on map
filtered <- df %>% filter(sample_id %notin% filtered_data$sample_id) %>% select(sample_id, rarefied, Resolve_Biome)
ggplot() +
  geom_polygon(data = map_data("world"), 
               aes(x = long, y = lat, group = group),
               fill = "#bababa",
               color = NA,
               size = 0.1) + 
  coord_fixed(1.1) +
  geom_point(data = df %>% filter(sample_id %notin% filtered_data$sample_id), 
             aes(x = Pixel_Long, y = Pixel_Lat, fill = rarefied),
             color = "black",
             pch = 21
  ) +
  scale_fill_gradientn(colors = brewer.pal(8, "YlOrRd"),
                       limits = c(0, 500),
                       oob = scales::squish,
                       name = "EM Richness Rarefied (Outliers)") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.box="horizontal",
        panel.grid = element_blank(),
        axis.title=element_blank(),
        axis.text=element_blank()) +
  guides(fill = guide_colorbar(title.position = "top"))

# AM
# Load data, rename biome names when writing per-biome summary file. Uncomment to retain integers (necessary for mapping)
df <- fread("/Users/johanvandenhoogen/SPUN/richness_maps/data/20230206_GFv4_AM_richness_rarefied_sampled_oneHot.csv") %>% 
  mutate(Resolve_Biome = as.integer(Resolve_Biome)) #%>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 1, "Tropical Moist Forests")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 2, "Tropical Dry Forests")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 3, "Tropical Coniferous Forests")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 4, "Temperate Broadleaf Forests")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 5, "Temperate Conifer Forests")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 6, "Boreal Forests")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 7, "Tropical Grasslands")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 8, "Temperate Grasslands")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 9, "Flooded Grasslands")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 10, "Montane Grasslands")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 11, "Tundra")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 12, "Mediterranean Forests")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 13, "Deserts")) %>%
# mutate(Resolve_Biome = replace(Resolve_Biome, Resolve_Biome == 14, "Mangroves"))

metadata <- fread('/Users/johanvandenhoogen/SPUN/richness_pipeline/data/SSU_metadata.csv')

# Filter data:
# 1. Get mean and sd per guild per biome, but exclude Bissett and Yan papers
# 2. Remove 

# Get mean & IQR values
summary <- df %>% 
  left_join(metadata, by = c('sample_id' = 'id')) %>% 
  group_by(Resolve_Biome) %>% 
  summarise(n = n(), median = median(rarefied), iqr = IQR(rarefied)) %>% 
  mutate(cutoff = median + 5 * iqr)

# Statistics
dropped_stats <- df %>%
  left_join(summary, by = c("Resolve_Biome")) %>%
  mutate(dropped = rarefied > cutoff) %>%
  group_by(Resolve_Biome) %>%
  summarise(n = mean(n), median = mean(median), iqr = mean(iqr), cutoff = mean(cutoff), n_dropped = sum(dropped))

# Write to file
fwrite(dropped_stats, '/Users/johanvandenhoogen/SPUN/richness_maps/output/20230508_AM_SSU_outlier_removal_stats.csv')

# Filter data
filtered_data <- df %>% 
  left_join(summary, by = c("Resolve_Biome")) %>%
  filter(rarefied <= cutoff) %>%
  mutate(Resolve_Biome = as.factor(Resolve_Biome)) 

# Write to file
fwrite(filtered_data, '/Users/johanvandenhoogen/SPUN/richness_maps/data/20230508_AM_SSU_sampled_onehot_outliersRemoved.csv')

# Per biome boxplots
filtered_data %>% 
  ggplot(aes(x = Resolve_Biome, y = rarefied)) +
  facet_wrap(vars(Resolve_Biome), scales = "free") +
  geom_jitter(aes(color = Resolve_Biome), alpha = 0.25) +
  geom_boxplot(fill = NA,  outlier.shape = NA) + 
  theme(legend.position = 'none') 

# Display data on map
ggplot() +
  geom_polygon(data = map_data("world"), 
               aes(x = long, y = lat, group = group),
               fill = "#bababa",
               color = NA,
               size = 0.1) + 
  coord_fixed(1.1) +
  geom_point(data = filtered_data %>% 
               arrange(rarefied), 
             aes(x = Pixel_Long, y = Pixel_Lat, fill = rarefied),
             color = "black",
             pch = 21
  ) +
  scale_fill_gradientn(colors = brewer.pal(8, "YlOrRd"),
                       limits = c(0, 100),
                       oob = scales::squish,
                       name = "AM Richness Rarefied") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.box="horizontal",
        panel.grid = element_blank(),
        axis.title=element_blank(),
        axis.text=element_blank()) +
  guides(fill = guide_colorbar(title.position = "top"))

# Display only filtered locations on map
filtered <- df %>% filter(sample_id %notin% filtered_data$sample_id) %>% select(sample_id, rarefied, Resolve_Biome)
ggplot() +
  geom_polygon(data = map_data("world"), 
               aes(x = long, y = lat, group = group),
               fill = "#bababa",
               color = NA,
               size = 0.1) + 
  coord_fixed(1.1) +
  geom_point(data = df %>% filter(sample_id %notin% filtered_data$sample_id), 
             aes(x = Pixel_Long, y = Pixel_Lat, fill = rarefied),
             color = "black",
             pch = 21
  ) +
  scale_fill_gradientn(colors = brewer.pal(8, "YlOrRd"),
                       limits = c(0, 100),
                       oob = scales::squish,
                       name = "AM Richness Rarefied (Outliers)") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.box="horizontal",
        panel.grid = element_blank(),
        axis.title=element_blank(),
        axis.text=element_blank()) +
  guides(fill = guide_colorbar(title.position = "top"))
