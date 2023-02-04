library(data.table)
library(tidyverse)

`%notin%` <- Negate(`%in%`)

# Load data, rename biome names
df <- fread("/Users/johanvandenhoogen/SPUN/richness_maps/data/20230203_GFv4_EM_richness_rarefied_sampled.csv") %>% 
  mutate(Resolve_Biome = as.integer(Resolve_Biome)) 
# %>%
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

summary_woBissetYan <- df %>% 
  left_join(metadata, by = c('sample_id' = 'sample_ID')) %>% 
  filter(paper_id %notin% c('Bissett_AAAA_2016', 'Yan_2018_A0B2')) %>% 
  # filter(guild == 'ectomycorrhizal') %>%
  group_by(Resolve_Biome, guild) %>% 
  # summarise(mean = mean(myco_diversity), sd = sd(myco_diversity))
  summarise(n = n(), median = median(rarefied), iqr = IQR(rarefied)) %>% 
  mutate(cutoff = median + 5 * iqr)


# Stats 
dropped_stats <- df %>%
  left_join(summary_woBissetYan, by = c("Resolve_Biome", "guild")) %>%
  # select(n, median, iqr, cutoff, myco_diversity) %>%
  mutate(dropped = rarefied > cutoff) %>%
  group_by(Resolve_Biome, guild) %>%
  summarise(n = mean(n), median = mean(median), iqr = mean(iqr), cutoff = mean(cutoff), n_dropped = sum(dropped))

fwrite(dropped_stats, '/Users/johanvandenhoogen/SPUN/richness_maps/output/20230203_GFv4_outlier_removal_stats.csv')

filtered_data <- df %>% 
  left_join(summary_woBissetYan, by = c("Resolve_Biome", "guild")) %>%
  filter(rarefied <= cutoff) %>% 
  mutate(Resolve_Biome = as.factor(Resolve_Biome))

fwrite(filtered_data, '/Users/johanvandenhoogen/SPUN/richness_maps/data/20230203_GFv4_sampled_outliersRemoved.csv')

filtered_data %>% 
  ggplot(aes(x = guild, y = rarefied)) +
  facet_wrap(vars(Resolve_Biome), scales = "free") +
  geom_jitter(aes(color = Resolve_Biome), alpha = 0.25) +
  geom_boxplot(fill = NA,  outlier.shape = NA) + 
  theme(legend.position = 'none') 

ggplot() +
  geom_polygon(data = map_data("world", region = "Australia"), 
               aes(x = long, y = lat, group = group),
               fill = "#bababa",
               color = NA,
               size = 0.1) + 
  coord_fixed(1.1) +
  geom_point(data = 
               df %>% 
               left_join(summary_woBissetYan, by = c("Resolve_Biome", "guild")) %>%
               left_join(metadata, by = c('sample_id' = 'sample_ID')) %>% 
               filter(guild == 'ectomycorrhizal') %>% 
               filter(myco_diversity <= cutoff) %>%
               filter(country == 'Australia') %>% 
               arrange(myco_diversity), 
             aes(x = Pixel_Long, y = Pixel_Lat, fill = myco_diversity),
             color = "black",
             pch = 21
  ) +
  scale_fill_gradientn(colors = brewer.pal(8, "YlOrRd"),
                       limits = c(0, 300),
                       oob = scales::squish,
                       name = "myco_diveristy") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.box="horizontal",
        panel.grid = element_blank(),
        axis.title=element_blank(),
        axis.text=element_blank()) +
  guides(fill = guide_colorbar(title.position = "top"))



# 
# 
# 
# # Some stats per biome and guild
# summ_stas <- df %>% 
#   group_by(guild, Resolve_Biome) %>% 
#   summarise(mean = mean(myco_diversity), sd = sd(myco_diversity)) 
# 
# # Plot histograms
# boxplots <- df %>% 
#   left_join(summ_stas) %>%
#   filter(abs(myco_diversity - mean) <= 5 * sd) %>%
#   # filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Boreal Forests' & myco_diversity > 400)) %>%
#   # filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Flooded Grasslands' & myco_diversity > 300)) %>%
#   # filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Mediterranean Forests' & myco_diversity > 100000)) %>%
#   # filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Tropical Grasslands' & myco_diversity > 900)) %>%
#   # filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Tropical Moist Forests' & myco_diversity > 2000)) %>%
#   # filter(!(guild == 'arbuscular_mycorrhizal' & Resolve_Biome == 'Mangroves' & myco_diversity > 30)) %>%
#   # filter(!(guild == 'arbuscular_mycorrhizal' & Resolve_Biome == 'Tundra' & myco_diversity > 40)) %>%
#   # filter(!(guild == 'arbuscular_mycorrhizal' & Resolve_Biome == 'Montane Grasslands' & myco_diversity > 150)) %>%
#   ggplot(aes(x = guild, y = myco_diversity)) +
#   facet_wrap(vars(Resolve_Biome), scales = "free") +
#   geom_jitter(aes(color = Resolve_Biome), alpha = 0.25) +
#   geom_boxplot(fill = NA,  outlier.shape = NA) + 
#   theme(legend.position = 'none') 
# 
# boxplots
# ggsave('/Users/johanvandenhoogen/SPUN/richness_maps/20221212_boxplots_5sdRemoved.pdf', boxplots)
# 
# 
# plot_hists <- df %>% 
#   filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Boreal Forests' & myco_diversity > 400)) %>%
#   filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Flooded Grasslands' & myco_diversity > 300)) %>%
#   filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Mediterranean Forests' & myco_diversity > 100000)) %>%
#   filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Tropical Grasslands' & myco_diversity > 900)) %>%
#   filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Tropical Moist Forests' & myco_diversity > 2000)) %>%
#   filter(!(guild == 'arbuscular_mycorrhizal' & Resolve_Biome == 'Mangroves' & myco_diversity > 30)) %>%
#   filter(!(guild == 'arbuscular_mycorrhizal' & Resolve_Biome == 'Tundra' & myco_diversity > 40)) %>%
#   filter(!(guild == 'arbuscular_mycorrhizal' & Resolve_Biome == 'Montane Grasslands' & myco_diversity > 150)) %>%
#   ggplot(aes(x = myco_diversity)) +
#   facet_wrap(~ guild + Resolve_Biome, scales = "free") +
#   # facet_wrap(vars(Resolve_Biome), scales = "free") +
#   geom_histogram(aes(y=..density.., fill = guild), colour="black", , alpha=0.5, position="dodge", pad = TRUE) +
#   # geom_density(alpha=.2, aes(y=..density.., fill = guild)) +
#   theme(legend.position = 'right')
# plot_hists
# ggsave('/Users/johanvandenhoogen/SPUN/richness_maps/20221212_histograms_outlierRemoved.pdf', plot_hists)
# 
# df %>% 
#   filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Boreal Forests' & myco_diversity > 400)) %>%
#   filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Flooded Grasslands' & myco_diversity > 300)) %>%
#   filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Mediterranean Forests' & myco_diversity > 100000)) %>%
#   filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Tropical Grasslands' & myco_diversity > 900)) %>%
#   filter(!(guild == 'ectomycorrhizal' & Resolve_Biome == 'Tropical Moist Forests' & myco_diversity > 2000)) %>%
#   filter(!(guild == 'arbuscular_mycorrhizal' & Resolve_Biome == 'Mangroves' & myco_diversity > 30)) %>%
#   filter(!(guild == 'arbuscular_mycorrhizal' & Resolve_Biome == 'Tundra' & myco_diversity > 40)) %>%
#   filter(!(guild == 'arbuscular_mycorrhizal' & Resolve_Biome == 'Montane Grasslands' & myco_diversity > 150)) %>%
#   filter(guild == 'arbuscular_mycorrhizal') %>%
#   filter(str_detect(Resolve_Biome, "^Tropical")) %>% 
#   ggplot(aes(x = myco_diversity)) +
#   facet_wrap(~ guild + Resolve_Biome, scales = "free") +
#   # facet_wrap(vars(Resolve_Biome), scales = "free") +
#   geom_histogram(aes(y=..density.., fill = guild), colour="black", , alpha=0.5, position="dodge", pad = TRUE) +
#   # geom_density(alpha=.2, aes(y=..density.., fill = guild)) +
#   theme(legend.position = 'none')
# 
# 
# 
# 
# df %>% filter(is.na(Resolve_Biome)) %>% 
#   select(sample_id, Pixel_Lat, Pixel_Long) %>% 
#   filter(!is.na(Pixel_Lat)) %>% 
#   rename(latitude = Pixel_Lat, longitude = Pixel_Long) %>% 
#   fwrite('/Users/johanvandenhoogen/SPUN/richness_maps/data/20221212_NA_Biome.csv')
# 
# tmp <- df %>% filter(Resolve_Biome == 'Tropical Moist Forests') %>% select(myco_diversity)
# 
# ggsave('/Users/johanvandenhoogen/SPUN/richness_maps/rarefied_per_biome.pdf', plot = plot)
# 
# # Remove rows where the value is more than 10 SD from the mean
# df %>% 
#   left_join(summ_stas) %>%
#   filter(abs(myco_diversity - mean) <= 5 * sd) %>% 
#   ggplot(aes(x = guild, y = myco_diversity)) +
#   facet_wrap(vars(Resolve_Biome), scales = "free") +
#   geom_jitter(aes(color = Resolve_Biome), alpha = 0.1) +
#   geom_boxplot(fill = NA, aes(color = Resolve_Biome), outlier.shape = NA) + 
#   theme(legend.position = 'none')
# 
# # df %>% filter(is.na(Resolve_Biome)) %>% fwrite('/Users/johanvandenhoogen/SPUN/richness_maps/data/20221210_GFv4_NA_afterGapFill.csv')
# 
# 
# library(rnaturalearth)
# library(RColorBrewer)
# 
# tmp <- df %>% filter(Resolve_Biome == 'Mediterranean Forests' & guild == 'ectomycorrhizal') %>% 
#   arrange(desc(myco_diversity)) %>% 
#   select(sample_id, Pixel_Lat, Pixel_Long, myco_diversity, Resolve_Biome, guild) 
# 
# metadata <- fread('/Users/johanvandenhoogen/SPUN/richness_pipeline/data/REL4_Colin_datasets_samples_metadata.csv')
# 
# metadata %>% filter(paper_id == 'Bissett_AAAA_2016')
# 
# tmp <- df %>% filter(Resolve_Biome %in% c('Deserts', 'Mediterranean Forests') & guild == 'ectomycorrhizal') %>% 
#   left_join(metadata, by = c('sample_id' = 'sample_ID')) %>% 
#   filter(paper_id %in% c('Bissett_AAAA_2016', 'Yan_2018_A0B2')) %>% 
#   arrange(desc(myco_diversity)) %>% 
#   select(sample_id, paper_id, myco_diversity, Resolve_Biome, guild, Pixel_Lat, Pixel_Long, country) 
# tmp
# 
# 
# 
# 
# 
# 
# 
# 
# tmp2 <- tmp %>% 
#   group_by(paper_id, Resolve_Biome) %>% 
#   summarise(mean = mean(myco_diversity), sd = sd(myco_diversity)) %>% 
#   arrange(desc(mean))
# 
# fwrite(tmp, "/Users/johanvandenhoogen/SPUN/richness_maps/data/20221213_high_med_samples.csv")
# 
# 
# ggplot() +
#   geom_polygon(data = map_data("world", region = "Australia"), 
#                aes(x = long, y = lat, group = group),
#                fill = "#bababa",
#                color = NA,
#                size = 0.1) + 
#   coord_fixed(1.1) +
#   geom_point(data = tmp %>% filter(myco_diversity > 500), 
#              aes(x = Pixel_Long, y = Pixel_Lat, fill = myco_diversity),
#              color = "black",
#              pch = 21
#   ) +
#   scale_fill_gradientn(colors = brewer.pal(8, "YlOrRd"),
#                        limits = c(0, 500),
#                        oob = scales::squish,
#                        name = "myco_diveristy") +
#   theme_minimal() +
#   theme(legend.position = "bottom",
#         legend.box="horizontal",
#         panel.grid = element_blank(),
#         axis.title=element_blank(),
#         axis.text=element_blank()) +
#   guides(fill = guide_colorbar(title.position = "top"))
# 
# 
# df %>% 
#   filter(myco_diversity > 1000) %>% 
#   ggplot(aes(x = guild, y = myco_diversity)) +
#   facet_wrap(vars(Resolve_Biome), scales = "free") +
#   geom_jitter(aes(color = Resolve_Biome), alpha = 0.25) +
#   geom_boxplot(fill = NA,  outlier.shape = NA) + 
#   theme(legend.position = 'none') 
# 
# 
# 
# ggplot() +
#   geom_polygon(data = map_data("world", region = "Australia"), 
#                aes(x = long, y = lat, group = group),
#                fill = "#bababa",
#                color = NA,
#                size = 0.1) + 
#   coord_fixed(1.1) +
#   geom_point(data = df %>% #filter(Resolve_Biome == 'Mediterranean Forests' & guild == 'ectomycorrhizal') %>% 
#                left_join(metadata, by = c('sample_id' = 'sample_ID')) %>% 
#                arrange(desc(myco_diversity)) %>% 
#                select(sample_id, paper_id, myco_diversity, Resolve_Biome, guild, Pixel_Lat, Pixel_Long, country) %>% 
#                filter(paper_id %notin% c('Bissett_AAAA_2016', 'Yan_2018_A0B2')) %>%
#                filter(country == 'Australia') %>% 
#                arrange((myco_diversity)), 
#              aes(x = Pixel_Long, y = Pixel_Lat, fill = myco_diversity),
#              color = "black",
#              pch = 21
#   ) +
#   scale_fill_gradientn(colors = brewer.pal(8, "YlOrRd"),
#                        limits = c(0, 100),
#                        oob = scales::squish,
#                        name = "without c('Bissett_AAAA_2016', 'Yan_2018_A0B2')") +
#   theme_minimal() +
#   theme(legend.position = "bottom",
#         legend.box="horizontal",
#         panel.grid = element_blank(),
#         axis.title=element_blank(),
#         axis.text=element_blank()) +
#   guides(fill = guide_colorbar(title.position = "top"))
# 
# 
# tmp %>% 
#   filter(paper_id %in% c('Bissett_AAAA_2016')) %>%
#   filter(country == 'Australia') %>% 
#   arrange((myco_diversity)) %>% 
#   group_by(Resolve_Biome) %>% 
#   summarise(mean = mean(myco_diversity))
# 
# 
# summ <- df %>% filter(guild == 'ectomycorrhizal') %>% 
#   left_join(metadata, by = c('sample_id' = 'sample_ID')) %>% 
#   group_by(paper_id, Resolve_Biome) %>% 
#   summarise(mean = mean(myco_diversity)) %>% 
#   arrange(Resolve_Biome, paper_id)
# 
# fwrite(summ, "/Users/johanvandenhoogen/SPUN/richness_maps/data/20221213_mean_perPaper.csv")
# 
# 
# 
# 
# ################################################################################################
# ################################################################################################
# ################################################################################################
# ################################################################################################
# 
# 
# 
# 
