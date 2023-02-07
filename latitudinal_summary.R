library(data.table)
library(tidyverse)

setwd('/Users/johanvandenhoogen/SPUN/richness_maps')

df <- fread("/Users/johanvandenhoogen/SPUN/richness_maps/output/20230207_ectomycorrhizal_latitude_summary.csv") %>% 
  mutate(Guild = 'Ectomycorrhizal') %>% 
  rbind(fread("/Users/johanvandenhoogen/SPUN/richness_maps/output/20230207_arbuscular_mycorrhizal_latitude_summary.csv") %>% 
          mutate(Guild = 'Arbuscular Mycorrhizal'))


df %>% 
  na.omit() %>% 
  ggplot(aes(x = latitude, y = mean)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill = Guild), alpha = 0.2) +
  geom_line(aes(color = Guild)) +
  coord_flip() +
  ylab("Richness") + xlab("Latitude") +
  theme_classic()



df <- fread("/Users/johanvandenhoogen/SPUN/richness_maps/output/20230207_ectomycorrhizal_latitude_summary.csv") %>% 
  mutate(Group = 'Ectomycorrhizal Fungi') %>% 
  rbind(fread("/Users/johanvandenhoogen/SPUN/richness_maps/output/20230207_arbuscular_mycorrhizal_latitude_summary.csv") %>% 
          mutate(Group = 'Arbuscular Mycorrhizal Fungi')) %>% 
  rbind(fread("/Users/johanvandenhoogen/SPUN/richness_maps/output/20230207_Sabatini2022_VegetationRichness_latitude_summary.csv") %>% 
          mutate(Group = 'Plants')) %>% 
rbind(fread("/Users/johanvandenhoogen/SPUN/richness_maps/output/20230207_Jenkins2013_MammalsRichness_latitude_summary.csv") %>% 
        mutate(Group = 'Mammals'))

df %>% 
  na.omit() %>% 
  ggplot(aes(x = latitude, y = mean)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill = Group), alpha = 0.2) +
  geom_line(aes(color = Group)) +
  coord_flip() +
  ylab("Richness") + xlab("Latitude") +
  theme_classic() +
  scale_fill_manual(values = c("Arbuscular Mycorrhizal Fungi" = "#F8766D", "Ectomycorrhizal Fungi" = "#619CFF", "Plants" = "#00BA38", "Mammals" = "#b06323")) +
  scale_color_manual(values = c("Arbuscular Mycorrhizal Fungi" = "#F8766D", "Ectomycorrhizal Fungi" = "#619CFF", "Plants" = "#00BA38", "Mammals" = "#b06323"))
