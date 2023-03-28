library(data.table)
library(tidyverse)

df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/arbuscular_mycorrhizal_richness_training_data.csv') %>% 
  mutate(Guild = 'Arbuscular Mycorrhiza') %>% 
  select(Pixel_Lat, Pixel_Long, Guild) %>% 
  rbind(., fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/ectomycorrhizal_richness_training_data.csv') %>% 
          mutate(Guild = 'Ectomycorrhiza') %>% 
  select(Pixel_Lat, Pixel_Long, Guild))

ggplot() +
  geom_polygon(data = map_data("world"), 
               aes(x = long, y = lat, group = group),
               fill = "#bababa",
               color = NA,
               size = 0.1) + 
  coord_fixed(1.1) +
  geom_point(data = df, 
             aes(x = Pixel_Long, y = Pixel_Lat, fill = Guild),
             color = "black",
             pch = 21
  ) +
  # scale_fill_gradientn(colors = brewer.pal(8, "YlOrRd"),
  #                      limits = c(0, 300),
  #                      oob = scales::squish,
  #                      name = "myco_diveristy") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.box="horizontal",
        panel.grid = element_blank(),
        axis.title=element_blank(),
        axis.text=element_blank()) +
  facet_wrap(vars(Guild))
