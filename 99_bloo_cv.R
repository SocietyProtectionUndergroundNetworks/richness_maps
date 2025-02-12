library(data.table)
library(ggpmisc)
library(tidyverse)
setwd('/Users/johanvandenhoogen/SPUN/richness_maps')

df <- fread('output/20250205_arbuscular_mycorrhizal_SLOO_CV.csv') %>% 
  group_by(buffer_size) %>% 
  summarise(lower = min(r2), upper = max(r2), mean = mean(r2)) %>% 
  mutate(Guild = 'Arbuscular Mycorrhizal') %>% 
  rbind(., fread('output/20250212_ectomycorrhizal_SLOO_CV_v2.csv') %>% 
          group_by(buffer_size) %>% 
          summarise(lower = min(r2), upper = max(r2), mean = mean(r2)) %>% 
          mutate(Guild = 'Ectomycorrhizal'))

# eq_fmt <- "`y`~`=`~%.3g~italic(e)^{%.3g~`x`}"
eq_fmt <- "`y`~`=`~`%.3g + %.3g log(x)`"

df %>% 
  ggplot(aes(x = buffer_size/1000, y = mean, group = Guild)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill = Guild), alpha = 0.1) +
  geom_line(aes(color = Guild)) +
  ylab("Coefficient of Determination R2") + xlab("Buffer Size (km)") +
  theme_classic() +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank()
  ) +
  ylim(c(0, 0.6)) +
  geom_hline(aes(yintercept = 0), linetype = 2)  +
  # geom_point(aes(color = Guild)) +
  geom_smooth(method = 'lm',
              formula = 'y ~ log(x)',
              aes(color = Guild),
              se = FALSE,
              size = 0.75,
              linetype = 'dashed') +
  stat_poly_eq(mapping = aes(x = buffer_size/1000, y = mean, group = Guild,
                             label = sprintf(eq_fmt,
                                             after_stat(b_0),
                                             after_stat(b_1))),
               formula =  y ~ log(x),
               output.type = "numeric",
               parse = TRUE
  ) 

ggsave('figures/20250212_mycorrhizal_SLOO_CV.png', width = 6, height = 4, dpi = 300)
