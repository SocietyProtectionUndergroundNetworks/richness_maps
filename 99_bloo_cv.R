library(data.table)
library(tidyverse)

df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/SLOO_CV_AM.csv') %>% 
  group_by(buffer_size) %>% 
  summarise(lower = min(r2), upper = max(r2), mean = mean(r2))

df %>% 
  ggplot(aes(x = buffer_size/1000, y = mean)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
  geom_line() +
  ylab("Coefficient of Determination R2") + xlab("Buffer Size (km)") +
  theme_classic() +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank()
  ) +
  ylim(c(0, 0.6)) +
  geom_hline(aes(yintercept = 0), linetype = 2)


library(data.table)
library(tidyverse)

df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/SLOO_CV_EM.csv') %>% 
  group_by(buffer_size) %>% 
  summarise(lower = min(r2), upper = max(r2), mean = mean(r2))

df %>% 
  ggplot(aes(x = buffer_size/1000, y = mean)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
  geom_line() +
  ylab("Coefficient of Determination R2") + xlab("Buffer Size (km)") +
  theme_classic() +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank()
  ) +
  ylim(c(0, 0.6)) +
  geom_hline(aes(yintercept = 0), linetype = 2)





library(data.table)
library(tidyverse)

df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/SLOO_CV_AM.csv') %>% 
  group_by(buffer_size) %>% 
  summarise(lower = min(r2), upper = max(r2), mean = mean(r2)) %>% 
  mutate(Guild = 'Arbuscular Mycorrhizal') %>% 
  rbind(., fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/SLOO_CV_EM.csv') %>% 
          group_by(buffer_size) %>% 
          summarise(lower = min(r2), upper = max(r2), mean = mean(r2)) %>% 
          mutate(Guild = 'Ectomycorrhizal'))

plot <- df %>% 
  ggplot(aes(x = buffer_size/1000, y = mean)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, group = Guild), alpha = 0.2) +
  geom_line(aes(color = Guild)) +
  ylab("Coefficient of Determination R2") + xlab("Buffer Size (km)") +
  theme_classic() +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank()
  ) +
  ylim(c(0, 0.6)) +
  geom_hline(aes(yintercept = 0), linetype = 2) 
plot
ggsave(plot = plot, "bloo_cv.png")
