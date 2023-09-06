library(data.table)
library(tidyverse)

df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/arbuscular_mycorrhizal_sloo_cv_results_wExtrapolation.csv') %>% 
  rename(r2 = R2_val) %>% 
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

df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/ectomycorrhizal_sloo_cv_results_wExtrapolation.csv') %>% 
  rename(r2 = R2_val) %>% 
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

df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/arbuscular_mycorrhizal_sloo_cv_results_wExtrapolation.csv') %>% 
  rename(r2 = R2_val) %>% 
  group_by(buffer_size) %>% 
  summarise(lower = min(r2), upper = max(r2), mean = mean(r2)) %>% 
  mutate(Guild = 'Arbuscular Mycorrhizal') %>% 
  rbind(., fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/ectomycorrhizal_sloo_cv_results_wExtrapolation.csv') %>% 
  rename(r2 = R2_val) %>% 
          group_by(buffer_size) %>% 
          summarise(lower = min(r2), upper = max(r2), mean = mean(r2)) %>% 
          mutate(Guild = 'Ectomycorrhizal'))


library(ggpmisc)

eq_fmt <- "`y`~`=`~%.3g~italic(e)^{%.3g~`x`}"
eq_fmt <- "`y`~`=`~`%.3g + %.3g log(x)`"

plot <- df %>% 
  ggplot(aes(x = buffer_size/1000, y = mean, group = Guild)) +
  # geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
  # geom_line(aes(color = Guild)) +
  ylab("Coefficient of Determination R2") + xlab("Buffer Size (km)") +
  theme_classic() +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank()
  ) +
  ylim(c(0, 0.6)) +
  geom_hline(aes(yintercept = 0), linetype = 2)  +
  # geom_point() +
  geom_smooth(method = 'lm',
              formula = 'y ~ log(x)',
              aes(color = Guild),
              se = FALSE) +
  stat_poly_eq(mapping = aes(x = buffer_size/1000, y = mean, group = Guild,
                             label = sprintf(eq_fmt,
                                             after_stat(b_0),
                                             after_stat(b_1))),
               formula =  y ~ log(x),
               output.type = "numeric",
               parse = TRUE
  ) 
plot







