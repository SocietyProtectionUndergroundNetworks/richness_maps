library(data.table)
library(tidyverse)
library(RColorBrewer)

setwd('/Users/johanvandenhoogen/SPUN/richness_maps')

# Define palette
paletteForUse <- c('#d10000', '#ff6622', '#ffda21', '#33dd00', '#1133cc', '#220066', '#330044')
colors <-  colorRampPalette(paletteForUse)(256)

df <- fread('data/20230323_AM_predObs_sampled.csv')
df

df$dens <- col2rgb(densCols(df[['arbuscular_mycorrhizal_richness']], df[['arbuscular_mycorrhizal_richness_Ensemble_mean']]))[1,] + 1L

# Map densities to colors
df$colors = colors[df$dens]

df %>% 
  ggplot(aes(x = arbuscular_mycorrhizal_richness, y = arbuscular_mycorrhizal_richness_Ensemble_mean)) +
  geom_point(color = df$colors) +
  scale_color_manual(values = rev(brewer.pal(6, "Paired"))) +
  geom_abline(linetype = 2) +
  geom_smooth(method = 'lm', formula = 'y ~ x', color = 'black', lwd = 0.5, se = F) +
  # scale_x_log10() + scale_y_log10() +
  theme_classic() +
  theme(legend.position = "none",
        aspect.ratio = 1) +
  labs(y = "Predicted AMF Richness", x = "Observed AMF Richness")


df <- fread('data/20230323_EM_predObs_sampled.csv') 

df$dens <- col2rgb(densCols(df[['ectomycorrhizal_richness']], df[['ectomycorrhizal_richness_Ensemble_mean']]))[1,] + 1L

# Map densities to colors
df$colors = colors[df$dens]

df %>% 
  ggplot(aes(x = ectomycorrhizal_richness, y = ectomycorrhizal_richness_Ensemble_mean)) +
  geom_point(color = df$colors) +
  scale_color_manual(values = rev(brewer.pal(6, "Paired"))) +
  geom_abline(linetype = 2) +
  geom_smooth(method = 'lm', formula = 'y ~ x', color = 'black', lwd = 0.5, se = F) +
  xlim(c(0, 600)) +
  ylim(c(0, 600)) +
   # scale_x_log10() + scale_y_log10() +
  theme_classic() +
  theme(legend.position = "none",
        aspect.ratio = 1) +
  labs(y = "Predicted EMF Richness", x = "Observed EMF Richness")


