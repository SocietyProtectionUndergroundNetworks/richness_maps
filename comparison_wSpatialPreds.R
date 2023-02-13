library(data.table)
library(patchwork)
library(tidyverse)

# Predicted observed data
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230213_arbuscular_mycorrhizal_richness_pred_obs.csv') %>% 
  mutate(version = 'Regular approach') %>% 
  rbind(., fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230213_arbuscular_mycorrhizal_richness_pred_obs_wSpatialPreds.csv') %>% 
          mutate(version = 'With spatial predictors'), fill = T)

df$dens <- col2rgb(densCols(df[['arbuscular_mycorrhizal_richness']], df[['arbuscular_mycorrhizal_richness_Predicted']]))[1,] + 1L

paletteForUse <- c('#d10000', '#ff6622', '#ffda21', '#33dd00', '#1133cc', '#220066', '#330044')
colors <-  colorRampPalette(paletteForUse)(256)

# Map densities to colors
df$colors = colors[df$dens]

df %>% ggplot(aes(x = arbuscular_mycorrhizal_richness, y = arbuscular_mycorrhizal_richness_Predicted)) +
  geom_point(color = df$colors) +
  geom_smooth(method = 'lm', formula = y ~ x, se = FALSE, linetype = 'dashed', color = 'black', size = 0.5) + 
  geom_abline(size = 0.5) +
  # geom_point(alpha = 0.2) +
  theme_minimal() +
  theme(aspect.ratio = 1,
        # panel.grid = element_blank(),
        panel.border = element_rect(fill = NA),
        plot.title = element_text(hjust = 0.5)) +
  # theme(axis.title = element_blank()) +
  facet_wrap(vars(version), scales = 'free') +
  xlab("Observed") + ylab("Predicted") 

# R2 comparison
fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230213_arbuscular_mycorrhizal_richness_grid_search_results_wSpatialPreds.csv') %>% 
  mutate(version = 'Regular approach') %>% 
  rbind(., fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230213_arbuscular_mycorrhizal_richness_grid_search_results.csv') %>% 
          mutate(version = 'With spatial predictors'), fill = T) %>% 
  group_by(version) %>% 
  top_n(10, Mean_R2) %>% 
  summarise(Mean_R2 = mean(Mean_R2), Mean_RMSE = mean(Mean_RMSE), Mean_MAE = mean(Mean_MAE))


# Feature importance
p1 <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230213_arbuscular_mycorrhizal_richness_featureImportances.csv') %>% 
  mutate(version = 'Regular approach') %>% 
  top_n(10, Feature_Importance) %>% 
  ggplot(aes(x = reorder(Variable, -Feature_Importance), y = Feature_Importance)) +
  geom_bar(stat = "identity") +
  facet_wrap(vars(version), scales = 'free', ncol = 1) +
  # scale_x_discrete(limits = Variable) +
  theme(axis.text.x=element_text(angle=60, hjust=1)) 

p2 <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230213_arbuscular_mycorrhizal_richness_featureImportances_wSpatialPreds.csv') %>% 
  mutate(version = 'With spatial predictors') %>% 
  top_n(10, Feature_Importance) %>% 
  ggplot(aes(x = reorder(Variable, -Feature_Importance), y = Feature_Importance)) +
  geom_bar(stat = "identity") +
  facet_wrap(vars(version), scales = 'free', ncol = 1) +
  # scale_x_discrete(limits = Variable) +
  theme(axis.text.x=element_text(angle=60, hjust=1)) 

p1 + p2




