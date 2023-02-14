library(data.table)
library(scales)
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

# version                 Mean_R2 Mean_RMSE Mean_MAE
# 1 Regular approach          0.705      11.8     8.40
# 2 With spatial predictors   0.701      11.8     8.39


# Feature importance
p1 <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230213_arbuscular_mycorrhizal_richness_featureImportances.csv') %>% 
  mutate(version = 'Regular approach') %>% 
  top_n(10, Feature_Importance) %>% 
  ggplot(aes(x = reorder(Variable, -Feature_Importance), y = Feature_Importance)) +
  geom_bar(stat = "identity") +
  facet_wrap(vars(version), scales = 'free', ncol = 1) +
  # scale_x_discrete(limits = Variable) +
  theme(axis.text.x=element_text(angle=60, hjust=1))  + xlab("")

p2 <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230213_arbuscular_mycorrhizal_richness_featureImportances_wSpatialPreds.csv') %>% 
  mutate(version = 'With spatial predictors') %>% 
  top_n(10, Feature_Importance) %>% 
  ggplot(aes(x = reorder(Variable, -Feature_Importance), y = Feature_Importance)) +
  geom_bar(stat = "identity") +
  facet_wrap(vars(version), scales = 'free', ncol = 1) +
  # scale_x_discrete(limits = Variable) +
  theme(axis.text.x=element_text(angle=60, hjust=1))  + xlab("")

p1 + p2




# Compare predictions

# Predicted observed data
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230213_arbuscular_mycorrhizal_richness_pred_obs.csv') %>% 
  select(sample_id, arbuscular_mycorrhizal_richness_Predicted) %>% 
  rename(arbuscular_mycorrhizal_richness_Predicted_Regular = arbuscular_mycorrhizal_richness_Predicted) %>% 
  left_join(., fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230213_arbuscular_mycorrhizal_richness_pred_obs_wSpatialPreds.csv') %>% 
              select(sample_id, arbuscular_mycorrhizal_richness_Predicted) %>% 
              rename(arbuscular_mycorrhizal_richness_Predicted_wSpatial = arbuscular_mycorrhizal_richness_Predicted),
            by = 'sample_id') %>% 
  na.omit()

R2val <- round(1 - (sum((df$arbuscular_mycorrhizal_richness_Predicted_Regular-df$arbuscular_mycorrhizal_richness_Predicted_wSpatial )^2)/sum((df$arbuscular_mycorrhizal_richness_Predicted_Regular-mean(df$arbuscular_mycorrhizal_richness_Predicted_Regular))^2)),3)

df %>% 
  ggplot(aes(x = arbuscular_mycorrhizal_richness_Predicted_Regular, y = arbuscular_mycorrhizal_richness_Predicted_wSpatial)) +
  geom_point() +  
  geom_smooth(method = 'lm', formula = y ~ x, se = FALSE, linetype = 'dashed', color = 'black', size = 0.5) + 
  geom_abline(size = 0.5) +
  theme_minimal() +
  theme(aspect.ratio = 1,
        # panel.grid = element_blank(),
        panel.border = element_rect(fill = NA),
        plot.title = element_text(hjust = 0.5)) +
  xlab("Regular approach") + ylab("With spatial predictors") +
  annotate("text", x = 80, y = 100, label= paste0("R2 = ", R2val))


# Random points sampled from maps
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20230214_random_pionts_sampled_AM.csv') %>% 
  mutate(arbuscular_mycorrhizal_richness_predicted = scales::rescale(arbuscular_mycorrhizal_richness_predicted, to = c(0, 1))) %>% 
  mutate(arbuscular_mycorrhizal_richness_predicted_wSpatialPreds = scales::rescale(arbuscular_mycorrhizal_richness_predicted_wSpatialPreds, to = c(0, 1)))

df$dens <- col2rgb(densCols(df[['arbuscular_mycorrhizal_richness_predicted']], df[['arbuscular_mycorrhizal_richness_predicted_wSpatialPreds']]))[1,] + 1L

paletteForUse <- c('#d10000', '#ff6622', '#ffda21', '#33dd00', '#1133cc', '#220066', '#330044')
colors <-  colorRampPalette(paletteForUse)(256)

# Map densities to colors
df$colors = colors[df$dens]

df %>% 
  ggplot(aes(x = arbuscular_mycorrhizal_richness_predicted, y = arbuscular_mycorrhizal_richness_predicted_wSpatialPreds)) +
  geom_point(color = df$colors) +  
  geom_smooth(method = 'lm', formula = y ~ x, se = FALSE, linetype = 'dashed', color = 'black', size = 0.5) + 
  geom_abline(size = 0.5) +
  theme_minimal() +
  theme(aspect.ratio = 1,
        # panel.grid = element_blank(),
        panel.border = element_rect(fill = NA),
        plot.title = element_text(hjust = 0.5)) +
  xlab("Regular approach") + ylab("With spatial predictors")
