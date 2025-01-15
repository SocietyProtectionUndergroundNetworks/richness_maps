library(data.table)
library(tidyverse)
library(RColorBrewer)

setwd('/Users/johanvandenhoogen/SPUN/richness_maps')

# Define palette
paletteForUse <- c('#d10000', '#ff6622', '#ffda21', '#33dd00', '#1133cc', '#220066', '#330044')
colors <-  colorRampPalette(paletteForUse)(256)

# List csv files in output folder
files <- list.files('/Users/johanvandenhoogen/SPUN/richness_maps/output/pred_obs', pattern = 'pred_obs.csv', full.names = T)

for (file in files){
  df <- fread(file) %>% 
    group_by(sample_id) 
  
  guild <- if (basename(file) %>% str_detect("ecto")) {'ectomycorrhizal'} else{'arbuscular_mycorrhizal'}
  
  sampling_intensity <- basename(file) %>% str_detect('_sampling_density')
  
  varofinterest <- basename(file) %>% str_split('mycorrhizal_') %>% last() %>% last() %>% str_remove("_pred_obs.csv") %>% 
    str_remove("_sampling_density")
  
  varname <- if (varofinterest == 'richness'){
    paste0(guild, "_", varofinterest)
  } else {
    varofinterest
  }
  
  df$dens <- col2rgb(densCols(df[[varname]], df[[paste0(varname, "_Predicted")]]))[1,] + 1L
  
  # Map densities to colors
  df$colors = colors[df$dens]
  
  plot <- df %>% 
    ggplot(aes_string(x = varname, y = paste0(varname, "_Predicted"))) +
    geom_point(color = df$colors) +
    scale_color_manual(values = rev(brewer.pal(6, "Paired"))) +
    geom_abline(linetype = 2) +
    geom_smooth(method = 'lm', formula = 'y ~ x', color = 'black', lwd = 0.5, se = F) +
    # xlim(c(0, 600)) +
    # ylim(c(0, 600)) +
    # scale_x_log10() + scale_y_log10() +
    # coord_fixed() +
    theme_classic() +
    theme(legend.position = "none",
          aspect.ratio = 1) +
    scale_x_continuous(limits = c(min(df[[varname]], df[[paste0(varname, "_Predicted")]]), 
                                  max(df[[varname]], df[[paste0(varname, "_Predicted")]]))) +
    scale_y_continuous(limits = c(min(df[[varname]], df[[paste0(varname, "_Predicted")]]), 
                                  max(df[[varname]], df[[paste0(varname, "_Predicted")]]))) +
    labs(x = paste0(guild, "_", varofinterest), y = paste0(guild, "_", varofinterest, "_Predicted"))
  
  if (sampling_intensity == T){
    ggsave(paste0('/Users/johanvandenhoogen/SPUN/richness_maps/output/pred_obs/', basename(file) %>% str_remove('.csv'), '_sampling_density.png'), plot)
  } else{
    ggsave(paste0('/Users/johanvandenhoogen/SPUN/richness_maps/output/pred_obs/', basename(file) %>% str_remove('.csv'), '.png'), plot)
  }
}
