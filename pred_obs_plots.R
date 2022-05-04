library(data.table)
library(tidyverse)
library(RColorBrewer)

df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/20220404_pred_obs_woAggregation.csv')
df
df %>% 
  group_by(Pixel_Lat, Pixel_Long) %>% 
  # rowwise() %>% 
  # summarise(AMF_diversity = mean(AMF_diversity), AMF_diversity_Predicted = mean(AMF_diversity_Predicted)) %>% 
  ggplot(aes(x = AMF_diversity, y = AMF_diversity_Predicted)) +
  geom_point() +
  xlim(c(0,100)) + ylim(c(0,100)) +
  geom_abline()

df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220411_ECM_diversity_pred_obs.csv')
df
df %>% 
  group_by(Pixel_Lat, Pixel_Long) %>%
  summarise(ECM_diversity = mean(ECM_diversity), ECM_diversity_Predicted = mean(ECM_diversity_Predicted)) %>%
  ggplot(aes(x = ECM_diversity, y = ECM_diversity_Predicted)) +
  geom_point() +
  xlim(c(0,30000)) + ylim(c(0,30000)) +
  geom_abline()


# Define palette
paletteForUse <- c('#d10000', '#ff6622', '#ffda21', '#33dd00', '#1133cc', '#220066', '#330044')
colors <-  colorRampPalette(paletteForUse)(256)

subset$dens <- col2rgb(densCols(subset[[modelledVar]], subset[[predictedVar]]))[1,] + 1L

# Map densities to colors
subset$colors = colors[subset$dens]

# df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220411_ECM_diversity_pred_obs_distictObs_woProjectVars.csv') %>% 
#   mutate(setup = 'distictObs_woProjectVars') %>% 
#   mutate(colors = colors[col2rgb(densCols((.) %>% pull(ECM_diversity), (.) %>% pull(ECM_diversity_Predicted)))[1,] + 1L])
# 


df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220411_ECM_diversity_pred_obs_distictObs_woProjectVars.csv') %>% 
  mutate(setup = 'distictObs_woProjectVars') %>% 
  rbind(fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220411_ECM_diversity_pred_obs_distictObs_wProjectVars.csv') %>% 
          mutate(setup = 'distictObs_wProjectVars')) %>% 
  rbind(fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220411_ECM_diversity_pred_obs_wopixelAgg_woProjectVars.csv') %>% 
          mutate(setup = 'wopixelAgg_woProjectVars')) %>% 
  rbind(fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220411_ECM_diversity_pred_obs_wopixelAgg_wProjectVars.csv') %>% 
          mutate(setup = 'wopixelAgg_wProjectVars')) %>% 
  rbind(fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220411_ECM_diversity_pred_obs_wpixelAgg_woProjectVars.csv') %>% 
          mutate(setup = 'wpixelAgg_woProjectVars')) %>% 
  rbind(fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220411_ECM_diversity_pred_obs_wpixelAgg_wProjectVars.csv') %>% 
          mutate(setup = 'wpixelAgg_wProjectVars')) 

df %>% 
  group_by(Pixel_Lat, Pixel_Long, setup) %>%
  summarise(ECM_diversity = median(ECM_diversity), ECM_diversity_Predicted = median(ECM_diversity_Predicted)) %>%
  # mutate(ECM_diversity_Predicted = ECM_diversity_Predicted + 0.001,
  #        ECM_diversity = ECM_diversity + 0.001) %>% 
  ggplot(aes(x = ECM_diversity, y = ECM_diversity_Predicted)) +
  # stat_smooth_func(geom = "text", method = "lm", hjust = 0, parse = TRUE) +
  geom_point(aes(color = setup)) +
  scale_color_manual(values = rev(brewer.pal(6, "Paired"))) +
  xlim(c(0,30000)) + ylim(c(0,30000)) +
  geom_abline() +
  geom_smooth(method = 'lm', formula = 'y ~ x', color = 'black', lwd = 0.5) +
  facet_wrap(vars(setup)) +
  # scale_x_log10() + scale_y_log10() +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(
    title = "Median per pixel",
    # subtitle = "Summary statistics for NDVI, Npp and PET",
    y = "Predicted EMF Richness", x = "Observed EMF Richness")





df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220414_AMF_diversity_pred_obs_distictObs_woProjectVars.csv') %>% 
  mutate(setup = 'distictObs_woProjectVars') %>% 
  rbind(fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220414_AMF_diversity_pred_obs_distictObs_wProjectVars.csv') %>% 
          mutate(setup = 'distictObs_wProjectVars')) %>% 
  rbind(fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220414_AMF_diversity_pred_obs_wopixelAgg_woProjectVars.csv') %>% 
          mutate(setup = 'wopixelAgg_woProjectVars')) %>% 
  rbind(fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220414_AMF_diversity_pred_obs_wopixelAgg_wProjectVars.csv') %>% 
          mutate(setup = 'wopixelAgg_wProjectVars')) %>% 
  rbind(fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220414_AMF_diversity_pred_obs_wpixelAgg_woProjectVars.csv') %>% 
          mutate(setup = 'wpixelAgg_woProjectVars')) %>% 
  rbind(fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220414_AMF_diversity_pred_obs_wpixelAgg_wProjectVars.csv') %>% 
          mutate(setup = 'wpixelAgg_wProjectVars')) 

df %>% 
  group_by(Pixel_Lat, Pixel_Long, setup) %>%
  summarise(AMF_diversity = mean(AMF_diversity), AMF_diversity_Predicted = mean(AMF_diversity_Predicted)) %>%
  # mutate(AMF_diversity_Predicted = AMF_diversity_Predicted + 0.001,
  #        AMF_diversity = AMF_diversity + 0.001) %>% 
  ggplot(aes(x = AMF_diversity, y = AMF_diversity_Predicted)) +
  # stat_smooth_func(geom = "text", method = "lm", hjust = 0, parse = TRUE) +
  geom_point(aes(color = setup)) +
  scale_color_manual(values = rev(brewer.pal(6, "Paired"))) +
  xlim(c(0,180)) + ylim(c(0,180)) +
  geom_abline() +
  geom_smooth(method = 'lm', formula = 'y ~ x', color = 'black', lwd = 0.5) +
  facet_wrap(vars(setup)) +
  theme_minimal() +
  scale_x_log10() + scale_y_log10() +
  theme(legend.position = "none") +
  labs(
    title = "Nean per pixel",
    # subtitle = "Summary statistics for NDVI, Npp and PET",
    y = "Predicted AMF Richness", x = "Observed AMF Richness")






df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20220425_ECM_diversity_pred_obs_distictObs_wProjectVars_logTransformed.csv') %>% 
  mutate(setup = 'distictObs_wProjectVars') 

df %>% 
  # group_by(Pixel_Lat, Pixel_Long, setup) %>%
  # summarise(ECM_diversity = mean(ECM_diversity), ECM_diversity_Predicted = mean(ECM_diversity_Predicted)) %>%
  ggplot(aes(x = ECM_diversity, y = ECM_diversity_Predicted)) +
  # stat_smooth_func(geom = "text", method = "lm", hjust = 0, parse = TRUE) +
  geom_point(aes(color = setup)) +
  scale_color_manual(values = rev(brewer.pal(6, "Paired"))) +
  xlim(c(0,40000)) + ylim(c(0,40000)) +
  geom_abline() +
  geom_smooth(method = 'lm', formula = 'y ~ x', color = 'black', lwd = 0.5) +
  # facet_wrap(vars(setup)) +
  theme_minimal() +
  scale_x_log10() + scale_y_log10() +
  theme(legend.position = "none") +
  labs(
    title = "Mean per pixel",
    # subtitle = "Summary statistics for NDVI, Npp and PET",
    y = "Predicted EMF Richness", x = "Observed EMF Richness")


hist(log(df$ECM_diversity + 1, base = 10))

# Source of "stat_smooth_func" and "StatSmoothFunc": https://gist.github.com/kdauria/524eade46135f6348140
# Slightly modified 

stat_smooth_func <- function(mapping = NULL, data = NULL,
                             geom = "smooth", position = "identity",
                             ...,
                             method = "auto",
                             formula = y ~ x,
                             se = TRUE,
                             n = 80,
                             span = 0.75,
                             fullrange = FALSE,
                             level = 0.95,
                             method.args = list(),
                             na.rm = FALSE,
                             show.legend = NA,
                             inherit.aes = TRUE,
                             xpos = NULL,
                             ypos = NULL) {
  layer(
    data = data,
    mapping = mapping,
    stat = StatSmoothFunc,
    geom = geom,
    position = position,
    show.legend = show.legend,
    inherit.aes = inherit.aes,
    params = list(
      method = method,
      formula = formula,
      se = se,
      n = n,
      fullrange = fullrange,
      level = level,
      na.rm = na.rm,
      method.args = method.args,
      span = span,
      xpos = xpos,
      ypos = ypos,
      ...
    )
  )
}


StatSmoothFunc <- ggproto("StatSmooth", Stat,
                          
                          setup_params = function(data, params) {
                            # Figure out what type of smoothing to do: loess for small datasets,
                            # gam with a cubic regression basis for large data
                            # This is based on the size of the _largest_ group.
                            if (identical(params$method, "auto")) {
                              max_group <- max(table(data$group))
                              
                              if (max_group < 1000) {
                                params$method <- "loess"
                              } else {
                                params$method <- "gam"
                                params$formula <- y ~ s(x, bs = "cs")
                              }
                            }
                            if (identical(params$method, "gam")) {
                              params$method <- mgcv::gam
                            }
                            
                            params
                          },
                          
                          compute_group = function(data, scales, method = "auto", formula = y~x,
                                                   se = TRUE, n = 80, span = 0.75, fullrange = FALSE,
                                                   xseq = NULL, level = 0.95, method.args = list(),
                                                   na.rm = FALSE, xpos=NULL, ypos=NULL) {
                            if (length(unique(data$x)) < 2) {
                              # Not enough data to perform fit
                              return(data.frame())
                            }
                            
                            if (is.null(data$weight)) data$weight <- 1
                            
                            if (is.null(xseq)) {
                              if (is.integer(data$x)) {
                                if (fullrange) {
                                  xseq <- scales$x$dimension()
                                } else {
                                  xseq <- sort(unique(data$x))
                                }
                              } else {
                                if (fullrange) {
                                  range <- scales$x$dimension()
                                } else {
                                  range <- range(data$x, na.rm = TRUE)
                                }
                                xseq <- seq(range[1], range[2], length.out = n)
                              }
                            }
                            # Special case span because it's the most commonly used model argument
                            if (identical(method, "loess")) {
                              method.args$span <- span
                            }
                            
                            if (is.character(method)) method <- match.fun(method)
                            
                            base.args <- list(quote(formula), data = quote(data), weights = quote(weight))
                            model <- do.call(method, c(base.args, method.args))
                            
                            m = model
                            eq <- substitute(italic(R)^2~"="~r2, 
                                             list(a = format(coef(m)[1], digits = 3), 
                                                  b = format(coef(m)[2], digits = 3), 
                                                  r2 = format(summary(m)$r.squared, digits = 3)))
                            func_string = as.character(as.expression(eq))
                            
                            if(is.null(xpos)) xpos = 5000
                            if(is.null(ypos)) ypos = 25000
                            data.frame(x=xpos, y=ypos, label=func_string)
                            
                          },
                          
                          required_aes = c("x", "y")
)


