# library(sf)
# library(spdep)
# library(tmap)
# 
# coords<-cbind(df[, c("Pixel_Long", "Pixel_Lat")]) # Set coordinates.
# coords<-as.matrix(coords)
# nb<-dnearneigh(coords,0,4000,longlat = TRUE) # Set distance class.
# lw<-nb2listw(nb, glist = NULL, style = 'W', zero.policy = TRUE)
# 
# moran.test(df$resid, lw, zero.policy = TRUE)
# 
# moran.mc(df$resid, lw, nsim=999, alternative="greater")
# 
# library(ape)
# Moran.I(df$resid, x.distance.weights, scaled = FALSE, na.rm = FALSE,
#         alternative = "two.sided")
# 
# x.distance.weights <- weights_from_distance_matrix(
#   distance.matrix = distance.matrix,
#   distance.threshold = 0
# )
# 
# listw <- mat2listw(x.distance.weights)
# 
# # Calculate Moran's I
# moran.test(df$resid, listw, zero.policy = TRUE)
# 
# 
# ggplot() +
#   geom_polygon(data = map_data("world"), 
#                aes(x = long, y = lat, group = group),
#                fill = "#bababa",
#                color = NA,
#                linewidth = 0.1) + 
#   coord_fixed(1.1) +
#   geom_point(data = df %>% mutate(resid = resid-mean(df$resid)) %>% filter(resid > 300), 
#              aes(x = Pixel_Long, y = Pixel_Lat, fill = resid),
#              color = "black",
#              pch = 21
#   ) +
#   theme_minimal() +
#   theme(#legend.position = "none",
#     #legend.box="horizontal",
#     panel.grid = element_blank(),
#     axis.title=element_blank(),
#     axis.text=element_blank()) 



# install necessary packages
# install.packages(c("shiny", "leaflet", "dplyr"))

library(shiny)
library(leaflet)
library(dplyr)
library(viridis)
library(patchwork)
library(data.table)
library(tidyverse)


# Distance matrix, calculate only once to speed things up
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230501_ectomycorrhizal_richness_pred_obs.csv') %>% 
  group_by(Pixel_Lat, Pixel_Long) %>%
  filter(row_number()==1) %>%
  mutate(resid = ectomycorrhizal_richness - ectomycorrhizal_richness_Predicted)

# Distance matrix, calculate only once to speed things up
df2 <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230501_ectomycorrhizal_richness_pred_obs.csv') %>% 
  mutate(resid = ectomycorrhizal_richness - ectomycorrhizal_richness_Predicted)


ui <- fluidPage(
  titlePanel("Interactive Map"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("residFilter", 
                  "Filter by Resid Value:", 
                  min = min(df$resid), 
                  max = max(df$resid), 
                  value = c(min(df$resid), max(df$resid)))
    ),
    mainPanel(
      leafletOutput("worldMap")
    )
  )
)

server <- function(input, output) {
  output$worldMap <- renderLeaflet({
    filtered_df <- df %>% mutate(resid = resid-mean(df$resid)) %>% filter(resid > input$residFilter[1] & resid < input$residFilter[2])
    
    # Create color palette function using viridis
    pal <- colorNumeric(palette = viridis(256), domain = filtered_df$resid)
    
    leaflet(data = filtered_df) %>%
      addProviderTiles(providers$CartoDB.Positron) %>% 
      addCircleMarkers(lng = ~Pixel_Long, lat = ~Pixel_Lat, 
                       color = ~pal(resid),
                       radius = 2,
                       label = ~as.character(resid), # display resid on hover
                       labelOptions = labelOptions(noHide = F, direction = 'auto'))
  })
}

# Run the application 
shinyApp(ui = ui, server = server)




library(shiny)
library(leaflet)
library(dplyr)
library(viridis)
library(patchwork)
library(data.table)
library(tidyverse)


# Distance matrix, calculate only once to speed things up
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230501_ectomycorrhizal_richness_pred_obs.csv') %>% 
  group_by(Pixel_Lat, Pixel_Long) %>%
  filter(row_number()==1) %>%
  mutate(resid = ectomycorrhizal_richness - ectomycorrhizal_richness_Predicted)

# Distance matrix, calculate only once to speed things up
df2 <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230501_ectomycorrhizal_richness_pred_obs.csv') %>% 
  mutate(resid = ectomycorrhizal_richness - ectomycorrhizal_richness_Predicted)

metadata <- fread('/Users/johanvandenhoogen/SPUN/richness_pipeline/data/REL4_Colin_datasets_samples_metadata.csv') %>% 
  select(sample_ID, paper_id) %>% 
  rename(sample_id = sample_ID)


ui <- fluidPage(
  titlePanel("Interactive Map"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("residFilter", 
                  "Filter by Resid Value:", 
                  min = min(df$resid), 
                  max = max(df$resid), 
                  value = c(min(df$resid), max(df$resid)))
    ),
    mainPanel(
      leafletOutput("worldMap"),
      tableOutput("table") # Add this line to display the table
    )
  )
)

server <- function(input, output) {
  output$worldMap <- renderLeaflet({
    filtered_df <- df %>% mutate(resid = resid-mean(df$resid)) %>% filter(resid > input$residFilter[1] & resid < input$residFilter[2])
    
    # Create color palette function using viridis
    pal <- colorNumeric(palette = viridis(256), domain = filtered_df$resid)
    
    leaflet(data = filtered_df) %>%
      addProviderTiles(providers$CartoDB.Positron) %>%
      addCircleMarkers(lng = ~Pixel_Long, lat = ~Pixel_Lat,
                       color = ~pal(resid),
                       radius = 2,
                       label = ~as.character(resid), # display resid on hover
                       labelOptions = labelOptions(noHide = F, direction = 'auto'))
  })
  
  # Render the table
  output$table <- renderTable({
    filtered_df <- df %>% mutate(resid = resid-mean(df$resid)) %>% filter(resid > input$residFilter[1] & resid < input$residFilter[2])
    df2_filtered <- df2 %>% 
      semi_join(filtered_df, by = c("Pixel_Long", "Pixel_Lat")) %>% 
      left_join(., metadata)
    df2_filtered %>% select(sample_id, resid, paper_id) %>% head(10)
  }, options = list(pageLength = 25)) # Adjust this number as per your need
  
}
# Run the application 
shinyApp(ui = ui, server = server)










library(data.table)
library(gstat)
library(spdep)
library(patchwork)
library(tidyverse)

classProperty <- 'ectomycorrhizal_richness'
resid <- 'resid'

# Distance matrix, calculate only once to speed things up
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230501_ectomycorrhizal_richness_pred_obs.csv') %>% 
  group_by(Pixel_Lat, Pixel_Long) %>%
  filter(row_number()==1) %>%
  mutate(resid = (ectomycorrhizal_richness - ectomycorrhizal_richness_Predicted))

coordinates(df) <- c("Pixel_Long", "Pixel_Lat")
proj4string(df) <- CRS("+init=epsg:4326") # WGS 84
CRS.new <- CRS("+init=epsg:3395")
df <- spTransform(df, CRS.new)
df <- as.data.frame(df)

dt_sp <- df %>% mutate(LAT = Pixel_Lat, LON = Pixel_Long)
coordinates(dt_sp) = ~LAT + LON

tmp <- variogram(as.formula(paste0(resid," ~ 1 ")), data = dt_sp, width = 10000, cutoff = 2000000) %>%
  mutate(var = "Model Residuals") %>% 
  rbind(., variogram(as.formula(paste0(classProperty," ~ 1 ")), data = dt_sp, width = 10000, cutoff = 2000000) %>% mutate(var = 'Ectomycorrhizal Richness (observed)'))

plot_em <- tmp %>% ggplot(aes(x = dist/1000, y = gamma, color = var)) +
  geom_point() +
  ylab("Semivariance") + xlab("Distance (km)") +
  geom_smooth(se = T)




classProperty <- 'arbuscular_mycorrhizal_richness'
resid <- 'resid'

# Distance matrix, calculate only once to speed things up
df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/output/20230510_arbuscular_mycorrhizal_richness_pred_obs.csv') %>% 
  group_by(Pixel_Lat, Pixel_Long) %>%
  filter(row_number()==1) %>%
  mutate(resid = (arbuscular_mycorrhizal_richness - arbuscular_mycorrhizal_richness_Predicted))

coordinates(df) <- c("Pixel_Long", "Pixel_Lat")
proj4string(df) <- CRS("+init=epsg:4326") # WGS 84
CRS.new <- CRS("+init=epsg:3395")
df <- spTransform(df, CRS.new)
df <- as.data.frame(df)

dt_sp <- df %>% mutate(LAT = Pixel_Lat, LON = Pixel_Long)
coordinates(dt_sp) = ~LAT + LON

tmp <- variogram(as.formula(paste0(resid," ~ 1 ")), data = dt_sp, width = 10000, cutoff = 2000000) %>%
  mutate(var = "Model Residuals") %>% 
  rbind(., variogram(as.formula(paste0(classProperty," ~ 1 ")), data = dt_sp, width = 10000, cutoff = 2000000) %>% mutate(var = 'Arbuscular Mycorrhizal Richness (observed)'))

plot_am <- tmp %>% ggplot(aes(x = dist/1000, y = gamma, color = var)) +
  geom_point() +
  ylab("Semivariance") + xlab("Distance (km)") +
  geom_smooth(se = T) 




plot_am + plot_em +
  plot_layout(guides = 'collect')


