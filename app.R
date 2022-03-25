library(data.table)
library(tidyverse)

df <- fread('/Users/johanvandenhoogen/SPUN/richness_maps/data/20211026_ECM_diversity_data_sampled.csv')

df %>% 
  mutate(Resolve_Biome = as.factor(Resolve_Biome)) %>% 
  ggplot(aes(x = Resolve_Biome, y = myco_diversity)) +
  geom_boxplot(outlier.shape = NA, fill = NA) +
  geom_jitter(aes(color = Resolve_Biome), alpha = 0.25) +
  facet_wrap(vars(Resolve_Biome), scales = "free") +
  theme(legend.position = 'none')


# 
# library(shiny)
# 
# 
# ui <- fluidPage(
#   fluidRow(
#     column(#width = 12,
#       plotOutput("plot1", height = 300,
#                  # Equivalent to: click = clickOpts(id = "plot_click")
#                  click = "plot1_click",
#                  brush = brushOpts(
#                    id = "plot1_brush"
#                  )
#       )
#     )
#   ),
#   fluidRow(
#     column(width = 6,
#            h4("Points near click"),
#            verbatimTextOutput("click_info")
#     ),
#     column(width = 6,
#            h4("Brushed points"),
#            verbatimTextOutput("brush_info")
#     )
#   )
# )
# 
# server <- function(input, output) {
#   output$plot1 <- renderPlot({
#     df %>% 
#       mutate(Resolve_Biome = as.factor(Resolve_Biome)) %>% 
#       ggplot(aes(x = Resolve_Biome, y = myco_diversity)) +
#       geom_boxplot(outlier.shape = NA, fill = NA) +
#       geom_jitter(aes(color = Resolve_Biome), alpha = 0.25) +
#       # facet_wrap(vars(Resolve_Biome), scales = "free") +
#       theme(legend.position = 'none')
#   })
#   
#   output$click_info <- renderPrint({
#     # Because it's a ggplot2, we don't need to supply xvar or yvar; if this
#     # were a base graphics plot, we'd need those.
#     nearPoints(df, input$plot1_click, addDist = TRUE, threshold = 10) %>% pull(sample_id)
#   })
#   output$brush_info <- renderPrint({
#     brushedPoints(df, input$plot1_brush)%>% pull(sample_id)
#   })
# }
# 
# shinyApp(ui, server)


library(shiny)
library(leaflet)
library(DT)
library(tidyverse)

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

pal <- colorFactor(gg_color_hue(length(unique(df$Resolve_Biome))), domain = unique(df$Resolve_Biome))

ui <- fluidPage(column(12, leafletOutput("wsmap")),
                # column(12, plotOutput("plot")),
                column(12, dataTableOutput('table01'))
                
)

server <- function(input, output) {
  
  qSub <-  reactive({
    subset <- df %>% select(sample_id, Resolve_Biome, myco_diversity, Pixel_Lat, Pixel_Long) %>% 
      rename(lat = Pixel_Lat, long = Pixel_Long)
  })
  
  output$table01 <- renderDataTable({
    DT::datatable(qSub(), 
                  selection = "single", options = list(stateSave = TRUE))
  })
  
  # to keep track of previously selected row
  prev_row <- reactiveVal()
  
  observeEvent(input$table01_rows_selected, {
    row_selected = qSub()[input$table01_rows_selected,]
    print(row_selected)
    leafletProxy('wsmap') %>%
      addCircleMarkers(popup=as.character(row_selected$sample_id),
                        layerId = as.character(row_selected$sample_id),
                        lng=row_selected$long, 
                        lat=row_selected$lat)
    
    # Reset previously selected marker
    # if(!is.null(prev_row()))
    # {leafletProxy('wsmap') %>%
    #     # removeMarker(layerId = as.character(prev_row$sample_id))
    #     
    # }
    # # set new value to reactiveVal
    # prev_row(row_selected)
  })
  
    ## leaflet map
  output$wsmap <- renderLeaflet({
    leaflet() %>% 
      addTiles() %>% 
      addCircleMarkers(data = qSub(),
                       layerId = ~unique(sample_id), 
                       popup = ~unique(sample_id),
                       color = ~pal(Resolve_Biome),
                       radius = 3,
                       stroke = FALSE, 
                       fillOpacity = 0.5) 
  })
  
  observeEvent(input$wsmap_marker_click, {
    clickId <- input$wsmap_marker_click$id
    print(clickId)
    # dataTableProxy("table01") %>%
    #   selectRows(which(qSub()$sample_id == clickId)) %>%
    #   selectPage(which(input$table01_rows_all == clickId) %/% input$table01_state$length + 1)
    
    leafletProxy('wsmap') %>%
          removeMarker(layerId =clickId)
  })
  # 
  # 
  # output$plot <- renderPlot({
  #   df %>% 
  #     mutate(Resolve_Biome = as.factor(Resolve_Biome)) %>% 
  #     ggplot(aes(x = Resolve_Biome, y = myco_diversity)) +
  #     geom_boxplot(outlier.shape = NA, fill = NA) +
  #     geom_jitter(aes(color = Resolve_Biome), alpha = 0.25) +
  #     facet_wrap(vars(Resolve_Biome), scales = "free") +
  #     theme(legend.position = 'none')
  # })
}

shinyApp(ui, server)







##############################################################################
# Data
##############################################################################
qDat <- quakes
qDat$id <- seq.int(nrow(qDat))
##############################################################################
# UI Side
##############################################################################
ui <- fluidPage(
  titlePanel("Visualization of Fiji Earthquake"),
  
  # side panel
  sidebarPanel(
    h3('Fiji Earthquake Data'),
    
    sliderInput(
      inputId = "sld01_Mag",
      label="Show earthquakes of magnitude:", 
      min=min(qDat$mag), max=max(qDat$mag),
      value=c(min(qDat$mag),max(qDat$mag)), step=0.1
    ),
    
    plotlyOutput('hist01')
  ),
  
  # main panel
  mainPanel(
    leafletOutput('map01'),
    dataTableOutput('table01')
  )
  
)
##############################################################################
# Server Side
##############################################################################
server <- function(input,output){
  qSub <-  reactive({
    
    subset <- subset(qDat, qDat$mag>=input$sld01_Mag[1] &
                       qDat$mag<=input$sld01_Mag[2]) %>% head(25)
  })
  
  # histogram
  output$hist01 <- renderPlotly({
    ggplot(data=qSub(), aes(x=stations)) + 
      geom_histogram(binwidth=5) +
      xlab('Number of Reporting Stations') +
      ylab('Count') +
      xlim(min(qDat$stations), max(qDat$stations))+
      ggtitle('Fiji Earthquake')
  })
  
  # table
  output$table01 <- renderDataTable({
    
    DT::datatable(qSub(), selection = "single",options=list(stateSave = TRUE))
  })
  
  # to keep track of previously selected row
  prev_row <- reactiveVal()
  
  # new icon style
  my_icon = makeAwesomeIcon(icon = 'flag', markerColor = 'red', iconColor = 'white')
  
  observeEvent(input$table01_rows_selected, {
    row_selected = qSub()[input$table01_rows_selected,]
    proxy <- leafletProxy('map01')
    print(row_selected)
    proxy %>%
      addAwesomeMarkers(popup=as.character(row_selected$mag),
                        layerId = as.character(row_selected$id),
                        lng=row_selected$long, 
                        lat=row_selected$lat,
                        icon = my_icon)
    
    # Reset previously selected marker
    if(!is.null(prev_row()))
    {
      proxy %>%
        addMarkers(popup=as.character(prev_row()$mag), 
                   layerId = as.character(prev_row()$id),
                   lng=prev_row()$long, 
                   lat=prev_row()$lat)
    }
    # set new value to reactiveVal 
    prev_row(row_selected)
  })
  
  # map
  output$map01 <- renderLeaflet({
    pal <- colorNumeric("YlOrRd", domain=c(min(quakes$mag), max(quakes$mag)))
    qMap <- leaflet(data = qSub()) %>% 
      addTiles() %>%
      addMarkers(popup=~as.character(mag), layerId = as.character(qSub()$id)) %>%
      addLegend("bottomright", pal = pal, values = ~mag,
                title = "Earthquake Magnitude",
                opacity = 1)
    qMap
  })
  
  observeEvent(input$map01_marker_click, {
    clickId <- input$map01_marker_click$id
    dataTableProxy("table01") %>%
      selectRows(which(qSub()$id == clickId)) %>%
      selectPage(which(input$table01_rows_all == clickId) %/% input$table01_state$length + 1)
  })
}

##############################################################################
shinyApp(ui = ui, server = server)
##############################################################################