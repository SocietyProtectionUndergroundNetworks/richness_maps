import ee
from ee_plugin import Map

ecto = ee.Image('users/johanvandenhoogen/000_SPUN_GFv4/ectomycorrhizal/ectomycorrhizal_richnessclassifiedImage_zeroInflated').select(2)

arb = ee.Image('users/johanvandenhoogen/000_SPUN_GFv4/arbuscular_mycorrhizal/arbuscular_mycorrhizal_richnessclassifiedImage_zeroInflated').select(2)

viridis = ["440154", "472D7B", "3B528B", "2C728E", "21908C", "27AD81", "5DC863", "AADC32", "FDE725"]

Map.addLayer(ecto, {'min':0, 'max': 60, 'palette': viridis}, 'Ectomycorrhizal Richness')
Map.addLayer(arb, {'min':0, 'max': 12, 'palette:' viridis}, 'Arbuscular Mycorrhizal Richness')