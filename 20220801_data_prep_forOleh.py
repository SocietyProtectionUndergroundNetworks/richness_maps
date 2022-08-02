# Import the modules of interest
import pandas as pd
import numpy as np

classProperty = 'ECM_diversity'

# List of the covariates to use
covariateList = [
'CGIAR_PET',
'CHELSA_BIO_Annual_Mean_Temperature',
'CHELSA_BIO_Annual_Precipitation',
'CHELSA_BIO_Max_Temperature_of_Warmest_Month',
'CHELSA_BIO_Precipitation_Seasonality',
'ConsensusLandCover_Human_Development_Percentage',
'ConsensusLandCoverClass_Barren',
'ConsensusLandCoverClass_Deciduous_Broadleaf_Trees',
'ConsensusLandCoverClass_Evergreen_Broadleaf_Trees',
'ConsensusLandCoverClass_Evergreen_Deciduous_Needleleaf_Trees',
'ConsensusLandCoverClass_Herbaceous_Vegetation',
'ConsensusLandCoverClass_Mixed_Other_Trees',
'ConsensusLandCoverClass_Shrubs',
'EarthEnvTexture_CoOfVar_EVI',
'EarthEnvTexture_Correlation_EVI',
'EarthEnvTexture_Homogeneity_EVI',
'EarthEnvTopoMed_AspectCosine',
'EarthEnvTopoMed_AspectSine',
'EarthEnvTopoMed_Elevation',
'EarthEnvTopoMed_Slope',
'EarthEnvTopoMed_TopoPositionIndex',
'EsaCci_BurntAreasProbability',
'GHS_Population_Density',
'GlobBiomass_AboveGroundBiomass',
'GlobPermafrost_PermafrostExtent',
'MODIS_NPP',
'PelletierEtAl_SoilAndSedimentaryDepositThicknesses',
'SG_Depth_to_bedrock',
'SG_Sand_Content_005cm',
'SG_SOC_Content_005cm',
'SG_Soil_pH_H2O_005cm',
]

project_vars = [
# 'top',
# 'bot',
# 'core_length',
'target_marker',
'sequencing_platform',
'sample_type',
'primers'
]

####################################################################################################################################################################
# Data processing
####################################################################################################################################################################
# Import raw data
rawPointCollection = pd.read_csv('data/20211026_ECM_diversity_data_sampled.csv', float_precision='round_trip')
rawPointCollection['source'] = 'GlobalFungi'

# Rename columnto be mapped
rawPointCollection.rename(columns={'myco_diversity': classProperty}, inplace=True)

# Convert factors to integers
rawPointCollection = rawPointCollection.assign(sequencing_platform = (rawPointCollection['sequencing_platform']).astype('category').cat.codes)
rawPointCollection = rawPointCollection.assign(sample_type = (rawPointCollection['sample_type']).astype('category').cat.codes)
rawPointCollection = rawPointCollection.assign(primers = (rawPointCollection['primers']).astype('category').cat.codes)
rawPointCollection = rawPointCollection.assign(target_marker = (rawPointCollection['target_marker']).astype('category').cat.codes)


# Shuffle the data frame while setting a new index to ensure geographic clumps of points are not clumped in any way
fcToAggregate = rawPointCollection.sample(frac = 1, random_state = 42).reset_index(drop=True)

preppedCollection = pd.DataFrame(fcToAggregate.groupby(['Pixel_Lat', 'Pixel_Long']).mean().to_records())[covariateList+["Resolve_Biome"]+[classProperty]+['Pixel_Lat', 'Pixel_Long']

print('Number of aggregated pixels', preppedCollection.shape[0])

# Drop NAs
preppedCollection = preppedCollection.dropna(how='any')
print('After dropping NAs', preppedCollection.shape[0])

# Log transform classProperty
preppedCollection[classProperty] = np.log(preppedCollection[classProperty] + 1)

# Convert biome column to int, to correct odd rounding errors
preppedCollection['Resolve_Biome'] = preppedCollection['Resolve_Biome'].astype(int)

# Add fold assignments to each of the points, stratified by biome
preppedCollection['CV_Fold'] = (preppedCollection.groupby('Resolve_Biome').cumcount() % k) + 1

# Write the CSV
preppedCollection.to_csv('data/ECM_diversity_pixelAggMeidan_for_Oleh.csv',index=False)
