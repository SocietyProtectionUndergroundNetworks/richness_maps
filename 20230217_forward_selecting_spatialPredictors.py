import ee
import multiprocessing
import pandas as pd
import numpy as np
from functools import partial
import datetime
from contextlib import contextmanager

ee.Initialize()

today = datetime.date.today().strftime("%Y%m%d")

guild = 'arbuscular_mycorrhizal'


####################################################################################################################################################################
# Helper functions
####################################################################################################################################################################
# Function to convert GEE FC to pd.DataFrame. Not ideal as it's calling .getInfo(), but does the job
def GEE_FC_to_pd(fc):
    result = []

    values = fc.toList(500000).getInfo()

    BANDS = fc.first().propertyNames().getInfo()

    if 'system:index' in BANDS: BANDS.remove('system:index')

    for item in values:
        values_item = item['properties']
        row = [values_item[key] for key in BANDS]
        result.append(row)

    df = pd.DataFrame([item for item in result], columns = BANDS)
    df.replace('None', np.nan, inplace = True)

    return df

# Add point coordinates to FC as properties
def addLatLon(f):
    lat = f.geometry().coordinates().get(1)
    lon = f.geometry().coordinates().get(0)
    return f.set(latString, lat).set(longString, lon)


####################################################################################################################################################################
# Configuration
####################################################################################################################################################################
# Input the name of the username that serves as the home folder for asset storage
usernameFolderString = 'johanvandenhoogen'

# Input the Cloud Storage Bucket that will hold the bootstrap collections when uploading them to Earth Engine
# !! This bucket should be pre-created before running this script
bucketOfInterest = 'johanvandenhoogen'

# Input the name of the classification property
classProperty = guild + '_richness'

# Input the name of the project folder inside which all of the assets will be stored
# This folder will be generated automatically below, if it isn't yet present
projectFolder = '000_SPUN_GFv4_8/' + guild + '_wSpatialPreds'

# Input the normal wait time (in seconds) for "wait and break" cells
normalWaitTime = 5

# Input a longer wait time (in seconds) for "wait and break" cells
longWaitTime = 10

# Specify the column names where the latitude and longitude information is stored
latString = 'Pixel_Lat'
longString = 'Pixel_Long'

def get_prebObs(iteration):
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
    # 'EarthEnvTopoMed_AspectCosine',
    # 'EarthEnvTopoMedAspectSine',
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
    # 'target_marker', # omit: all the same
    'sequencing_platform',
    'sample_type',
    'primers'
    ]

    spatial_preds = ['MEM1', 'MEM10', 'MEM11', 'MEM13', 'MEM18', 'MEM19', 'MEM20', 'MEM30', 'MEM35', 'MEM37', 'MEM4', 'MEM45', 'MEM51', 'MEM52', 'MEM58', 'MEM6', 'MEM7', 'MEM8', 'MEM81', 'MEM9']

    covariateList = covariateList + project_vars + spatial_preds[0:iteration]

    ##################################################################################################################################################################
    # Predicted - Observed
    ##################################################################################################################################################################
    fcOI = ee.FeatureCollection('users/johanvandenhoogen/000_SPUN_GFv4_8/arbuscular_mycorrhizal_wSpatialPreds/arbuscular_mycorrhizal_richness_training_data')

    classifier = ee.Classifier.smileRandomForest(
            numberOfTrees = 250,
            # variablesPerSplit = vps,
            # minLeafPopulation = lp,
            bagFraction = 0.632,
            seed = 42
            ).setOutputMode('REGRESSION')

    # Train the classifier with the collection
    trainedClassifier = classifier.train(fcOI, classProperty, covariateList)

    # Classify the FC
    classifiedFC = fcOI.classify(trainedClassifier,classProperty+'_Predicted')

    # Add coordinates to FC
    predObs = classifiedFC.map(addLatLon)

    # Add residuals to FC
    predObs_wResiduals = predObs.map(lambda f: f.set('AbsResidual', ee.Number(f.get(classProperty+'_Predicted')).subtract(f.get(classProperty)).abs()))

    # Convert to pd
    predObs_df = GEE_FC_to_pd(predObs_wResiduals)

    # Group by sample ID to return mean across ensemble prediction
    predObs_df = pd.DataFrame(predObs_df.groupby('sample_id').mean().to_records())
    predObs_df['number_of_spatialpredictors'] = iteration
    return predObs_df
    # predObs_df.to_csv('output/'+today+'_'+classProperty+'_pred_obs_w_'+iteration+'_spatialPreds.csv')


@contextmanager
def poolcontext(*args, **kwargs):
		"""This just makes the multiprocessing easier with a generator."""
		pool = multiprocessing.Pool(*args, **kwargs)
		yield pool
		pool.terminate()

if __name__ == '__main__':

		# How many concurrent processors to use.  If you're hitting lots of
		# "Too many aggregation" errors (more than ~10/minute), then make this
		# number smaller.  You should be able to always use at least 20.
		NPROC = 20
		with poolcontext(NPROC) as pool:
				results = pool.map(
						partial(get_prebObs),
						list(range(0,21)))
				results = pd.concat(results)
				results.to_csv('data/'+today+'_predObs_forwardselected_spatialpredictors.csv')
