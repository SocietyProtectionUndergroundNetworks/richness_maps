import ee
from time import sleep
import multiprocessing
import math
import time
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial
from contextlib import contextmanager
from ctypes import c_int
from multiprocessing import Value, Lock, Process
import datetime as datetime

ee.Initialize()

guild = 'arbuscular_mycorrhizal'

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
projectFolder = '000_SPUN_GFv4_9/' + guild

# Specify whether to use spatial or random CV
spatialCV = False 

# Input the name of the property that holds the CV fold assignment
cvFoldHeader = 'CV_Fold'

cvFoldString_Spatial = cvFoldHeader + '_Spatial'
cvFoldString_Random = cvFoldHeader + '_Random'

# Input the title of the CSV that will hold all of the data that has been given a CV fold assignment
titleOfCSVWithCVAssignments = classProperty+"_training_data"

# Set k for k-fold CV
k = 10

# Make a list of the k-fold CV assignments to use
kList = list(range(1,k+1))

# Set number of trees in RF models
nTrees = 250

# Metric to use for sorting k-fold CV hyperparameter tuning (default: R2)
sort_acc_prop = 'Mean_R2' # (either one of 'Mean_R2', 'Mean_MAE', 'Mean_RMSE')

if spatialCV == True:
    sort_acc_prop = sort_acc_prop + '_Spatial'
else:
    sort_acc_prop = sort_acc_prop + '_Random'

# Log transform classProperty? Boolean, either True or False
log_transform_classProperty = True

# Specify the column names where the latitude and longitude information is stored
latString = 'Pixel_Lat'
longString = 'Pixel_Long'

####################################################################################################################################################################
# Covariate data settings
####################################################################################################################################################################

# List of the covariates to use
covariateList = [
'CGIAR_PET',
'CHELSA_BIO_Annual_Mean_Temperature',
'CHELSA_BIO_Annual_Precipitation',
'CHELSA_BIO_Max_Temperature_of_Warmest_Month',
'CHELSA_BIO_Precipitation_Seasonality',
'ConsensusLandCover_Human_Development_Percentage',
# 'ConsensusLandCoverClass_Barren',
# 'ConsensusLandCoverClass_Deciduous_Broadleaf_Trees',
# 'ConsensusLandCoverClass_Evergreen_Broadleaf_Trees',
# 'ConsensusLandCoverClass_Evergreen_Deciduous_Needleleaf_Trees',
# 'ConsensusLandCoverClass_Herbaceous_Vegetation',
# 'ConsensusLandCoverClass_Mixed_Other_Trees',
# 'ConsensusLandCoverClass_Shrubs',
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
# 'GlobPermafrost_PermafrostExtent',
'MODIS_NPP',
# 'PelletierEtAl_SoilAndSedimentaryDepositThicknesses',
'SG_Depth_to_bedrock',
'SG_Sand_Content_005cm',
'SG_SOC_Content_005cm',
'SG_Soil_pH_H2O_005cm',
]

compositeOfInterest = ee.Image('projects/crowtherlab/Composite/CrowtherLab_Composite_30ArcSec')

project_vars = [
'sequencing_platform454Roche',
'sequencing_platformIllumina',
'sample_typerhizosphere_soil',
'sample_typesoil',
'sample_typetopsoil',
'primersAML1_AML2_then_AMV4_5NF_AMDGR',
'primersAML1_AML2_then_NS31_AM1',
'primersAML1_AML2_then_nu_SSU_0595_5__nu_SSU_0948_3_',
'primersAMV4_5F_AMDGR',
'primersAMV4_5NF_AMDGR',
'primersGeoA2_AML2_then_NS31_AMDGR',
'primersGeoA2_NS4_then_NS31_AML2',
'primersGlomerWT0_Glomer1536_then_NS31_AM1A_and_GlomerWT0_Glomer1536_then_NS31_AM1B',
'primersGlomerWT0_Glomer1536_then_NS31_AM1A__GlomerWT0_Glomer1536_then_NS31_AM1B',
'primersNS1_NS4_then_AML1_AML2',
'primersNS1_NS4_then_AMV4_5NF_AMDGR',
'primersNS1_NS4_then_NS31_AM1',
'primersNS1_NS41_then_AML1_AML2',
'primersNS31_AM1',
'primersNS31_AML2',
'primersWANDA_AML2',
]

covariateList = covariateList + project_vars

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



fcOI = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)

fcOI = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)
print(guild + ' hyperparameter tuning')

# Define hyperparameters for grid search
varsPerSplit_list = list(range(4,14,2))
leafPop_list = list(range(2,14,2))

classifierList = []
# Create list of classifiers for regression
for vps in varsPerSplit_list:
    for lp in leafPop_list:

        model_name = classProperty + '_rf_VPS' + str(vps) + '_LP' + str(lp) + '_REGRESSION'

        rf = ee.Feature(ee.Geometry.Point([0,0])).set('cName',model_name,'c',ee.Classifier.smileRandomForest(
        numberOfTrees = nTrees,
        variablesPerSplit = vps,
        minLeafPopulation = lp,
        bagFraction = 0.632,
        seed = 42
        ).setOutputMode('REGRESSION'))

        classifierList.append(rf)

# Fetch FC from GEE
grid_search_results = ee.FeatureCollection([ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/hyperparameter_tuning/'+rf.get('cName').getInfo()) for rf in classifierList]).flatten()

# Get top model name
bestModelName = grid_search_results.limit(1, sort_acc_prop, False).first().get('cName')

# Get top 10 models
top_10Models = grid_search_results.limit(10, sort_acc_prop, False).aggregate_array('cName')


##################################################################################################################################################################
# Predicted - Observed
##################################################################################################################################################################
fcOI = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)

try:
    predObs_wResiduals = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_pred_obs')
    predObs_wResiduals.size().getInfo()

except Exception as e:
    def predObs(fcOI):
            def classifyFC(classifierName):
                # Load the best model from the classifier list
                classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', classifierName).first()).get('c'))
            
                # Train the classifier with the collection
                trainedClassifier = classifier.train(fcOI, classProperty, covariateList)

                # Classify the FC
                classifiedFC = fcOI.classify(trainedClassifier,classProperty+'_Predicted')

                return classifiedFC

            # Map function over list of models in ensemble
            classifiedFC = ee.FeatureCollection(top_10Models.map(classifyFC)).flatten()

            return classifiedFC

    # Classify FC
    predObs = predObs(fcOI)

    # Add coordinates to FC
    predObs = predObs.map(addLatLon)

    # reverse log transform predicted and observed values
    if log_transform_classProperty == True:
        predObs = predObs.map(lambda f: f.set(classProperty, ee.Number(f.get(classProperty)).exp().subtract(1)))
        predObs = predObs.map(lambda f: f.set(classProperty+'_Predicted', ee.Number(f.get(classProperty+'_Predicted')).exp().subtract(1)))

    # Add residuals to FC
    predObs_wResiduals = predObs.map(lambda f: f.set('AbsResidual', ee.Number(f.get(classProperty+'_Predicted')).subtract(f.get(classProperty)).abs()))

    # Export to Assets
    predObsexport = ee.batch.Export.table.toAsset(
        collection = predObs_wResiduals,
        description = classProperty+'_pred_obs',
        assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_pred_obs'
    )
    predObsexport.start()

    # !! Break and wait
    count = 1
    while count >= 1:
        taskList = [str(i) for i in ee.batch.Task.list()]
        subsetList = [s for s in taskList if classProperty in s]
        subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
        count = len(subsubList)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Waiting for pred/obs to complete...', end = '\r')
        time.sleep(normalWaitTime)
    print('Moving on...')

    predObs_wResiduals = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_pred_obs')

# Convert to pd
predObs_df = GEE_FC_to_pd(predObs_wResiduals)

# Group by sample ID to return mean across ensemble prediction
predObs_df = pd.DataFrame(predObs_df.groupby('sample_id').mean().to_records())

predObs_df.to_csv('output/20230501_'+classProperty+'_pred_obs.csv')