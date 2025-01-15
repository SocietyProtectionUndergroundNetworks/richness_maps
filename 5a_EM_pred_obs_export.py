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

ee.Initialize()

guild = 'ectomycorrhizal'

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
'sequencing_platformIonTorrent',
'sequencing_platformPacBio',
'sample_typerhizosphere_soil',
'sample_typesoil',
'sample_typetopsoil',
'primers5_8S_Fun_ITS4_Fun',
'primersfITS7_ITS4',
'primersfITS9_ITS4',
'primersgITS7_ITS4',
'primersgITS7_ITS4_then_ITS9_ITS4',
'primersgITS7_ITS4_ITS4arch',
'primersgITS7_ITS4m',
'primersgITS7_ITS4ngs',
'primersgITS7ngs_ITS4ngsUni',
'primersITS_S2F___ITS3_mixed_1_1_ITS4',
'primersITS1_ITS4',
'primersITS1F_ITS4',
'primersITS1F_ITS4_then_fITS7_ITS4',
'primersITS1F_ITS4_then_ITS3_ITS4',
'primersITS1ngs_ITS4ngs_or_ITS1Fngs_ITS4ngs',
'primersITS3_KYO2_ITS4',
'primersITS3_ITS4',
'primersITS3ngs1_to_5___ITS3ngs10_ITS4ngs',
'primersITS3ngs1_to_ITS3ngs11_ITS4ngs',
'primersITS86F_ITS4',
'primersITS9MUNngs_ITS4ngsUni',
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

# Define hyperparameters for grid search
varsPerSplit_list = list(range(4,14,2))
leafPop_list = list(range(2,14,2))

classifierListRegression = []
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

        classifierListRegression.append(rf)

classifierListClassification = []
# Create list of classifiers for classification
for vps in varsPerSplit_list:
    for lp in leafPop_list:

        model_name = classProperty + '_rf_VPS' + str(vps) + '_LP' + str(lp) + 'CLASSIFICATION'

        rf = ee.Feature(ee.Geometry.Point([0,0])).set('cName',model_name,'c',ee.Classifier.smileRandomForest(
        numberOfTrees = nTrees,
        variablesPerSplit = vps,
        minLeafPopulation = lp,
        bagFraction = 0.632,
        seed = 42
        ).setOutputMode('CLASSIFICATION'))

        classifierListClassification.append(rf)

# Fetch FC from GEE
grid_search_resultsRegression = ee.FeatureCollection([ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/hyperparameter_tuning/'+rf.get('cName').getInfo()) for rf in classifierListRegression]).flatten()
grid_search_resultsClassification = ee.FeatureCollection([ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/hyperparameter_tuning/'+rf.get('cName').getInfo()) for rf in classifierListClassification]).flatten()

# Get top model name
bestModelNameRegression = grid_search_resultsRegression.limit(1, sort_acc_prop, False).first().get('cName')
bestModelNameClassification = grid_search_resultsClassification.limit(1, 'Mean_overallAccuracy_Random', False).first().get('cName')

# Get top 10 models
top_10ModelsRegression = grid_search_resultsRegression.limit(10, sort_acc_prop, False).aggregate_array('cName')
top_10ModelsClassification = grid_search_resultsClassification.limit(10, 'Mean_overallAccuracy_Random', False).aggregate_array('cName')

##################################################################################################################################################################
# Predicted - Observed
##################################################################################################################################################################
fcOI = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)

try:
    predObs_wResiduals = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_pred_obs')
    predObs_wResiduals.size().getInfo()

except Exception as e:
    for n in list(range(0,10)):
        modelNameRegression = top_10ModelsRegression.get(n)
        modelNameClassification = top_10ModelsClassification.get(n)

        # Load the best model from the classifier list
        classifierRegression = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListRegression).filterMetadata('cName', 'equals', modelNameRegression).first()).get('c'))
        classifierClassification = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListClassification).filterMetadata('cName', 'equals', modelNameClassification).first()).get('c'))

        # Train the classifier with the collection
        # REGRESSION
        fcOI_forRegression = fcOI.filter(ee.Filter.neq(classProperty, 0))
        trainedClassiferRegression = classifierRegression.train(fcOI_forRegression, classProperty, covariateList)

        # Classification
        fcOI_forClassification = fcOI.map(lambda f: f.set(classProperty+'_forClassification', ee.Number(f.get(classProperty)).divide(f.get(classProperty)))) # train classifier on 0 (classProperty == 0) or 1 (classProperty != 0)
        trainedClassiferClassification = classifierClassification.train(fcOI_forClassification, classProperty+'_forClassification', covariateList)

        # Classify the FC
        def classifyFunction(f):
            classfiedRegression = ee.FeatureCollection([f]).classify(trainedClassiferRegression,classProperty+'_Regressed').first()
            classfiedClassification = ee.FeatureCollection([f]).classify(trainedClassiferClassification,classProperty+'_Classified').first()

            featureToReturn = classfiedRegression.set(classProperty+'_Classified', classfiedClassification.get(classProperty+'_Classified'))

            # Calculate final predicted value as product of classification and regression
            featureToReturn = featureToReturn.set(classProperty+'_Predicted', ee.Number(featureToReturn.get(classProperty+'_Classified')).multiply(ee.Number(featureToReturn.get(classProperty+'_Regressed'))))
            return featureToReturn

        # Classify fcOI
        predObs = fcOI.map(classifyFunction)

        # Add coordinates to FC
        predObs = predObs.map(addLatLon)

        # back-log transform predicted and observed values
        if log_transform_classProperty == True:
            predObs = predObs.map(lambda f: f.set(classProperty, ee.Number(f.get(classProperty)).exp().subtract(1)))
            predObs = predObs.map(lambda f: f.set(classProperty+'_Predicted', ee.Number(f.get(classProperty+'_Predicted')).exp().subtract(1)))
            predObs = predObs.map(lambda f: f.set(classProperty+'_Regressed', ee.Number(f.get(classProperty+'_Regressed')).exp().subtract(1)))

        # Add residuals to FC
        predObs_wResiduals = predObs.map(lambda f: f.set('AbsResidual', ee.Number(f.get(classProperty+'_Predicted')).subtract(f.get(classProperty)).abs()))

        # Export to Assets
        predObsexport = ee.batch.Export.table.toAsset(
            collection = predObs_wResiduals,
            description = classProperty+'_pred_obs_rep_'+str(n),
            assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_pred_obs_rep_'+str(n)
        )
        predObsexport.start()