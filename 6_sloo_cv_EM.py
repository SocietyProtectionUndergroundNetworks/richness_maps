# Import the modules of interest
import pandas as pd
import numpy as np
import subprocess
import time
import datetime
import ee
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from itertools import combinations
from itertools import repeat
from functions.determineBlockSizeForCV import *

ee.Initialize()

today = datetime.date.today().strftime("%Y%m%d")

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
projectFolder = '000_SPUN_GFv4_12/' + guild

# Input the normal wait time (in seconds) for "wait and break" cells
normalWaitTime = 5

# Input a longer wait time (in seconds) for "wait and break" cells
longWaitTime = 10

# Specify the column names where the latitude and longitude information is stored
latString = 'Pixel_Lat'
longString = 'Pixel_Long'

# Log transform classProperty? Boolean, either True or False
log_transform_classProperty = True

# Ensemble of top 10 models?
ensemble = True

# Spatial leave-one-out cross-validation settings
# skip test points outside training space after removing points in buffer zone? This might reduce extrapolation but overestimate accuracy
loo_cv_wPointRemoval = False

# Define buffer size in meters; use Moran's I or other test to determine SAC range
# Alternatively: specify buffer size as list, to test across multiple buffer sizes
buffer_size = 100000

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
'plant_diversity',
'climate_stability_index'
]

plant_diversity = ee.Image('projects/crowtherlab/johan/SPUN_layers/plant_SR_Ensemble_rasterized').rename('plant_diversity')
climate_stability_index = ee.Image('projects/crowtherlab/johan/SPUN_layers/csi_past').rename('climate_stability_index')

composite = ee.Image.cat([
		ee.Image("projects/crowtherlab/Composite/CrowtherLab_bioComposite_30ArcSec"),
		ee.Image("projects/crowtherlab/Composite/CrowtherLab_climateComposite_30ArcSec"),
		ee.Image("projects/crowtherlab/Composite/CrowtherLab_geoComposite_30ArcSec"),
		ee.Image("projects/crowtherlab/Composite/CrowtherLab_processComposite_30ArcSec"),
		])

# Add plant diversity and climate stability index to composite and reproject to composite projection
compositeOfInterest = composite.addBands(plant_diversity).addBands(climate_stability_index).reproject(composite.projection()) 

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
'area_sampled',
'extraction_dna_mass',
]

covariateList = covariateList + project_vars

####################################################################################################################################################################
# Cross validation settings
####################################################################################################################################################################
# Set k for k-fold CV
k = 10

# Make a list of the k-fold CV assignments to use
kList = list(range(1,k+1))

# Set number of trees in RF models
nTrees = 250

# Specify whether to use spatial or random CV
spatialCV = True 

# Input the name of the property that holds the CV fold assignment
cvFoldHeader = 'CV_Fold'

cvFoldString_Spatial = cvFoldHeader + '_Spatial'
cvFoldString_Random = cvFoldHeader + '_Random'

# Metric to use for sorting k-fold CV hyperparameter tuning (default: R2)
sort_acc_prop = 'Mean_R2' # (either one of 'Mean_R2', 'Mean_MAE', 'Mean_RMSE')

if spatialCV == True:
    sort_acc_prop = sort_acc_prop + '_Spatial'
else:
    sort_acc_prop = sort_acc_prop + '_Random'

# Input the title of the CSV that will hold all of the data that has been given a CV fold assignment
titleOfCSVWithCVAssignments = today + "_" + classProperty+"_training_data"

# Asset ID of uploaded dataset after processing
assetIDForCVAssignedColl = 'users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments

# Write the name of a local staging area folder for outputted CSV's
holdingFolder = '/Users/johanvandenhoogen/SPUN/richness_maps/data/'
outputFolder = '/Users/johanvandenhoogen/SPUN/richness_maps/output'

# Create directory to hold training data
Path(holdingFolder).mkdir(parents=True, exist_ok=True)

####################################################################################################################################################################
# Export settings
####################################################################################################################################################################

# Set pyramidingPolicy for exporting purposes
pyramidingPolicy = 'mean'

# Load a geometry to use for the export
exportingGeometry = ee.Geometry.Polygon([[[-180, 88], [180, 88], [180, -88], [-180, -88]]], None, False)

####################################################################################################################################################################
# Bootstrap settings
####################################################################################################################################################################

# Number of bootstrap iterations
bootstrapIterations = 100

# Generate the seeds for bootstrapping
seedsToUseForBootstrapping = list(range(1, bootstrapIterations+1))

# Input the header text that will name the bootstrapped dataset
bootstrapSamples = classProperty+'_bootstrapSamples'

# Write the name of the variable used for stratification
stratificationVariableString = "Resolve_Biome"

# Input the dictionary of values for each of the stratification category levels
# !! This area breakdown determines the proportion of each biome to include in every bootstrap
strataDict = {
    1: 14.900835665820974,
    2: 2.941697660221864,
    3: 0.526059731441294,
    4: 9.56387696566245,
    5: 2.865354077500338,
    6: 11.519674266872787,
    7: 16.26999434439293,
    8: 8.047078485979089,
    9: 0.861212221078014,
    10: 3.623974712557433,
    11: 6.063922959332467,
    12: 2.5132866428302836,
    13: 20.037841544639985,
    14: 0.26519072167008,
}

####################################################################################################################################################################
# Bash and Google Cloud Bucket settings
####################################################################################################################################################################
# Specify the necessary arguments to upload the files to a Cloud Storage bucket
# I.e., create bash variables in order to create/check/delete Earth Engine Assets

# Specify main bash functions being used
bashFunction_EarthEngine = '/Users/johanvandenhoogen/miniconda3/envs/ee/bin/earthengine'
bashFunctionGSUtil = '/Users/johanvandenhoogen/google-cloud-sdk/bin/gsutil'

# Specify the arguments to these functions
arglist_preEEUploadTable = ['upload','table']
arglist_postEEUploadTable = ['--x_column', longString, '--y_column', latString]
arglist_preGSUtilUploadFile = ['cp']
formattedBucketOI = 'gs://'+bucketOfInterest
assetIDStringPrefix = '--asset_id='
arglist_CreateCollection = ['create','collection']
arglist_CreateFolder = ['create','folder']
arglist_Detect = ['asset','info']
arglist_Delete = ['rm','-r']
arglist_ls = ['ls']
stringsOfInterest = ['Asset does not exist or is not accessible']

# Compose the arguments into lists that can be run via the subprocess module
bashCommandList_Detect = [bashFunction_EarthEngine]+arglist_Detect
bashCommandList_Delete = [bashFunction_EarthEngine]+arglist_Delete
bashCommandList_ls = [bashFunction_EarthEngine]+arglist_ls
bashCommandList_CreateCollection = [bashFunction_EarthEngine]+arglist_CreateCollection
bashCommandList_CreateFolder = [bashFunction_EarthEngine]+arglist_CreateFolder

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

# R^2 function
def coefficientOfDetermination(fcOI,propertyOfInterest,propertyOfInterest_Predicted):
    # Compute the mean of the property of interest
    propertyOfInterestMean = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).select([propertyOfInterest]).reduceColumns(ee.Reducer.mean(),[propertyOfInterest])).get('mean'))

    # Compute the total sum of squares
    def totalSoSFunction(f):
        return f.set('Difference_Squared',ee.Number(ee.Feature(f).get(propertyOfInterest)).subtract(propertyOfInterestMean).pow(ee.Number(2)))
    totalSumOfSquares = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).map(totalSoSFunction).select(['Difference_Squared']).reduceColumns(ee.Reducer.sum(),['Difference_Squared'])).get('sum'))

    # Compute the residual sum of squares
    def residualSoSFunction(f):
        return f.set('Residual_Squared',ee.Number(ee.Feature(f).get(propertyOfInterest)).subtract(ee.Number(ee.Feature(f).get(propertyOfInterest_Predicted))).pow(ee.Number(2)))
    residualSumOfSquares = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).map(residualSoSFunction).select(['Residual_Squared']).reduceColumns(ee.Reducer.sum(),['Residual_Squared'])).get('sum'))

    # Finalize the calculation
    r2 = ee.Number(1).subtract(residualSumOfSquares.divide(totalSumOfSquares))

    return ee.Number(r2)

# RMSE function
def RMSE(fcOI,propertyOfInterest,propertyOfInterest_Predicted):
    # Compute the squared difference between observed and predicted
    def propDiff(f):
        diff = ee.Number(f.get(propertyOfInterest)).subtract(ee.Number(f.get(propertyOfInterest_Predicted)))

        return f.set('diff', diff.pow(2))

    # calculate RMSE from squared difference
    rmse = ee.Number(fcOI.map(propDiff).reduceColumns(ee.Reducer.mean(), ['diff']).get('mean')).sqrt()

    return rmse

# MAE function
def MAE(fcOI,propertyOfInterest,propertyOfInterest_Predicted):
    # Compute the absolute difference between observed and predicted
    def propDiff(f):
        diff = ee.Number(f.get(propertyOfInterest)).subtract(ee.Number(f.get(propertyOfInterest_Predicted)))

        return f.set('diff', diff.abs())

    # calculate MAE from squared difference
    MAE = ee.Number(fcOI.map(propDiff).reduceColumns(ee.Reducer.mean(), ['diff']).get('mean'))

    return MAE

# Function to take a feature with a classifier of interest
def computeCVAccuracyAndRMSE(featureWithClassifier):
    # Pull the classifier from the feature
    cOI = ee.Classifier(featureWithClassifier.get('c'))

    # Get the model type
    modelType = cOI.mode().getInfo()

    # Create a function to map through the fold assignments and compute the overall accuracy
    # for all validation folds
    def computeAccuracyForFold(foldFeature):
        # Organize the training and validation data

        foldNumber = ee.Number(ee.Feature(foldFeature).get('Fold'))
        trainingData_Random = fcOI.filterMetadata(cvFoldString_Random,'not_equals',foldNumber)
        validationData_Random = fcOI.filterMetadata(cvFoldString_Random,'equals',foldNumber)

        trainingData_Spatial = fcOI.filterMetadata(cvFoldString_Spatial,'not_equals',foldNumber)
        validationData_Spatial = fcOI.filterMetadata(cvFoldString_Spatial,'equals',foldNumber)

        # Train the classifier and classify the validation dataset
        trainedClassifier_Random = cOI.train(trainingData_Random,classProperty,covariateList)
        outputtedPropName_Random = classProperty+'_Predicted_Random'
        classifiedValidationData_Random = validationData_Random.classify(trainedClassifier_Random,outputtedPropName_Random)

        trainedClassifier_Spatial = cOI.train(trainingData_Spatial,classProperty,covariateList)
        outputtedPropName_Spatial = classProperty+'_Predicted_Spatial'
        classifiedValidationData_Spatial = validationData_Spatial.classify(trainedClassifier_Spatial,outputtedPropName_Spatial)

        if modelType == 'CLASSIFICATION':
            # Compute the categorical levels of the class property
            categoricalLevels = ee.Dictionary(ee.FeatureCollection(fcOI).reduceColumns(ee.Reducer.frequencyHistogram(),[classProperty])).keys()  
            
            # Compute the overall accuracy of the classification
            errorMatrix_Random = classifiedValidationData_Random.errorMatrix(classProperty,outputtedPropName_Random,categoricalLevels)
            overallAccuracy_Random = ee.Number(errorMatrix_Random.accuracy())

            errorMatrix_Spatial = classifiedValidationData_Spatial.errorMatrix(classProperty,outputtedPropName_Spatial,categoricalLevels)
            overallAccuracy_Spatial = ee.Number(errorMatrix_Spatial.accuracy())
            return foldFeature.set('overallAccuracy_Random',overallAccuracy_Random).set('overallAccuracy_Spatial',overallAccuracy_Spatial)
        
        if modelType == 'REGRESSION':
            # Compute accuracy metrics
            r2ToSet_Random = coefficientOfDetermination(classifiedValidationData_Random,classProperty,outputtedPropName_Random)
            rmseToSet_Random = RMSE(classifiedValidationData_Random,classProperty,outputtedPropName_Random)
            maeToSet_Random = MAE(classifiedValidationData_Random,classProperty,outputtedPropName_Random)

            r2ToSet_Spatial = coefficientOfDetermination(classifiedValidationData_Spatial,classProperty,outputtedPropName_Spatial)
            rmseToSet_Spatial = RMSE(classifiedValidationData_Spatial,classProperty,outputtedPropName_Spatial)
            maeToSet_Spatial = MAE(classifiedValidationData_Spatial,classProperty,outputtedPropName_Spatial)
            return foldFeature.set('R2_Random',r2ToSet_Random).set('RMSE_Random', rmseToSet_Random).set('MAE_Random', maeToSet_Random)\
                                .set('R2_Spatial',r2ToSet_Spatial).set('RMSE_Spatial', rmseToSet_Spatial).set('MAE_Spatial', maeToSet_Spatial)

    # Compute the mean and std dev of the accuracy values of the classifier across all folds
    accuracyFC = kFoldAssignmentFC.map(computeAccuracyForFold)

    if modelType == 'REGRESSION':
        meanAccuracy_Random = accuracyFC.aggregate_mean('R2_Random')
        tsdAccuracy_Random = accuracyFC.aggregate_total_sd('R2_Random')
        meanAccuracy_Spatial = accuracyFC.aggregate_mean('R2_Spatial')
        tsdAccuracy_Spatial = accuracyFC.aggregate_total_sd('R2_Spatial')

        # Calculate mean and std dev of RMSE
        RMSEvals_Random = accuracyFC.aggregate_array('RMSE_Random')
        RMSEvalsSquared_Random = RMSEvals_Random.map(lambda f: ee.Number(f).multiply(f))
        sumOfRMSEvalsSquared_Random = RMSEvalsSquared_Random.reduce(ee.Reducer.sum())
        meanRMSE_Random = ee.Number.sqrt(ee.Number(sumOfRMSEvalsSquared_Random).divide(k))
        RMSEvals_Spatial = accuracyFC.aggregate_array('RMSE_Spatial')
        RMSEvalsSquared_Spatial = RMSEvals_Spatial.map(lambda f: ee.Number(f).multiply(f))
        sumOfRMSEvalsSquared_Spatial = RMSEvalsSquared_Spatial.reduce(ee.Reducer.sum())
        meanRMSE_Spatial = ee.Number.sqrt(ee.Number(sumOfRMSEvalsSquared_Spatial).divide(k))

        RMSEdiff_Random = accuracyFC.aggregate_array('RMSE_Random').map(lambda f: ee.Number(ee.Number(f).subtract(meanRMSE_Random)).pow(2))
        sumOfRMSEdiff_Random = RMSEdiff_Random.reduce(ee.Reducer.sum())
        sdRMSE_Random = ee.Number.sqrt(ee.Number(sumOfRMSEdiff_Random).divide(k))
        RMSEdiff_Spatial = accuracyFC.aggregate_array('RMSE_Spatial').map(lambda f: ee.Number(ee.Number(f).subtract(meanRMSE_Spatial)).pow(2))
        sumOfRMSEdiff_Spatial = RMSEdiff_Spatial.reduce(ee.Reducer.sum())
        sdRMSE_Spatial = ee.Number.sqrt(ee.Number(sumOfRMSEdiff_Spatial).divide(k))

        # Calculate mean and std dev of MAE
        meanMAE_Random = accuracyFC.aggregate_mean('MAE_Random')
        tsdMAE_Random= accuracyFC.aggregate_total_sd('MAE_Random')
        meanMAE_Spatial = accuracyFC.aggregate_mean('MAE_Spatial')
        tsdMAE_Spatial= accuracyFC.aggregate_total_sd('MAE_Spatial')

        # Compute the feature to return
        featureToReturn = featureWithClassifier.select(['cName']).set('Mean_R2_Random',meanAccuracy_Random,'StDev_R2_Random',tsdAccuracy_Random, 'Mean_RMSE_Random',meanRMSE_Random,'StDev_RMSE_Random',sdRMSE_Random, 'Mean_MAE_Random',meanMAE_Random,'StDev_MAE_Random',tsdMAE_Random)\
                                                                .set('Mean_R2_Spatial',meanAccuracy_Spatial,'StDev_R2_Spatial',tsdAccuracy_Spatial, 'Mean_RMSE_Spatial',meanRMSE_Spatial,'StDev_RMSE_Spatial',sdRMSE_Spatial, 'Mean_MAE_Spatial',meanMAE_Spatial,'StDev_MAE_Spatial',tsdMAE_Spatial)

    if modelType == 'CLASSIFICATION':
        accuracyFC_Random = kFoldAssignmentFC.map(computeAccuracyForFold)
        meanAccuracy_Random = accuracyFC_Random.aggregate_mean('overallAccuracy_Random')
        tsdAccuracy_Random = accuracyFC_Random.aggregate_total_sd('overallAccuracy_Random')
        accuracyFC_Spatial = kFoldAssignmentFC.map(computeAccuracyForFold)
        meanAccuracy_Spatial = accuracyFC_Spatial.aggregate_mean('overallAccuracy_Spatial')
        tsdAccuracy_Spatial = accuracyFC_Spatial.aggregate_total_sd('overallAccuracy_Spatial')

        # Compute the feature to return
        featureToReturn = featureWithClassifier.select(['cName']).set('Mean_overallAccuracy_Random',meanAccuracy_Random,'StDev_overallAccuracy_Random',tsdAccuracy_Random)\
                                                                .set('Mean_overallAccuracy_Spatial',meanAccuracy_Spatial,'StDev_overallAccuracy_Spatial',tsdAccuracy_Spatial)  

    return featureToReturn

fcOI = ee.FeatureCollection(assetIDForCVAssignedColl)
print(fcOI.size().getInfo(), 'features in', assetIDForCVAssignedColl)

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

kFoldAssignmentFC = ee.FeatureCollection(ee.List(kList).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('Fold',n)))

grid_search_results = ee.FeatureCollection([ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/hyperparameter_tuning/'+rf.get('cName').getInfo()) for rf in classifierList]).flatten()

# Get top 10 models
top_10Models = grid_search_results.limit(10, sort_acc_prop, False).aggregate_array('cName')


##################################################################################################################################################################
# Spatial Leave-One-Out cross validation
##################################################################################################################################################################
assetIDToCreate_Folder = 'projects/crowtherlab/johan/SPUN/EM_sloo_cv'
if any(x in subprocess.run(bashCommandList_Detect+[assetIDToCreate_Folder],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in stringsOfInterest) == False:
    pass
else:
    # perform the folder creation
    print(assetIDToCreate_Folder,'being created...')

    # Create the folder within Earth Engine
    subprocess.run(bashCommandList_CreateFolder+[assetIDToCreate_Folder])
    while any(x in subprocess.run(bashCommandList_Detect+[assetIDToCreate_Folder],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in stringsOfInterest):
        print('Waiting for asset to be created...')
        time.sleep(normalWaitTime)
    print('Asset created!')

    # Sleep to allow the server time to receive incoming requests
    time.sleep(normalWaitTime/2)


# !! NOTE: this is a fairly computatinally intensive excercise, so there are some precautions to take to ensure servers aren't overloaded
# !! This operaion SHOULD NOT be performed on the entire dataset

# Define buffer sizes to test (in meters)
buffer_sizes = [1000, 2500, 5000, 10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000]

# # Set number of random points to test
# if preppedCollection.shape[0] > 1000:
#     n_points = 1000 # Don't increase this value!
# else:
#     n_points = preppedCollection.shape[0]
n_points = 1000

# Set number of repetitions
n_reps = 10
nList = list(range(0,n_reps))

# Perform BLOO-CV
for rep in nList:
    for buffer in buffer_sizes:
        # mapList = []
        # for item in nList:
        #     mapList = mapList + (list(zip([buffer], repeat(item))))

        # Make a feature collection from the buffer sizes list
        # fc_toMap = ee.FeatureCollection(ee.List(mapList).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('buffer_size',ee.List(n).get(0)).set('rep',ee.List(n).get(1))))
        fc_toMap = ee.FeatureCollection(ee.Feature(ee.Geometry.Point([0,0])).set('buffer_size',buffer).set('rep',rep))

        grid_search_results = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_grid_search_results')

        # Get top model name
        bestModelName = grid_search_results.limit(1, 'Mean_R2', False).first().get('cName')

        # Get top 10 models
        top_10Models = grid_search_results.limit(10, 'Mean_R2', False).aggregate_array('cName')

        # Helper function 1: assess whether point is within sampled range
        def WithinRange(f):
            testFeature = f
            # Training FeatureCollection: all samples not within geometry of test feature
            trainFC = fcOI.filter(ee.Filter.geometry(f.geometry()).Not())

            # Make a FC with the band names
            fcWithBandNames = ee.FeatureCollection(ee.List(covariateList).map(lambda bandName: ee.Feature(None).set('BandName',bandName)))

            # Helper function 1b: assess whether training point is within sampled range; per band
            def getRange(f):
                bandBeingComputed = f.get('BandName')
                minValue = trainFC.aggregate_min(bandBeingComputed)
                maxValue = trainFC.aggregate_max(bandBeingComputed)
                testFeatureWithinRange = ee.Number(testFeature.get(bandBeingComputed)).gte(ee.Number(minValue)).bitwiseAnd(ee.Number(testFeature.get(bandBeingComputed)).lte(ee.Number(maxValue)))
                return f.set('within_range', testFeatureWithinRange)

            # Return value of 1 if all bands are within sampled range
            within_range = fcWithBandNames.map(getRange).aggregate_min('within_range')

            return f.set('within_range', within_range)

        # Helper function 1: Spatial Leave One Out cross-validation function:
        def BLOOcv(f):
            rep = f.get('rep')
            # Test feature
            testFeature = ee.FeatureCollection(f)

            # Training set: all samples not within geometry of test feature
            trainFC = fcOI.filter(ee.Filter.neq(classProperty, 0)).filter(ee.Filter.geometry(testFeature).Not())

            # Classifier to test: same hyperparameter settings as from grid search procedure
            classifierName = top_10Models.get(rep)
            classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', classifierName).first()).get('c'))

            # Train classifier
            trainedClassifer = classifier.train(trainFC, classProperty, covariateList)

            # Apply classifier
            classified = testFeature.classify(classifier = trainedClassifer, outputName = 'predicted')

            # Get predicted value
            predicted = classified.first().get('predicted')

            # Set predicted value to feature
            return f.set('predicted', predicted).copyProperties(f)

        # Helper function 2: R2 calculation function
        def calc_R2(f):
            rep = f.get('rep')
            # FeatureCollection holding the buffer radius
            buffer_size = f.get('buffer_size')

            # Sample 1000 validation points from the data
            fc_withRandom = fcOI.filter(ee.Filter.neq(classProperty, 0)).randomColumn(seed = rep)
            subsetData = fc_withRandom.sort('random').limit(n_points)

            # Add the iteration ID to the FC
            fc_toValidate = subsetData.map(lambda f: f.set('rep', rep))

            # Add the buffer around the validation data
            fc_wBuffer = fc_toValidate.map(lambda f: f.buffer(buffer_size))

            # Remove points not within sampled range
            fc_withinSampledRange = fc_wBuffer.map(WithinRange).filter(ee.Filter.eq('within_range', 1))

            # Apply blocked leave one out CV function
            predicted = fc_withinSampledRange.map(BLOOcv)
            # predicted = fc_wBuffer.map(BLOOcv)
            
            # Calculate R2 value
            R2_val = coefficientOfDetermination(predicted, classProperty, 'predicted')

            return(f.set('R2_val', R2_val))

        # Calculate R2 across range of buffer sizes
        sloo_cv = fc_toMap.map(calc_R2)

        # Export FC to assets
        bloo_cv_fc_export = ee.batch.Export.table.toAsset(
            collection = sloo_cv,
            description = classProperty+'_sloo_cv_results_woExtrapolation_'+str(buffer),
            assetId = 'projects/crowtherlab/johan/SPUN/EM_sloo_cv/'+classProperty+'_sloo_cv_results_wExtrapolation_'+str(buffer)+'_rep_'+str(rep),
        )

        # bloo_cv_fc_export.start()

    print('Blocked Leave-One-Out started! Moving on...')
