# Import the modules of interest
import pandas as pd
import numpy as np
import subprocess
import tqdm
import time
import datetime
import ee
from math import sqrt
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from itertools import combinations
from itertools import repeat
from pathlib import Path
from contextlib import contextmanager
import multiprocessing

ee.Initialize()

setup = 'distictObs_wProjectVars'
####################################################################################################################################################################
# Configuration
####################################################################################################################################################################
# Input the name of the username that serves as the home folder for asset storage
usernameFolderString = 'johanvandenhoogen'

# Input the Cloud Storage Bucket that will hold the bootstrap collections when uploading them to Earth Engine
# !! This bucket should be pre-created before running this script
bucketOfInterest = 'johanvandenhoogen'

# Input the name of the classification property
classProperty = 'AMF_diversity'

# Input the name of the project folder inside which all of the assets will be stored
# This folder will be generated automatically below, if it isn't yet present
projectFolder = '000_SPUN_zeroInflated'

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

compositeOfInterest = ee.Image('projects/crowtherlab/Composite/CrowtherLab_Composite_30ArcSec')

project_vars = [
# 'top',
# 'bot',
# 'core_length',
'target_marker',
'sequencing_platform',
'sample_type',
'primers'
]

# covariateList = covariateList + project_vars
if setup == 'wpixelAgg_wProjectVars':
	covariateList = covariateList + project_vars
	pixel_agg = True
	distinctObs = False
if setup == 'wpixelAgg_woProjectVars':
	covariateList = covariateList
	pixel_agg = True
	distinctObs = False
if setup == 'wopixelAgg_wProjectVars':
	covariateList = covariateList + project_vars
	pixel_agg = False
	distinctObs = False
if setup == 'wopixelAgg_woProjectVars':
	covariateList = covariateList
	pixel_agg = False
	distinctObs = False
if setup == 'distictObs_wProjectVars':
	covariateList = covariateList + project_vars
	pixel_agg = False
	distinctObs = True
if setup == 'distictObs_woProjectVars':
	covariateList = covariateList
	pixel_agg = False
	distinctObs = True

####################################################################################################################################################################
# Cross validation settings
####################################################################################################################################################################
# Metric to use for sorting k-fold CV hyperparameter tuning (default: R2)
sort_acc_prop = 'Mean_R2' # (either one of 'Mean_R2', 'Mean_MAE', 'Mean_RMSE')

# Set k for k-fold CV
k = 10

# Make a list of the k-fold CV assignments to use
kList = list(range(1,k+1))

# Set number of trees in RF models
nTrees = 250

# Input the name of the property that holds the CV fold assignment
cvFoldString = 'CV_Fold'

# Input the title of the CSV that will hold all of the data that has been given a CV fold assignment
titleOfCSVWithCVAssignments = '20220613_AMF_tedersoo_validationdata'

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

# Input the name of a folder used to hold the bootstrap collections
bootstrapCollFolder = 'Bootstrap_Collections'

# Input the header text that will name each bootstrapped dataset
fileNameHeader = classProperty+'BootstrapColl_'

# Stratification inputs
# Write the name of the variable used for stratification
# !! This variable should be included in the input dataset
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
bashFunction_EarthEngine = '/Users/johanvandenhoogen/opt/anaconda3/envs/ee/bin/earthengine'
bashFunctionGSUtil = '/Users/johanvandenhoogen/exec -l /bin/bash/google-cloud-sdk/bin/gsutil'

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
stringsOfInterest = ['Asset does not exist or is not accessible']

# Compose the arguments into lists that can be run via the subprocess module
bashCommandList_Detect = [bashFunction_EarthEngine]+arglist_Detect
bashCommandList_Delete = [bashFunction_EarthEngine]+arglist_Delete
bashCommandList_CreateCollection = [bashFunction_EarthEngine]+arglist_CreateCollection
bashCommandList_CreateFolder = [bashFunction_EarthEngine]+arglist_CreateFolder

####################################################################################################################################################################
# Helper functions
####################################################################################################################################################################
# Function to convert GEE FC to pd.DataFrame. Not ideal as it's calling .getInfo(), but does the job
def GEE_FC_to_pd(fc):
	result = []

	values = fc.toList(100000).getInfo()

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
# Data processing
####################################################################################################################################################################
# Import the raw CSV
rawPointCollection = pd.read_csv('data/20220613_all_taxa_tedersoo_Arbuscular_Mycorrhizal.csv', float_precision='round_trip')

# Rename columnto be mapped
rawPointCollection.rename(columns={'myco_diversity': classProperty}, inplace=True)

# Primers: ITS9mun and ITS4ngsuni
# sequencing_platform: PacBio  Sequel II
# sample_type: soil
# target_marker: ITS2
# FMS25266v2

GF_data = pd.read_csv('data/20211026_AMF_diversity_data_sampled.csv', float_precision='round_trip')
GF_data[GF_data['sequencing_platform'] == 'PacBio']


GF_data = GF_data.assign(sequencing_platform = (GF_data['sequencing_platform']).astype('category').cat.codes)
GF_data = GF_data.assign(sample_type = (GF_data['sample_type']).astype('category').cat.codes)
GF_data = GF_data.assign(primers = (GF_data['primers']).astype('category').cat.codes)
GF_data = GF_data.assign(target_marker = (GF_data['target_marker']).astype('category').cat.codes)
# Convert factors to integers
# rawPointCollection = rawPointCollection.assign(sequencing_platform = (rawPointCollection['sequencing_platform']).astype('category').cat.codes)
# rawPointCollection = rawPointCollection.assign(sample_type = (rawPointCollection['sample_type']).astype('category').cat.codes)
# rawPointCollection = rawPointCollection.assign(primers = (rawPointCollection['primers']).astype('category').cat.codes)
# rawPointCollection = rawPointCollection.assign(target_marker = (rawPointCollection['target_marker']).astype('category').cat.codes)
rawPointCollection['sequencing_platform'] = 1
rawPointCollection['sample_type'] = 1
rawPointCollection['primers'] = 1
rawPointCollection['target_marker'] = 1
# Print basic information on the csv

# Shuffle the data frame while setting a new index to ensure geographic clumps of points are not clumped in any way
fcToAggregate = rawPointCollection.sample(frac = 1, random_state = 42).reset_index(drop=True)

# Remove duplicates or pixel aggregate
if pixel_agg == True:
	preppedCollection = pd.DataFrame(fcToAggregate.groupby(['Pixel_Lat', 'Pixel_Long']).mean().to_records())[covariateList+["Resolve_Biome"]+[classProperty]+['Pixel_Lat', 'Pixel_Long']]

if distinctObs == True:
	preppedCollection = fcToAggregate.drop_duplicates(subset = covariateList+[classProperty], keep = False)[['sample_id']+covariateList+["Resolve_Biome"]+[classProperty]+['Pixel_Lat', 'Pixel_Long']]

if pixel_agg == False and distinctObs == False:
	preppedCollection = fcToAggregate[['sample_id']+covariateList+["Resolve_Biome"]+[classProperty]+['Pixel_Lat', 'Pixel_Long']]

print('Number of aggregated pixels', preppedCollection.shape[0])

# Drop NAs
preppedCollection = preppedCollection.dropna(how='any')
print('After dropping NAs', preppedCollection.shape[0])

# Log transform classProperty if specified
if log_transform_classProperty == True:
	preppedCollection[classProperty] = np.log(preppedCollection[classProperty] + 1)

# Convert biome column to int, to correct odd rounding errors
preppedCollection[stratificationVariableString] = preppedCollection[stratificationVariableString].astype(int)

# Add fold assignments to each of the points, stratified by biome
preppedCollection[cvFoldString] = (preppedCollection.groupby('Resolve_Biome').cumcount() % k) + 1

# Write the CSV to disk and upload it to Earth Engine as a Feature Collection
localPathToCVAssignedData = holdingFolder+'/'+titleOfCSVWithCVAssignments+'.csv'
preppedCollection.to_csv(localPathToCVAssignedData,index=False)

# Format the bash call to upload the file to the Google Cloud Storage bucket
gsutilBashUploadList = [bashFunctionGSUtil]+arglist_preGSUtilUploadFile+[localPathToCVAssignedData]+[formattedBucketOI]
subprocess.run(gsutilBashUploadList)
print(titleOfCSVWithCVAssignments+' uploaded to a GCSB!')

# Wait for a short period to ensure the command has been received by the server
time.sleep(normalWaitTime/2)

# Wait for the GSUTIL uploading process to finish before moving on
while not all(x in subprocess.run([bashFunctionGSUtil,'ls',formattedBucketOI],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in [titleOfCSVWithCVAssignments]):
	print('Not everything is uploaded...')
	time.sleep(normalWaitTime)
print('Everything is uploaded; moving on...')

# Upload the file into Earth Engine as a table asset
assetIDForCVAssignedColl = 'users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments
earthEngineUploadTableCommands = [bashFunction_EarthEngine]+arglist_preEEUploadTable+[assetIDStringPrefix+assetIDForCVAssignedColl]+[formattedBucketOI+'/'+titleOfCSVWithCVAssignments+'.csv']+arglist_postEEUploadTable
subprocess.run(earthEngineUploadTableCommands)
print('Upload to EE queued!')

# Wait for a short period to ensure the command has been received by the server
time.sleep(normalWaitTime/2)

# !! Break and wait
count = 1
while count >= 1:
	taskList = [str(i) for i in ee.batch.Task.list()]
	subsetList = [s for s in taskList if classProperty in s]
	subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
	count = len(subsubList)
	print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:', count)
	time.sleep(normalWaitTime)
print('Moving on...')

##################################################################################################################################################################
# Hyperparameter tuning
##################################################################################################################################################################
grid_search_resultsRegression = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+setup+'_grid_search_results_Regression')
grid_search_resultsClassification = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+setup+'_grid_search_results_Classification')

# Get top model name
bestModelNameRegression = grid_search_resultsRegression.limit(1, 'Mean_R2', False).first().get('cName')
bestModelNameClassification = grid_search_resultsClassification.limit(1, 'Mean_R2', False).first().get('cName')

# Get top 10 models
top_10ModelsRegression = grid_search_resultsRegression.limit(10, 'Mean_R2', False).aggregate_array('cName')
top_10ModelsClassification = grid_search_resultsClassification.limit(10, 'Mean_R2', False).aggregate_array('cName')

len(grid_search_resultsRegression.first().getInfo())
len(grid_search_resultsClassification.first().getInfo())


# Define hyperparameters for grid search
varsPerSplit_list = list(range(2,8))
leafPop_list = list(range(2,8))

classifierListRegression = []
# Create list of classifiers
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
# Create list of classifiers
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

##################################################################################################################################################################
# Predicted - Observed
##################################################################################################################################################################
fcOI_validate = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)
fcOI_train = ee.FeatureCollection('users/johanvandenhoogen/000_SPUN_zeroInflated/AMF_diversitydistictObs_wProjectVars_wCV_folds_data')

def predObsClassification(fcOI):
	if ensemble == False:
		# Load the best model from the classifier list
		classifierRegression = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListRegression).filterMetadata('cName', 'equals', bestModelNameRegression).first()).get('c'))
		classifierClassification = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListClassification).filterMetadata('cName', 'equals', bestModelNameClassification).first()).get('c'))

		# Train the classifier with the collection
		# REGRESSION
		fcOI_forRegression = fcOI_train.filter(ee.Filter.neq(classProperty, 0)).filter(ee.Filter.eq('source', 'GlobalFungi')) #  train classifier only on data not equalling zero / remove Tedersoo data
		trainedClassiferRegression = classifierRegression.train(fcOI_forRegression, classProperty, covariateList)

		# Classification
		fcOI_forClassification = fcOI_train.map(lambda f: f.set(classProperty+'_forClassification', ee.Number(f.get(classProperty)).divide(f.get(classProperty)))) # train classifier on 0 (classProperty == 0) or 1 (classProperty != 0)
		trainedClassiferClassification = classifierClassification.train(fcOI_forClassification, classProperty+'_forClassification', covariateList)

		# Classify the FC
		classifiedFC_Regression = fcOI_validate.classify(trainedClassiferRegression,classProperty+'_Regressed')
		classifiedFC_Classification = fcOI_validate.classify(trainedClassiferClassification,classProperty+'_Classified')

		# Join classified FCs
		filter = ee.Filter.equals(leftField = 'sample_id', rightField = 'sample_id')
		innerJoin = ee.Join.inner()
		classifiedFC = innerJoin.apply(classifiedFC_Regression, classifiedFC_Classification, filter)

		# Return as FC with properties
		classifiedFC = classifiedFC.map(lambda pair: ee.Feature(pair.get('primary')).set(ee.Feature(pair.get('secondary')).toDictionary()))

		# Calculate final predicted value as product of classification and regression
		classifiedFC = classifiedFC.map(lambda f: f.set(classProperty+'_Predicted', ee.Number(f.get(classProperty+'_Classified')).multiply(ee.Number(f.get(classProperty+'_Regressed')))))

	if ensemble == True:
		def classifyFC(classifiers):
			modelNameRegression = ee.List(classifiers).get(0)
			modelNameClassification = ee.List(classifiers).get(1)

			# Load the best model from the classifier list
			classifierRegression = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListRegression).filterMetadata('cName', 'equals', modelNameRegression).first()).get('c'))
			classifierClassification = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListClassification).filterMetadata('cName', 'equals', modelNameClassification).first()).get('c'))

			# Train the classifier with the collection
			# REGRESSION
			fcOI_forRegression = fcOI_train.filter(ee.Filter.neq(classProperty, 0)).filter(ee.Filter.eq('source', 'GlobalFungi')) #  train classifier only on data not equalling zero / remove Tedersoo data
			trainedClassiferRegression = classifierRegression.train(fcOI_forRegression, classProperty, covariateList)

			# Classification
			fcOI_forClassification = fcOI_train.map(lambda f: f.set(classProperty+'_forClassification', ee.Number(f.get(classProperty)).divide(f.get(classProperty)))) # train classifier on 0 (classProperty == 0) or 1 (classProperty != 0)
			trainedClassiferClassification = classifierClassification.train(fcOI_forClassification, classProperty+'_forClassification', covariateList)

			# Classify the FC
			# classifiedFC_Regression = fcOI.classify(trainedClassiferRegression,classProperty+'_Regressed')
			# classifiedFC_Classification = fcOI.classify(trainedClassiferClassification,classProperty+'_Classified')

			def classifyFunction(f):
				classfiedRegression = ee.FeatureCollection(f).classify(trainedClassiferRegression,classProperty+'_Regressed').first()
				classfiedClassification = ee.FeatureCollection(f).classify(trainedClassiferClassification,classProperty+'_Classified').first()

				featureToReturn = classfiedRegression.set(classProperty+'_Classified', classfiedClassification.get(classProperty+'_Classified'))
				return featureToReturn

			classifiedFC = fcOI.map(classifyFunction)

			#
			# # Join classified fc_sorted// Use an equals filter to specify how the collections match.
			# filter = ee.Filter.equals(leftField = 'system:index', rightField = 'system:index')
			#
			# # Define the join.
			# innerJoin = ee.Join.inner()
			#
			# # Apply the join.
			# classifiedFC = innerJoin.apply(classifiedFC_Regression, classifiedFC_Classification, filter)
			#
			# # Return as FC with properties
			# classifiedFC = classifiedFC.map(lambda pair: ee.Feature(pair.get('primary')).set(ee.Feature(pair.get('secondary')).toDictionary()))

			# Calculate final predicted value as product of classification and regression
			classifiedFC = classifiedFC.map(lambda f: f.set(classProperty+'_Predicted', ee.Number(f.get(classProperty+'_Classified')).multiply(ee.Number(f.get(classProperty+'_Regressed')))))

			return classifiedFC

		# Classify the FC
		classifiedFC = ee.FeatureCollection(top_10ModelsRegression.zip(top_10ModelsClassification).map(classifyFC)).flatten()

	return classifiedFC

# Classify FC
predObs = predObsClassification(fcOI_validate)

# Add coordinates to FC
predObs = predObs.map(addLatLon)

# Add residuals to FC
predObs_wResiduals = predObs.map(lambda f: f.set('AbsResidual', ee.Number(f.get(classProperty+'_Predicted')).subtract(f.get(classProperty)).abs()))

# Export to Assets
predObsexport = ee.batch.Export.table.toAsset(
	collection = predObs_wResiduals,
	description = '20220613_AMF_validation_wTedersoo',
	assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+'20220613_AMF_validation_wTedersoo'
)
predObsexport.start()

# !! Break and wait
count = 1
while count >= 1:
	taskList = [str(i) for i in ee.batch.Task.list()]
	subsetList = [s for s in taskList if '20220613_AMF_validation_wTedersoo' in s]
	count = len(subsubList)
	print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Waiting for pred/obs to complete...', end = '\r')
	time.sleep(normalWaitTime)
print('Moving on...')

predObs_wResiduals = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+'20220613_AMF_validation_wTedersoo')

# Convert to pd
predObs_df = GEE_FC_to_pd(predObs_wResiduals)

# back-log transform predicted and observed values
if log_transform_classProperty == True:
	predObs_df[classProperty+'_Predicted'] = np.exp(predObs_df[classProperty+'_Predicted']) - 1
	predObs_df[classProperty+'_Regressed'] = np.exp(predObs_df[classProperty+'_Regressed']) - 1
	predObs_df[classProperty+'_Classified'] = predObs_df[classProperty+'_Classified']
	predObs_df[classProperty] = np.exp(predObs_df[classProperty]) - 1
	predObs_df['AbsResidual'] = np.exp(predObs_df['AbsResidual'])

# Group by sample ID to return mean across ensemble prediction
if pixel_agg == False:
	predObs_df = pd.DataFrame(predObs_df.groupby('sample_id').mean().to_records())

# Write to file
if log_transform_classProperty == True:
	predObs_df.to_csv('output/20220613_AMF_validation_wTedersoo.csv')

if log_transform_classProperty == False:
	predObs_df.to_csv('output/20220613_AMF_validation_wTedersoo.csv')

print('done')
