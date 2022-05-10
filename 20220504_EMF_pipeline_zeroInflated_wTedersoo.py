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

covariateSet =[
# 'wpixelAgg_wProjectVars',
# 'wpixelAgg_woProjectVars',
'wopixelAgg_wProjectVars',
'wopixelAgg_woProjectVars',
'distictObs_wProjectVars',
'distictObs_woProjectVars',
]

def pipeline(setup):
	####################################################################################################################################################################
	# Configuration
	####################################################################################################################################################################
	# Input the name of the username that serves as the home folder for asset storage
	usernameFolderString = 'johanvandenhoogen'

	# Input the Cloud Storage Bucket that will hold the bootstrap collections when uploading them to Earth Engine
	# !! This bucket should be pre-created before running this script
	bucketOfInterest = 'johanvandenhoogen'

	# Input the name of the classification property
	classProperty = 'ECM_diversity'

	# Input the name of the project folder inside which all of the assets will be stored
	# This folder will be generated automatically below, if it isn't yet present
	projectFolder = '000_SPUN_temp_log_zeroInflated_wTedersoo'

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
	titleOfCSVWithCVAssignments = classProperty+setup+"_wCV_folds_data"

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
			trainingData = fcOI.filterMetadata(cvFoldString,'not_equals',foldNumber)
			validationData = fcOI.filterMetadata(cvFoldString,'equals',foldNumber)

			# Train the classifier and classify the validation dataset
			trainedClassifier = cOI.train(trainingData,classProperty,covariateList)
			outputtedPropName = classProperty+'_Predicted'
			classifiedValidationData = validationData.classify(trainedClassifier,outputtedPropName)

			if modelType == 'CLASSIFICATION':
				# Compute the overall accuracy of the classification
				errorMatrix = classifiedValidationData.errorMatrix(classProperty,outputtedPropName,categoricalLevels)
				overallAccuracy = ee.Number(errorMatrix.accuracy())
				return foldFeature.set('overallAccuracy',overallAccuracy)
			if modelType == 'REGRESSION':
				# Compute accuracy metrics
				r2ToSet = coefficientOfDetermination(classifiedValidationData,classProperty,outputtedPropName)
				rmseToSet = RMSE(classifiedValidationData,classProperty,outputtedPropName)
				maeToSet = MAE(classifiedValidationData,classProperty,outputtedPropName)
				return foldFeature.set('R2',r2ToSet).set('RMSE', rmseToSet).set('MAE', maeToSet)

		# Compute the mean and std dev of the accuracy values of the classifier across all folds
		accuracyFC = kFoldAssignmentFC.map(computeAccuracyForFold)

		if modelType == 'REGRESSION':
			meanAccuracy = accuracyFC.aggregate_mean('R2')
			tsdAccuracy = accuracyFC.aggregate_total_sd('R2')

			# Calculate mean and std dev of RMSE
			RMSEvals = accuracyFC.aggregate_array('RMSE')
			RMSEvalsSquared = RMSEvals.map(lambda f: ee.Number(f).multiply(f))
			sumOfRMSEvalsSquared = RMSEvalsSquared.reduce(ee.Reducer.sum())
			meanRMSE = ee.Number.sqrt(ee.Number(sumOfRMSEvalsSquared).divide(k))

			RMSEdiff = accuracyFC.aggregate_array('RMSE').map(lambda f: ee.Number(ee.Number(f).subtract(meanRMSE)).pow(2))
			sumOfRMSEdiff = RMSEdiff.reduce(ee.Reducer.sum())
			sdRMSE = ee.Number.sqrt(ee.Number(sumOfRMSEdiff).divide(k))

			# Calculate mean and std dev of MAE
			meanMAE = accuracyFC.aggregate_mean('MAE')
			tsdMAE= accuracyFC.aggregate_total_sd('MAE')

			# Compute the feature to return
			featureToReturn = featureWithClassifier.select(['cName']).set('Mean_R2',meanAccuracy,'StDev_R2',tsdAccuracy, 'Mean_RMSE',meanRMSE,'StDev_RMSE',sdRMSE, 'Mean_MAE',meanMAE,'StDev_MAE',tsdMAE)

		if modelType == 'CLASSIFICATION':
			accuracyFC = kFoldAssignmentFC.map(computeAccuracyForFold)
			meanAccuracy = accuracyFC.aggregate_mean('overallAccuracy')
			tsdAccuracy = accuracyFC.aggregate_total_sd('overallAccuracy')

			# Compute the feature to return
			featureToReturn = featureWithClassifier.select(['cName']).set('Mean_overallAccuracy',meanAccuracy,'StDev_overallAccuracy',tsdAccuracy)

		return featureToReturn

	####################################################################################################################################################################
	# Initialization
	####################################################################################################################################################################
	# Turn the folder string into an assetID
	assetIDToCreate_Folder = 'users/'+usernameFolderString+'/'+projectFolder
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

	####################################################################################################################################################################
	# Data processing
	####################################################################################################################################################################
	try:
		# try whether fcOI is present
		fcOI = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)

		fcOI.size().getInfo()

	except Exception as e:

		# Import the raw CSV
		GF_data = pd.read_csv('data/20211026_ECM_diversity_data_sampled.csv', float_precision='round_trip')
		GF_data['source'] = 'GlobalFungi'
		tedersoo_data = pd.read_csv('data/20220509_all_taxa_tedersoo_Ectomycorrhizal_sampled.csv', float_precision='round_trip')
		tedersoo_data['source'] = 'Tedersoo'

		rawPointCollection = pd.concat([GF_data, tedersoo_data])

		# Rename columnto be mapped
		rawPointCollection.rename(columns={'myco_diversity': classProperty}, inplace=True)

		# Convert factors to integers
		rawPointCollection = rawPointCollection.assign(sequencing_platform = (rawPointCollection['sequencing_platform']).astype('category').cat.codes)
		rawPointCollection = rawPointCollection.assign(sample_type = (rawPointCollection['sample_type']).astype('category').cat.codes)
		rawPointCollection = rawPointCollection.assign(primers = (rawPointCollection['primers']).astype('category').cat.codes)
		rawPointCollection = rawPointCollection.assign(target_marker = (rawPointCollection['target_marker']).astype('category').cat.codes)

		# Print basic information on the csv
		print('Original Collection', rawPointCollection.shape[0])

		# Shuffle the data frame while setting a new index to ensure geographic clumps of points are not clumped in any way
		fcToAggregate = rawPointCollection.sample(frac = 1, random_state = 42).reset_index(drop=True)

		# Remove duplicates or pixel aggregate
		if pixel_agg == True:
			preppedCollection = pd.DataFrame(fcToAggregate.groupby(['Pixel_Lat', 'Pixel_Long']).mean().to_records())[covariateList+["Resolve_Biome"]+[classProperty]+['Pixel_Lat', 'Pixel_Long']+['source']]

		if distinctObs == True:
			preppedCollection = fcToAggregate.drop_duplicates(subset = covariateList+[classProperty], keep = False)[['sample_id']+covariateList+["Resolve_Biome"]+[classProperty]+['Pixel_Lat', 'Pixel_Long']+['source']]

		if pixel_agg == False and distinctObs == False:
			preppedCollection = fcToAggregate[['sample_id']+covariateList+["Resolve_Biome"]+[classProperty]+['Pixel_Lat', 'Pixel_Long']+['source']]

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

	# ##################################################################################################################################################################
	# # Hyperparameter tuning
	# ##################################################################################################################################################################
	# fcOI = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)
	#
	# # Define hyperparameters for grid search
	# varsPerSplit_list = list(range(2,8))
	# leafPop_list = list(range(4,8))
	# classifierList = []
	#
	# # Create list of classifiers
	# for vps in varsPerSplit_list:
	# 	for lp in leafPop_list:
	#
	# 		model_name = classProperty + '_rf_VPS' + str(vps) + '_LP' + str(lp)
	#
	# 		rf = ee.Feature(ee.Geometry.Point([0,0])).set('cName',model_name,'c',ee.Classifier.smileRandomForest(
	# 		numberOfTrees = nTrees,
	# 		variablesPerSplit = vps,
	# 		minLeafPopulation = lp,
	# 		bagFraction = 0.632,
	# 		seed = 42
	# 		).setOutputMode('REGRESSION'))
	#
	# 		classifierList.append(rf)
	#
	# try:
	# 	# Grid search results as FC
	# 	grid_search_results = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+setup+'grid_search_results_woAggregation')
	#
	# 	# Get top model name
	# 	bestModelName = grid_search_results.limit(1, 'Mean_R2', False).first().get('cName')
	#
	# 	# Get top 10 models
	# 	top_10Models = grid_search_results.limit(10, 'Mean_R2', False).aggregate_array('cName')
	#
	# 	len(grid_search_results.first().getInfo())
	# except Exception as e:
	# #     # Make a feature collection from the k-fold assignment list
	# #     kFoldAssignmentFC = ee.FeatureCollection(ee.List(kList).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('Fold',n)))
	# #
	# #     # Perform grid search
	# #     hyperparameter_tuning = ee.FeatureCollection(list(map(computeCVAccuracyAndRMSE,classifierList)))
	# #
	# #     # Export to assets
	# #     gridSearchExport = ee.batch.Export.table.toAsset(
	# #     	collection = hyperparameter_tuning,
	# #     	description = classProperty+'grid_search_results',
	# #     	assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'grid_search_results_woAggregation'
	# #     )
	# #     gridSearchExport.start()
	# #
	# #     # !! Break and wait
	# #     count = 1
	# #     while count >= 1:
	# #     	taskList = [str(i) for i in ee.batch.Task.list()]
	# #     	subsetList = [s for s in taskList if classProperty in s]
	# #     	subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
	# #     	count = len(subsubList)
	# #     	print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Waiting for grid search to complete...')
	# #     	time.sleep(normalWaitTime)
	# #     print('Moving on...')
	# #
	# #     # Grid search results as FC
	# #     grid_search_results = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'grid_search_results_woAggregation')
	#
	# 	# Make a feature collection from the k-fold assignment list
	# 	kFoldAssignmentFC = ee.FeatureCollection(ee.List(kList).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('Fold',n)))
	#
	# 	classDf = pd.DataFrame(columns = ['Mean_R2', 'StDev_R2','Mean_RMSE', 'StDev_RMSE','Mean_MAE', 'StDev_MAE', 'cName'])
	#
	# 	for rf in classifierList:
	# 		print('Testing model', classifierList.index(rf), 'out of total of', len(classifierList))
	#
	# 		accuracy_feature = ee.Feature(computeCVAccuracyAndRMSE(rf))
	#
	# 		classDf = classDf.append(pd.DataFrame(accuracy_feature.getInfo()['properties'], index = [0]))
	#
	# 	classDfSorted = classDf.sort_values([sort_acc_prop], ascending = False)
	#
	# 	# Write model results to csv
	# 	classDfSorted.to_csv('output/'+classProperty+setup+'_grid_search_results_tmp.csv', index=False)
	#
	# 	# Format the bash call to upload the file to the Google Cloud Storage bucket
	# 	gsutilBashUploadList = [bashFunctionGSUtil]+arglist_preGSUtilUploadFile+['output/'+classProperty+setup+'_grid_search_results_tmp.csv']+[formattedBucketOI]
	# 	subprocess.run(gsutilBashUploadList)
	# 	print('grid_search_results'+' uploaded to a GCSB!')
	#
	# 	# Wait for a short period to ensure the command has been received by the server
	# 	time.sleep(normalWaitTime/2)
	#
	# 	# Wait for the GSUTIL uploading process to finish before moving on
	# 	while not all(x in subprocess.run([bashFunctionGSUtil,'ls',formattedBucketOI],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in ['_grid_search_results_tmp']):
	# 		print('Not everything is uploaded...')
	# 		time.sleep(normalWaitTime)
	# 	print('Everything is uploaded; moving on...')
	#
	# 	# Upload the file into Earth Engine as a table asset
	# 	assetIdForGridSearchResults = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+setup+'grid_search_results_woAggregation'
	# 	earthEngineUploadTableCommands = [bashFunction_EarthEngine]+arglist_preEEUploadTable+[assetIDStringPrefix+assetIdForGridSearchResults]+[formattedBucketOI+'/'+classProperty+setup+'_grid_search_results_tmp.csv']+arglist_postEEUploadTable
	# 	subprocess.run(earthEngineUploadTableCommands)
	# 	print('Upload to EE queued!')
	#
	# 	# Wait for a short period to ensure the command has been received by the server
	# 	time.sleep(normalWaitTime/2)
	#
	# 	# !! Break and wait
	# 	count = 1
	# 	while count >= 1:
	# 		taskList = [str(i) for i in ee.batch.Task.list()]
	# 		subsetList = [s for s in taskList if classProperty in s]
	# 		subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
	# 		count = len(subsubList)
	# 		print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:', count)
	# 		time.sleep(normalWaitTime)
	# 	print('Moving on...')
	#
	# 	# Grid search results as FC
	# 	grid_search_results = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+setup+'grid_search_results_woAggregation')
	#
	# 	# Get top model name
	# 	bestModelName = grid_search_results.limit(1, 'Mean_R2', False).first().get('cName')
	#
	# 	# Get top 10 models
	# 	top_10Models = grid_search_results.limit(10, 'Mean_R2', False).aggregate_array('cName')

	##################################################################################################################################################################
	# Hyperparameter tuning
	##################################################################################################################################################################
	fcOI = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)

	# Define hyperparameters for grid search
	varsPerSplit_list = list(range(2,8))
	leafPop_list = list(range(4,8))

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
	try:
		# Grid search results as FC
		grid_search_resultsRegression = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+setup+'_grid_search_results_Regression').filter(ee.Filter.eq('source', 'GlobalFungi'))
		grid_search_resultsClassification = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+setup+'_grid_search_results_Classification')

		# Get top model name
		bestModelNameRegression = grid_search_resultsRegression.limit(1, 'Mean_R2', False).first().get('cName')
		bestModelNameClassification = grid_search_resultsClassification.limit(1, 'Mean_R2', False).first().get('cName')

		# Get top 10 models
		top_10ModelsRegression = grid_search_resultsRegression.limit(10, 'Mean_R2', False).aggregate_array('cName')
		top_10ModelsClassification = grid_search_resultsClassification.limit(10, 'Mean_R2', False).aggregate_array('cName')

		len(grid_search_resultsRegression.first().getInfo())
		len(grid_search_resultsClassification.first().getInfo())

	except Exception as e:
		# Make a feature collection from the k-fold assignment list
		kFoldAssignmentFC = ee.FeatureCollection(ee.List(kList).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('Fold',n)))

		classDfRegression = pd.DataFrame(columns = ['Mean_R2', 'StDev_R2','Mean_RMSE', 'StDev_RMSE','Mean_MAE', 'StDev_MAE', 'cName'])
		classDfClassification = pd.DataFrame(columns = ['Mean_overallAccuracy', 'StDev_overallAccuracy', 'cName'])

		for rf in classifierListRegression:
			print('Testing model', classifierListRegression.index(rf), 'out of total of', len(classifierListRegression))

			#  train classifier only on data not equalling zero
			# train classifier only on GlobalFungi data
			fcOI = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)\
					.filter(ee.Filter.neq(classProperty, 0))\
					.filter(ee.Filter.eq('source', 'GlobalFungi'))
			accuracy_feature = ee.Feature(computeCVAccuracyAndRMSE(rf))

			classDfRegression = classDfRegression.append(pd.DataFrame(accuracy_feature.getInfo()['properties'], index = [0]))

		for rf in classifierListClassification:
			print('Testing model', classifierListClassification.index(rf), 'out of total of', len(classifierListClassification))

			fcOI = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)
			fcOI = fcOI.map(lambda f: f.set(classProperty, ee.Number(f.get(classProperty)).divide(f.get(classProperty)))) # train classifier on 0 (classProperty == 0) or 1 (classProperty != 0)
			categoricalLevels = fcOI.aggregate_array(classProperty).distinct().getInfo()

			accuracy_feature = ee.Feature(computeCVAccuracyAndRMSE(rf))

			classDfClassification = classDfClassification.append(pd.DataFrame(accuracy_feature.getInfo()['properties'], index = [0]))

		classDfSortedRegression = classDfRegression.sort_values([sort_acc_prop], ascending = False)
		classDfSortedClassification = classDfClassification.sort_values(['Mean_overallAccuracy'], ascending = False) #Sorting Classification model by R2 doens't make sense - test accuracy instead

		# Write model results to csv
		classDfSortedRegression.to_csv('output/'+classProperty+setup+'_grid_search_results_Regression_zeroInflated.csv', index=False)
		classDfSortedClassification.to_csv('output/'+classProperty+setup+'_grid_search_results_Classification_zeroInflated.csv', index=False)

		# Format the bash call to upload the file to the Google Cloud Storage bucket
		gsutilBashUploadList = [bashFunctionGSUtil]+arglist_preGSUtilUploadFile+['output/'+classProperty+setup+'_grid_search_results_Regression_zeroInflated.csv']+[formattedBucketOI]
		subprocess.run(gsutilBashUploadList)
		print('grid_search_results'+' uploaded to a GCSB!')

		gsutilBashUploadList = [bashFunctionGSUtil]+arglist_preGSUtilUploadFile+['output/'+classProperty+setup+'_grid_search_results_Classification_zeroInflated.csv']+[formattedBucketOI]
		subprocess.run(gsutilBashUploadList)
		print('grid_search_results'+' uploaded to a GCSB!')

		# Wait for a short period to ensure the command has been received by the server
		time.sleep(normalWaitTime/2)

		# Wait for the GSUTIL uploading process to finish before moving on
		while not all(x in subprocess.run([bashFunctionGSUtil,'ls',formattedBucketOI],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in ['_grid_search_results_Regression']):
			print('Not everything is uploaded...')
			time.sleep(normalWaitTime)
		print('Everything is uploaded; moving on...')

		while not all(x in subprocess.run([bashFunctionGSUtil,'ls',formattedBucketOI],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in ['_grid_search_results_Classification']):
			print('Not everything is uploaded...')
			time.sleep(normalWaitTime)
		print('Everything is uploaded; moving on...')

		# Upload the file into Earth Engine as a table asset
		assetIdForGridSearchResults = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+setup+'_grid_search_results_Regression'
		earthEngineUploadTableCommands = [bashFunction_EarthEngine]+arglist_preEEUploadTable+[assetIDStringPrefix+assetIdForGridSearchResults]+[formattedBucketOI+'/'+classProperty+setup+'_grid_search_results_Regression.csv']+arglist_postEEUploadTable
		subprocess.run(earthEngineUploadTableCommands)
		assetIdForGridSearchResults = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+setup+'_grid_search_results_Classification'
		earthEngineUploadTableCommands = [bashFunction_EarthEngine]+arglist_preEEUploadTable+[assetIDStringPrefix+assetIdForGridSearchResults]+[formattedBucketOI+'/'+classProperty+setup+'_grid_search_results_Classification.csv']+arglist_postEEUploadTable
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

		# Grid search results as FC
		grid_search_resultsRegression = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+setup+'_grid_search_results_Regression')
		grid_search_resultsClassification = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+setup+'_grid_search_results_Classification')

		# Get top model name
		bestModelNameRegression = grid_search_resultsRegression.limit(1, 'Mean_R2', False).first().get('cName')
		bestModelNameClassification = grid_search_resultsClassification.limit(1, 'Mean_overallAccuracy', False).first().get('cName')

		# Get top 10 models
		top_10ModelsRegression = grid_search_resultsRegression.limit(10, 'Mean_R2', False).aggregate_array('cName')
		top_10ModelsClassification = grid_search_resultsClassification.limit(10, 'Mean_overallAccuracy', False).aggregate_array('cName')

		print('Moving on...')

	# # Write grid search results to csv
	# GEE_FC_to_pd(grid_search_results.limit(10, 'Mean_R2', False)).to_csv('output/'+classProperty+'_grid_search_results.csv')

	##################################################################################################################################################################
	# Classify image
	##################################################################################################################################################################
	# Reference covariate levels for mapping:
	# top: 0
	# bot: 10
	# core.length: 10
	# Sample.type: soil
	# target_marker: Illumina
	# target_marker: ITS2
	# Primers: ITS3/ITS4

	# Sample FMS17564v2 has the reference levels:
	# rawPointCollection[rawPointCollection['sample_id'] == 'FMS17564v2']

	# Project-specific variables
	# top = ee.Image.constant(0)
	# bot = ee.Image.constant(10)
	# corelength = ee.Image.constant(10)
	# target_marker = ee.Image.constant(int(rawPointCollection[rawPointCollection['sample_id'] == 'FMS17564v2']['target_marker']))
	# sequencing_platform = ee.Image.constant(int(rawPointCollection[rawPointCollection['sample_id'] == 'FMS17564v2']['sequencing_platform']))
	# sample_type = ee.Image.constant(int(rawPointCollection[rawPointCollection['sample_id'] == 'FMS17564v2']['sample_type']))
	# primers = ee.Image.constant(int(rawPointCollection[rawPointCollection['sample_id'] == 'FMS17564v2']['primers']))
	#
	# constant_imgs = ee.ImageCollection.fromImages([target_marker, sequencing_platform, sample_type, primers]).toBands().rename(['target_marker', 'sequencing_platform', 'sample_type', 'primers'])
	#
	# def finalImageClassification(compositeImg):
	# 	if ensemble == False:
	# 		# Load the best model from the classifier list
	# 		classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', bestModelName).first()).get('c'))
	#
	# 		# Train the classifier with the collection
	# 		trainedClassifer = classifier.train(fcOI, classProperty, covariateList)
	#
	# 		# Classify the image
	# 		classifiedImage = compositeImg.classify(trainedClassifer,classProperty+'_Predicted')
	#
	# 	if ensemble == True:
	# 		def classifyImage(classifierName):
	# 			# Load the best model from the classifier list
	# 			classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', classifierName).first()).get('c'))
	#
	# 			# Train the classifier with the collection
	# 			trainedClassifer = classifier.train(fcOI, classProperty, covariateList)
	#
	# 			# Classify the image
	# 			classifiedImage = compositeImg.classify(trainedClassifer,classProperty+'_Predicted')
	#
	# 			return classifiedImage
	#
	# 		# Classify the images, return mean
	# 		classifiedImage = ee.ImageCollection(top_10Models.map(classifyImage)).mean()
	#
	# 	return classifiedImage
	#
	# # Create appropriate composite image with bands to use
	# if setup == 'wpixelAgg_wProjectVars':
	# 	compositeToClassify = compositeOfInterest.addBands(constant_imgs).select(covariateList).reproject(compositeOfInterest.projection())
	# if setup == 'wpixelAgg_woProjectVars':
	# 	compositeToClassify = compositeOfInterest
	# if setup == 'wopixelAgg_wProjectVars':
	# 	compositeToClassify = compositeOfInterest.addBands(constant_imgs).select(covariateList).reproject(compositeOfInterest.projection())
	# if setup == 'wopixelAgg_woProjectVars':
	# 	compositeToClassify = compositeOfInterest
	# if setup == 'distictObs_wProjectVars':
	# 	compositeToClassify = compositeOfInterest.addBands(constant_imgs).select(covariateList).reproject(compositeOfInterest.projection())
	# if setup == 'distictObs_woProjectVars':
	# 	compositeToClassify = compositeOfInterest

	# image_toExport = finalImageClassification(compositeToClassify)

	# imgExport = ee.batch.Export.image.toAsset(
	#     image = image_toExport.toFloat(),
	#     description = classProperty+'classifiedImg_wFuturePreds',
	#     assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'classifiedImg_wFuturePreds' ,
	#     crs = 'EPSG:4326',
	#     crsTransform = '[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
	#     region = exportingGeometry,
	#     maxPixels = int(1e13),
	#     pyramidingPolicy = {".default": pyramidingPolicy}
	# )
	# imgExport.start()
	#
	# print('Image export started')


	##################################################################################################################################################################
	##################################################################################################################################################################
	##################################################################################################################################################################
	##################################################################################################################################################################
	# Predicted - Observed
	##################################################################################################################################################################
	fcOI = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)

	# TEMP - first we actually only really care about predicted-observed
	try:
		predObs_wResiduals = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_pred_obs_zeroInflated_'+setup+'_logTransformed')
		predObs_wResiduals.size().getInfo()

	except Exception as e:
		def predObsClassification(fcOI):
			if ensemble == False:
				# Load the best model from the classifier list
				classifierRegression = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListRegression).filterMetadata('cName', 'equals', bestModelNameRegression).first()).get('c'))
				classifierClassification = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListClassification).filterMetadata('cName', 'equals', bestModelNameClassification).first()).get('c'))

				# Train the classifier with the collection
				# REGRESSION
				fcOI_forRegression = fcOI.filter(ee.Filter.neq(classProperty, 0)).filter(ee.Filter.eq('source', 'GlobalFungi')) #  train classifier only on data not equalling zero / remove Tedersoo data
				trainedClassiferRegression = classifierRegression.train(fcOI_forRegression, classProperty, covariateList)

				# Classification
				fcOI_forClassification = fcOI.map(lambda f: f.set(classProperty+'_forClassification', ee.Number(f.get('ECM_diversity')).divide(f.get('ECM_diversity')))) # train classifier on 0 (classProperty == 0) or 1 (classProperty != 0)
				trainedClassiferClassification = classifierClassification.train(fcOI_forClassification, classProperty+'_forClassification', covariateList)

				# Classify the FC
				classifiedFC_Regression = fcOI.classify(trainedClassiferRegression,classProperty+'_Regressed')
				classifiedFC_Classification = fcOI.classify(trainedClassiferClassification,classProperty+'_Classified')

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
					fcOI_forRegression = fcOI.filter(ee.Filter.neq(classProperty, 0)).filter(ee.Filter.eq('source', 'GlobalFungi')) #  train classifier only on data not equalling zero / remove Tedersoo data
					trainedClassiferRegression = classifierRegression.train(fcOI_forRegression, classProperty, covariateList)

					# Classification
					fcOI_forClassification = fcOI.map(lambda f: f.set(classProperty+'_forClassification', ee.Number(f.get('ECM_diversity')).divide(f.get('ECM_diversity')))) # train classifier on 0 (classProperty == 0) or 1 (classProperty != 0)
					trainedClassiferClassification = classifierClassification.train(fcOI_forClassification, classProperty+'_forClassification', covariateList)

					# Classify the FC
					classifiedFC_Regression = fcOI.classify(trainedClassiferRegression,classProperty+'_Regressed')
					classifiedFC_Classification = fcOI.classify(trainedClassiferClassification,classProperty+'_Classified')

					# Join classified fc_sorted// Use an equals filter to specify how the collections match.
					filter = ee.Filter.equals(leftField = 'sample_id', rightField = 'sample_id')

					# Define the join.
					innerJoin = ee.Join.inner()

					# Apply the join.
					classifiedFC = innerJoin.apply(classifiedFC_Regression, classifiedFC_Classification, filter)

					# Return as FC with properties
					classifiedFC = classifiedFC.map(lambda pair: ee.Feature(pair.get('primary')).set(ee.Feature(pair.get('secondary')).toDictionary()))

					# Calculate final predicted value as product of classification and regression
					classifiedFC = classifiedFC.map(lambda f: f.set(classProperty+'_Predicted', ee.Number(f.get(classProperty+'_Classified')).multiply(ee.Number(f.get(classProperty+'_Regressed')))))

					return classifiedFC

				# Classify the FC
				classifiedFC = ee.FeatureCollection(top_10ModelsRegression.zip(top_10ModelsClassification).map(classifyFC)).flatten()

			return classifiedFC

		# Classify FC
		predObs = predObsClassification(fcOI).filter(ee.Filter.eq('source', 'GlobalFungi'))

		# Add coordinates to FC
		predObs = predObs.map(addLatLon)

		# Add residuals to FC
		predObs_wResiduals = predObs.map(lambda f: f.set('AbsResidual', ee.Number(f.get(classProperty+'_Predicted')).subtract(f.get(classProperty)).abs()))

		# Export to Assets
		predObsexport = ee.batch.Export.table.toAsset(
			collection = predObs_wResiduals,
			description = classProperty+'_pred_obs_zeroInflated_'+setup+'_logTransformed',
			assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_pred_obs_zeroInflated_'+setup+'_logTransformed'
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

		predObs_wResiduals = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_pred_obs_zeroInflated_'+setup+'_logTransformed')

	# Convert to pd
	predObs_df = GEE_FC_to_pd(predObs_wResiduals)

	# back-log transform predicted and observed values
	if log_transform_classProperty == True:
		predObs_df[classProperty+'_Predicted'] = np.exp(predObs_df[classProperty+'_Predicted']) - 1
		predObs_df[classProperty+'_Regressed'] = np.exp(predObs_df[classProperty+'_Regressed']) - 1
		predObs_df[classProperty+'_Classified'] = np.exp(predObs_df[classProperty+'_Classified']) - 1
		predObs_df[classProperty] = np.exp(predObs_df[classProperty]) - 1
		predObs_df['AbsResidual'] = np.exp(predObs_df['AbsResidual'])

	# Group by sample ID to return mean across ensemble prediction
	if pixel_agg == False:
		predObs_df = pd.DataFrame(predObs_df.groupby('sample_id').mean().to_records())

	# Write to file
	if log_transform_classProperty == True:
		predObs_df.to_csv('output/20220504_'+classProperty+'_pred_obs_zeroInflated_'+setup+'_logTransformed'+'_wTedersoo'+'.csv')

	if log_transform_classProperty == False:
		predObs_df.to_csv('output/20220504_'+classProperty+'_pred_obs_zeroInflated_'+setup+'_wo_logTransformed'+'_wTedersoo'+'.csv')

	print('done')

	# ##################################################################################################################################################################
	# # Variable importance metrics
	# ##################################################################################################################################################################
	# if ensemble == False:
	# 	classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', bestModelName).first()).get('c'))
	#
	# 	# Train the classifier with the collection
	# 	trainedClassifer = classifier.train(fcOI, classProperty, covariateList)
	#
	# 	# Get the feature importance from the trained classifier and write to a .csv file and as a bar plot as .png file
	# 	featureImportances = trainedClassifer.explain().get('importance').getInfo()
	#
	# 	featureImportances = pd.DataFrame(featureImportances.items(),
	# 									  columns=['Variable', 'Feature_Importance']).sort_values(by='Feature_Importance',
	# 																								ascending=False)
	#
	# 	# Scale values
	# 	featureImportances['Feature_Importance'] = featureImportances['Feature_Importance'] - featureImportances['Feature_Importance'].min()
	# 	featureImportances['Feature_Importance'] = featureImportances['Feature_Importance'] / featureImportances['Feature_Importance'].max()
	#
	# if ensemble == True:
	# 	# Instantiate empty dataframe
	# 	featureImportances = pd.DataFrame(columns=['Variable', 'Feature_Importance'])
	#
	# 	for i in list(range(0,10)):
	# 		classifierName = top_10Models.get(i)
	# 		classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', classifierName).first()).get('c'))
	#
	# 		# Train the classifier with the collection
	# 		trainedClassifer = classifier.train(fcOI, classProperty, covariateList)
	#
	# 		# Get the feature importance from the trained classifier and write to a .csv file and as a bar plot as .png file
	# 		featureImportancesToAdd = trainedClassifer.explain().get('importance').getInfo()
	# 		featureImportancesToAdd = pd.DataFrame(featureImportancesToAdd.items(),
	# 										  columns=['Variable', 'Feature_Importance']).sort_values(by='Feature_Importance',
	# 																									ascending=False)
	#
	# 		# Scale values
	# 		featureImportancesToAdd['Feature_Importance'] = featureImportancesToAdd['Feature_Importance'] - featureImportancesToAdd['Feature_Importance'].min()
	# 		featureImportancesToAdd['Feature_Importance'] = featureImportancesToAdd['Feature_Importance'] / featureImportancesToAdd['Feature_Importance'].max()
	#
	# 		featureImportances = pd.concat([featureImportances, featureImportancesToAdd])
	#
	# 	featureImportances = pd.DataFrame(featureImportances.groupby('Variable').mean().to_records())
	#
	# # Write to csv
	# featureImportances.to_csv('output/'+classProperty+'_featureImportances.csv')
	# featureImportances.sort_values('Feature_Importance', ascending = False ,inplace = True)
	#
	# # Create and save plot
	# plt = featureImportances[:10].plot(x='Variable', y='Feature_Importance', kind='bar', legend=False,
	# 							  title='Feature Importances')
	# fig = plt.get_figure()
	# fig.savefig('output/'+classProperty+'_FeatureImportances.png', bbox_inches='tight')
	#
	# print('Variable importance metrics complete! Moving on...')

	# ##################################################################################################################################################################
	# # Bootstrapping
	# ##################################################################################################################################################################
	# # Input the number of points to use for each bootstrap model: equal to number of observations in training dataset
	# bootstrapModelSize = preppedCollection.shape[0]
	#
	# # Run a for loop to create multiple bootstrap iterations and upload them to the Google Cloud Storage Bucket
	# # Create an empty list to store all of the file name strings being uploaded (for later use)
	# fileNameList = []
	# for n in seedsToUseForBootstrapping:
	# 	# Perform the subsetting
	# 	stratSample = preppedCollection.groupby(stratificationVariableString, group_keys=False).apply(lambda x: x.sample(n=int(round((strataDict.get(x.name)/100)*bootstrapModelSize)), replace=True, random_state=n))
	#
	# 	# Format the title of the CSV and export it to a holding location
	# 	titleOfBootstrapCSV = fileNameHeader+str(n).zfill(3)
	# 	fileNameList.append(titleOfBootstrapCSV)
	#
	# # Load the best model from the classifier list
	# classifierToBootstrap = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName','equals',bestModelName).first()).get('c'))
	#
	# # Create empty list to store all fcs
	# fcList = []
	# # Run a for loop to create multiple bootstrap iterations
	# for n in seedsToUseForBootstrapping:
	#
	# 	# Format the title of the CSV and export it to a holding location
	# 	titleOfColl = fileNameHeader+str(n).zfill(3)
	# 	collectionPath = 'users/'+usernameFolderString+'/'+projectFolder+'/'+bootstrapCollFolder+'/'+titleOfColl
	#
	# 	# Load the collection from the path
	# 	fcToTrain = ee.FeatureCollection(collectionPath)
	#
	# 	# Append fc to list
	# 	fcList.append(fcToTrain)
	#
	# pred_climate_current = staticCompositeImg.addBands(climate_2015).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
	# pred_futureClimate_rcp26_2050 = staticCompositeImg.addBands(climate_rcp26_2050).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
	# pred_futureClimate_rcp26_2070 = staticCompositeImg.addBands(climate_rcp26_2070).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
	# pred_futureClimate_rcp45_2050 = staticCompositeImg.addBands(climate_rcp45_2050).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
	# pred_futureClimate_rcp45_2070 = staticCompositeImg.addBands(climate_rcp45_2070).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
	# pred_futureClimate_rcp60_2050 = staticCompositeImg.addBands(climate_rcp60_2050).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
	# pred_futureClimate_rcp60_2070 = staticCompositeImg.addBands(climate_rcp60_2070).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
	# pred_futureClimate_rcp85_2050 = staticCompositeImg.addBands(climate_rcp85_2050).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
	# pred_futureClimate_rcp85_2070 = staticCompositeImg.addBands(climate_rcp85_2070).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
	#
	# compositeList = [pred_climate_current,
	# pred_futureClimate_rcp26_2050,
	# pred_futureClimate_rcp26_2070,
	# pred_futureClimate_rcp45_2050,
	# pred_futureClimate_rcp45_2070,
	# pred_futureClimate_rcp60_2050,
	# pred_futureClimate_rcp60_2070,
	# pred_futureClimate_rcp85_2050,
	# pred_futureClimate_rcp85_2070]
	#
	# names = ['climate_current',
	# 'futureClimate_rcp26_2050',
	# 'futureClimate_rcp26_2070',
	# 'futureClimate_rcp45_2050',
	# 'futureClimate_rcp45_2070',
	# 'futureClimate_rcp60_2050',
	# 'futureClimate_rcp60_2070',
	# 'futureClimate_rcp85_2050',
	# 'futureClimate_rcp85_2070']
	#
	# dictToBoostrap = {compositeList[i]: names[i] for i in range(len(names))}
	#
	# for key, value in dictToBoostrap.items():
	# 	composite = key
	#
	# 	# Helper fucntion to train a RF classifier and classify the composite image
	# 	def bootstrapFunc(fc):
	# 		# Train the classifier with the collection
	# 		trainedClassifer = classifierToBootstrap.train(fc,classProperty,covariateList)
	#
	# 		# Classify the image
	# 		classifiedImage = composite.classify(trainedClassifer,classProperty+'_Predicted')
	#
	# 		return classifiedImage
	#
	#
	# 	# Reduce bootstrap images to mean
	# 	meanImage = ee.ImageCollection.fromImages(list(map(bootstrapFunc, fcList))).reduce(
	# 		reducer = ee.Reducer.mean()
	# 	)
	#
	# 	# Reduce bootstrap images to lower and upper CIs
	# 	upperLowerCIImage = ee.ImageCollection.fromImages(list(map(bootstrapFunc, fcList))).reduce(
	# 		reducer = ee.Reducer.percentile([2.5,97.5],['lower','upper'])
	# 	)
	#
	# 	# Reduce bootstrap images to standard deviation
	# 	stdDevImage = ee.ImageCollection.fromImages(list(map(bootstrapFunc, fcList))).reduce(
	# 		reducer = ee.Reducer.stdDev()
	# 	)
	#
	# 	imageToExport = ee.Image.cat(meanImage,
	# 		upperLowerCIImage,
	# 		stdDevImage).rename(['bootstrapped_mean_'+value,
	# 							 'bootstrapped_lower_'+value,
	# 							 'bootstrapped_upper_'+value,
	# 							 'bootstrapped_stdDev_'+value])
	#
	# 	boostrapExport = ee.batch.Export.image.toAsset(
	# 		image = imageToExport.toFloat(),
	# 		description = classProperty+'_bootstrapped_'+value,
	# 		assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_bootstrapped_'+value,
	# 		crs = 'EPSG:4326',
	# 		crsTransform = '[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
	# 		region = exportingGeometry,
	# 		maxPixels = int(1e13),
	# 		pyramidingPolicy = {".default": pyramidingPolicy}
	# 	)
	# 	boostrapExport.start()
	#
	# ##################################################################################################################################################################
	# # Univariate int-ext analysis
	# ##################################################################################################################################################################
	# # Create a feature collection with only the values from the image bands
	# fcForMinMax = fcOI.select(covariateList)
	#
	# def univariateIntExt(composite):
	# 	compositeForIntExt = composite.addBands(constant_imgs).select(covariateList).reproject(composite.projection())
	#
	# 	# Make a FC with the band names
	# 	fcWithBandNames = ee.FeatureCollection(ee.List(covariateList).map(lambda bandName: ee.Feature(None).set('BandName',bandName)))
	#
	# 	def calcMinMax(f):
	# 	  bandBeingComputed = f.get('BandName')
	# 	  maxValueToSet = fcForMinMax.reduceColumns(ee.Reducer.minMax(),[bandBeingComputed])
	# 	  return f.set('MinValue',maxValueToSet.get('min')).set('MaxValue',maxValueToSet.get('max'))
	#
	# 	# Map function
	# 	fcWithMinMaxValues = ee.FeatureCollection(fcWithBandNames).map(calcMinMax)
	#
	# 	# Make two images from these values (a min and a max image)
	# 	maxValuesWNulls = fcWithMinMaxValues.toList(1000).map(lambda f: ee.Feature(f).get('MaxValue'))
	# 	maxDict = ee.Dictionary.fromLists(covariateList,maxValuesWNulls)
	# 	minValuesWNulls = fcWithMinMaxValues.toList(1000).map(lambda f: ee.Feature(f).get('MinValue'))
	# 	minDict = ee.Dictionary.fromLists(covariateList,minValuesWNulls)
	# 	minImage = minDict.toImage()
	# 	maxImage = maxDict.toImage()
	#
	# 	totalBandsBinary = compositeForIntExt.gte(minImage.select(covariateList)).lt(maxImage.select(covariateList))
	# 	univariate_int_ext_image = totalBandsBinary.reduce('sum').divide(compositeForIntExt.bandNames().length()).rename('univariate_pct_int_ext')
	#
	# 	return univariate_int_ext_image
	#
	# univariate_int_ext_image = ee.ImageCollection.fromImages(list(map(univariateIntExt, compositeList))).toBands().rename(['int_ext_current',
	# 'int_ext_rcp26_2050',
	# 'int_ext_rcp26_2070',
	# 'int_ext_rcp45_2050',
	# 'int_ext_rcp45_2070',
	# 'int_ext_rcp60_2050',
	# 'int_ext_rcp60_2070',
	# 'int_ext_rcp85_2050',
	# 'int_ext_rcp85_2070'])
	#
	# univariate_int_ext_export = ee.batch.Export.image.toAsset(
	# 	image = univariate_int_ext_image.toFloat(),
	# 	description = classProperty+'_univariate_int_ext',
	# 	assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_univariate_int_ext',
	# 	crs = 'EPSG:4326',
	# 	crsTransform = '[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
	# 	region = exportingGeometry,
	# 	maxPixels = int(1e13),
	# 	pyramidingPolicy = {".default": pyramidingPolicy}
	# )
	# # univariate_int_ext_export.start()
	#
	#
	# ##################################################################################################################################################################
	# # Multivariate (PCA) int-ext analysis
	# ##################################################################################################################################################################
	#
	# # Input the proportion of variance that you would like to cover
	# propOfVariance = 90
	#
	# def PCA_int_extFunc(compositeForIntExt):
	#
	# 	# PCA interpolation/extrapolation helper function
	# 	def assessExtrapolation(fcOfInterest, propOfVariance):
	# 		# Compute the mean and standard deviation of each band, then standardize the point data
	# 		meanVector = fcOfInterest.mean()
	# 		stdVector = fcOfInterest.std()
	# 		standardizedData = (fcOfInterest-meanVector)/stdVector
	#
	# 		# Then standardize the composite from which the points were sampled
	# 		meanList = meanVector.tolist()
	# 		stdList = stdVector.tolist()
	# 		bandNames = list(meanVector.index)
	# 		meanImage = ee.Image(meanList).rename(bandNames)
	# 		stdImage = ee.Image(stdList).rename(bandNames)
	# 		standardizedImage = compositeForIntExt.subtract(meanImage).divide(stdImage)
	#
	# 		# Run a PCA on the point samples
	# 		pcaOutput = PCA()
	# 		pcaOutput.fit(standardizedData)
	#
	# 		# Save the cumulative variance represented by each PC
	# 		cumulativeVariance = np.cumsum(np.round(pcaOutput.explained_variance_ratio_, decimals=4)*100)
	#
	# 		# Make a list of PC names for future organizational purposes
	# 		pcNames = ['PC'+str(x) for x in range(1,fcOfInterest.shape[1]+1)]
	#
	# 		# Get the PC loadings as a data frame
	# 		loadingsDF = pd.DataFrame(pcaOutput.components_,columns=[str(x)+'_Loads' for x in bandNames],index=pcNames)
	#
	# 		# Get the original data transformed into PC space
	# 		transformedData = pd.DataFrame(pcaOutput.fit_transform(standardizedData,standardizedData),columns=pcNames)
	#
	# 		# Make principal components images, multiplying the standardized image by each of the eigenvectors
	# 		# Collect each one of the images in a single image collection
	#
	# 		# First step: make an image collection wherein each image is a PC loadings image
	# 		listOfLoadings = ee.List(loadingsDF.values.tolist())
	# 		eePCNames = ee.List(pcNames)
	# 		zippedList = eePCNames.zip(listOfLoadings)
	# 		def makeLoadingsImage(zippedValue):
	# 			return ee.Image.constant(ee.List(zippedValue).get(1)).rename(bandNames).set('PC',ee.List(zippedValue).get(0))
	# 		loadingsImageCollection = ee.ImageCollection(zippedList.map(makeLoadingsImage))
	#
	# 		# Second step: multiply each of the loadings image by the standardized image and reduce it using a "sum"
	# 		# to finalize the matrix multiplication
	# 		def finalizePCImages(loadingsImage):
	# 			PCName = ee.String(ee.Image(loadingsImage).get('PC'))
	# 			return ee.Image(loadingsImage).multiply(standardizedImage).reduce('sum').rename([PCName]).set('PC',PCName)
	# 		principalComponentsImages = loadingsImageCollection.map(finalizePCImages)
	#
	# 		# Choose how many principal components are of interest in this analysis based on amount of
	# 		# variance explained
	# 		numberOfComponents = sum(i < propOfVariance for i in cumulativeVariance)+1
	# 		print('Number of Principal Components being used:',numberOfComponents)
	#
	# 		# Compute the combinations of the principal components being used to compute the 2-D convex hulls
	# 		tupleCombinations = list(combinations(list(pcNames[0:numberOfComponents]),2))
	# 		print('Number of Combinations being used:',len(tupleCombinations))
	#
	# 		# Generate convex hulls for an example of the principal components of interest
	# 		cHullCoordsList = list()
	# 		for c in tupleCombinations:
	# 			firstPC = c[0]
	# 			secondPC = c[1]
	# 			outputCHull = ConvexHull(transformedData[[firstPC,secondPC]])
	# 			listOfCoordinates = transformedData.loc[outputCHull.vertices][[firstPC,secondPC]].values.tolist()
	# 			flattenedList = [val for sublist in listOfCoordinates for val in sublist]
	# 			cHullCoordsList.append(flattenedList)
	#
	# 		# Reformat the image collection to an image with band names that can be selected programmatically
	# 		pcImage = principalComponentsImages.toBands().rename(pcNames)
	#
	# 		# Generate an image collection with each PC selected with it's matching PC
	# 		listOfPCs = ee.List(tupleCombinations)
	# 		listOfCHullCoords = ee.List(cHullCoordsList)
	# 		zippedListPCsAndCHulls = listOfPCs.zip(listOfCHullCoords)
	#
	# 		def makeToClassifyImages(zippedListPCsAndCHulls):
	# 			imageToClassify = pcImage.select(ee.List(zippedListPCsAndCHulls).get(0)).set('CHullCoords',ee.List(zippedListPCsAndCHulls).get(1))
	# 			classifiedImage = imageToClassify.rename('u','v').classify(ee.Classifier.spectralRegion([imageToClassify.get('CHullCoords')]))
	# 			return classifiedImage
	#
	# 		classifedImages = ee.ImageCollection(zippedListPCsAndCHulls.map(makeToClassifyImages))
	# 		finalImageToExport = classifedImages.sum().divide(ee.Image.constant(len(tupleCombinations)))
	#
	# 		return finalImageToExport
	#
	# 	# PCA interpolation-extrapolation image
	# 	PCA_int_ext = assessExtrapolation(preppedCollection[covariateList], propOfVariance).rename('PCA_pct_int_ext')
	# 	return PCA_int_ext
	#
	# PCA_int_ext = ee.ImageCollection.fromImages(list(map(PCA_int_extFunc, compositeList))).toBands().rename(['PCA_int_ext_current',
	# 'PCA_int_ext_rcp26_2050',
	# 'PCA_int_ext_rcp26_2070',
	# 'PCA_int_ext_rcp45_2050',
	# 'PCA_int_ext_rcp45_2070',
	# 'PCA_int_ext_rcp60_2050',
	# 'PCA_int_ext_rcp60_2070',
	# 'PCA_int_ext_rcp85_2050',
	# 'PCA_int_ext_rcp85_2070'])
	#
	# PCA_int_ext_export = ee.batch.Export.image.toAsset(
	# 	image = PCA_int_ext.toFloat(),
	# 	description = classProperty+'_PCA_int_ext',
	# 	assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_PCA_int_ext',
	# 	crs = 'EPSG:4326',
	# 	crsTransform = '[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
	# 	region = exportingGeometry,
	# 	maxPixels = int(1e13),
	# 	pyramidingPolicy = {".default": pyramidingPolicy}
	# )
	# # PCA_int_ext_export.start()
	#
	# ##################################################################################################################################################################
	# # Final image export
	# ##################################################################################################################################################################
	#
	# # Construct final image to export
	# if log_transform_classProperty == True:
	# 	finalImageToExport = ee.Image.cat(image_toExport.exp(),
	# 	meanImage.exp(),
	# 	upperLowerCIImage.exp(),
	# 	stdDevImage.exp(),
	# 	# univariate_int_ext_image,
	# 	# PCA_int_ext
	# 	)
	# else:
	# 	finalImageToExport = ee.Image.cat(image_toExport,
	# 	meanImage,
	# 	upperLowerCIImage,
	# 	stdDevImage,
	# 	# univariate_int_ext_image,
	# 	# PCA_int_ext
	# 	)
	#
	# FinalImageExport = ee.batch.Export.image.toAsset(
	# 	image = finalImageToExport.toFloat(),
	# 	description = classProperty+'_Bootstrapped_MultibandImage',
	# 	# assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_Classified_MultibandImage_wPrimers',
	# 	assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_Classified_MultibandImage_wFuturePreds',
	# 	crs = 'EPSG:4326',
	# 	crsTransform = '[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
	# 	region = exportingGeometry,
	# 	maxPixels = int(1e13),
	# 	pyramidingPolicy = {".default": pyramidingPolicy}
	# )
	# FinalImageExport.start()
	#
	# print('Map exports started! Moving on...')
	#
	# ##################################################################################################################################################################
	# # Spatial Leave-One-Out cross validation
	# ##################################################################################################################################################################
	#
	# # !! NOTE: this is a fairly computatinally intensive excercise, so there are some precautions to take to ensure servers aren't overloaded
	# # !! This operaion SHOULD NOT be performed on the entire dataset
	#
	# # Set number of random points to test
	# if preppedCollection.shape[0] > 1000:
	# 	n_points = 1000 # Don't increase this value!
	# else:
	# 	n_points = preppedCollection.shape[0]
	#
	# # Set number of repetitions
	# n_reps = 10
	# nList = list(range(0,n_reps))
	#
	# if buffer_size == list():
	# 	# create list with species + thresholds
	# 	mapList = []
	# 	for item in nList:
	# 		mapList = mapList + (list(zip(buffer_size, repeat(item))))
	#
	# 	# Make a feature collection from the buffer sizes list
	# 	fc_toMap = ee.FeatureCollection(ee.List(mapList).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('buffer_size',ee.List(n).get(0)).set('rep',ee.List(n).get(1))))
	#
	# else:
	# 	fc_toMap = ee.FeatureCollection(ee.List(nList).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('buffer_size',buffer_size).set('rep',n)))
	#
	# # Helper function 1: assess whether point is within sampled range
	# def WithinRange(f):
	# 	testFeature = f
	# 	# Training FeatureCollection: all samples not within geometry of test feature
	# 	trainFC = fcOI.filter(ee.Filter.geometry(f.geometry()).Not())
	#
	# 	# Make a FC with the band names
	# 	fcWithBandNames = ee.FeatureCollection(ee.List(covariateList).map(lambda bandName: ee.Feature(None).set('BandName',bandName)))
	#
	# 	# Helper function 1b: assess whether training point is within sampled range; per band
	# 	def getRange(f):
	# 		bandBeingComputed = f.get('BandName')
	# 		minValue = trainFC.aggregate_min(bandBeingComputed)
	# 		maxValue = trainFC.aggregate_max(bandBeingComputed)
	# 		testFeatureWithinRange = ee.Number(testFeature.get(bandBeingComputed)).gte(ee.Number(minValue)).bitwiseAnd(ee.Number(testFeature.get(bandBeingComputed)).lte(ee.Number(maxValue)))
	# 		return f.set('within_range', testFeatureWithinRange)
	#
	# 	# Return value of 1 if all bands are within sampled range
	# 	within_range = fcWithBandNames.map(getRange).aggregate_min('within_range')
	#
	# 	return f.set('within_range', within_range)
	#
	# # Helper function 2: Spatial Leave One Out cross-validation function:
	# def BLOOcv(f):
	# 	# Get iteration ID
	# 	rep = f.get('rep')
	#
	# 	# Test feature
	# 	testFeature = ee.FeatureCollection(f)
	#
	# 	# Training FeatureCollection: all samples not within geometry of test feature
	# 	trainFC = fcOI.filter(ee.Filter.geometry(testFeature).Not())
	#
	# 	if ensemble == False:
	# 		# Classifier to test: same hyperparameter settings as top model from grid search procedure
	# 		classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', bestModelName).first()).get('c'))
	#
	# 	if ensemble == True:
	# 		# Classifiers to test: top 10 models from grid search hyperparameter tuning
	# 		classifierName = top_10Models.get(rep)
	# 		classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', classifierName).first()).get('c'))
	#
	# 	# Train classifier
	# 	trainedClassifer = classifier.train(trainFC, classProperty, covariateList)
	#
	# 	# Apply classifier
	# 	classified = testFeature.classify(classifier = trainedClassifer, outputName = 'predicted')
	#
	# 	# Get predicted value
	# 	predicted = classified.first().get('predicted')
	#
	# 	# Set predicted value to feature
	# 	return f.set('predicted', predicted).copyProperties(f)
	#
	# # Helper function 3: R2 calculation function
	# def calc_R2(f):
	# 	# Get iteration ID
	# 	rep = f.get('rep')
	#
	# 	# FeatureCollection holding the buffer radius
	# 	buffer_size = f.get('buffer_size')
	#
	# 	# Sample 1000 validation points from the data
	# 	subsetData = fcOI.randomColumn(seed = rep).sort('random').limit(n_points)
	#
	# 	# Add the buffer around the validation data
	# 	fc_wBuffer = subsetData.map(lambda f: f.buffer(buffer_size))
	#
	# 	# Add the iteration ID to the FC
	# 	fc_toValidate = fc_wBuffer.map(lambda f: f.set('rep', rep))
	#
	# 	if loo_cv_wPointRemoval == True:
	# 		# Remove points not within sampled range
	# 		fc_withinSampledRange = fc_toValidate.map(WithinRange).filter(ee.Filter.eq('within_range', 1))
	#
	# 		# Apply blocked leave one out CV function
	# 		predicted = fc_withinSampledRange.map(BLOOcv)
	#
	# 		# outputName = '_sloo_cv_results_woExtrapolation'
	#
	# 	if loo_cv_wPointRemoval == False:
	# 		# Apply blocked leave one out CV function
	# 		predicted = fc_toValidate.map(BLOOcv)
	#
	# 		# outputName = '_sloo_cv_results_wExtrapolation'
	#
	# 	# Calculate R2 value
	# 	R2_val = coefficientOfDetermination(predicted, classProperty, 'predicted')
	#
	# 	return f.set('R2_val', R2_val)
	#
	# # Calculate R2 across range of buffer sizes
	# sloo_cv = fc_toMap.map(calc_R2)
	#
	# # Export FC to assets
	# bloo_cv_fc_export = ee.batch.Export.table.toAsset(
	# 	collection = sloo_cv,
	# 	description = classProperty+'_sloo_cv',
	# 	assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_sloo_cv_results_wExtrapolation'
	# )
	# bloo_cv_fc_export.start()
	#
	# print('Blocked Leave-One-Out started! Moving on...')
	#
	#
	# print('All tasks started! Output files will apear in this folder: users/'+usernameFolderString+'/'+projectFolder)

number_of_processes = 6

@contextmanager
def poolcontext(*args, **kwargs):
		"""This just makes the multiprocessing easier with a generator."""
		pool = multiprocessing.Pool(*args, **kwargs)
		yield pool
		pool.terminate()

if __name__ == '__main__':

		with poolcontext(number_of_processes) as pool:

				results = pool.map(pipeline, covariateSet)
