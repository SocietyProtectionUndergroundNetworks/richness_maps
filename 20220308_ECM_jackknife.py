# Import the modules of interest
import pandas as pd
import numpy as np
import subprocess
import time
import datetime
import ee
from pathlib import Path

ee.Initialize()

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
projectFolder = '000_SPUN/diversity'

# Input the normal wait time (in seconds) for "wait and break" cells
normalWaitTime = 5

# Input a longer wait time (in seconds) for "wait and break" cells
longWaitTime = 10

# Specify the column names where the latitude and longitude information is stored
latString = 'Pixel_Lat'
longString = 'Pixel_Long'

# Log transform classProperty? Boolean, either True or False
log_transform_classProperty = False

# Ensemble of top 10 models?
ensemble = True

# Spatial leave-one-out cross-validation settings
# skip test points outside training space after removing points in buffer zone? This might reduce extrapolation but overestimate accuracy
loo_cv_wPointRemoval = False

# Define buffer size in meters; use Moran's I or other test to determine SAC range
# Alternatively: specify buffer size as list, to test across multiple buffer sizes
buffer_size = 0

####################################################################################################################################################################
# Covariate data settings
####################################################################################################################################################################


climate_vars = [
"CHELSA_BIO_Annual_Mean_Temperature",
"CHELSA_BIO_Annual_Precipitation",
"CHELSA_BIO_Max_Temperature_of_Warmest_Month",
"CHELSA_BIO_Precipitation_of_Coldest_Quarter",
"CHELSA_BIO_Precipitation_Seasonality",
]

composite_vars = [
"CGIAR_Aridity_Index",
"CGIAR_PET",
"EarthEnvTopoMed_AspectCosine",
"EarthEnvTopoMed_AspectSine",
"EarthEnvTopoMed_Elevation",
"EarthEnvTopoMed_Slope",
"EarthEnvTopoMed_TopoPositionIndex",
"EsaCci_BurntAreasProbability",
"FanEtAl_Depth_to_Water_Table_AnnualMean",
"GlobPermafrost_PermafrostExtent",
"PelletierEtAl_SoilAndSedimentaryDepositThicknesses",
"SG_Depth_to_bedrock",
"SG_Sand_Content_005cm",
"SG_SOC_Content_005cm",
"SG_Soil_pH_H2O_005cm",
]

landUse_vars = [
"NET",
"NDT",
"BET",
"BDT",
"BES",
"BDS",
"C3",
"C4",
"Crops",
"Urban",
"Barren"
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

covariateList = climate_vars + composite_vars + project_vars

climate_2015 = ee.Image('projects/crowtherlab/t3/CHELSA/CHELSA_BioClim_1994_2013_180ArcSec').select(climate_vars)

# Future climate scenarios
climate_rcp26_2050 = ee.Image('projects/crowtherlab/t3/CHELSA/Future_BioClim_Ensembles/rcp26_2050s_mean')
climate_rcp26_2070 = ee.Image('projects/crowtherlab/t3/CHELSA/Future_BioClim_Ensembles/rcp26_2070s_mean')
climate_rcp45_2050 = ee.Image('projects/crowtherlab/t3/CHELSA/Future_BioClim_Ensembles/rcp45_2050s_mean')
climate_rcp45_2070 = ee.Image('projects/crowtherlab/t3/CHELSA/Future_BioClim_Ensembles/rcp45_2070s_mean')
climate_rcp60_2050 = ee.Image('projects/crowtherlab/t3/CHELSA/Future_BioClim_Ensembles/rcp60_2050s_mean')
climate_rcp60_2070 = ee.Image('projects/crowtherlab/t3/CHELSA/Future_BioClim_Ensembles/rcp60_2070s_mean')
climate_rcp85_2050 = ee.Image('projects/crowtherlab/t3/CHELSA/Future_BioClim_Ensembles/rcp85_2050s_mean')
climate_rcp85_2070 = ee.Image('projects/crowtherlab/t3/CHELSA/Future_BioClim_Ensembles/rcp85_2070s_mean')

# Future land use
ssp1_rcp26_2015_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp1_rcp26_2015_mean_11PFTs')
ssp1_rcp26_2060_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp1_rcp26_2060_mean_11PFTs')
ssp1_rcp26_2080_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp1_rcp26_2080_mean_11PFTs')
ssp2_rcp45_2015_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp2_rcp45_2015_mean_11PFTs')
ssp2_rcp45_2060_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp2_rcp45_2060_mean_11PFTs')
ssp2_rcp45_2080_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp2_rcp45_2080_mean_11PFTs')
ssp3_rcp60_2015_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp3_rcp60_2015_mean_11PFTs')
ssp3_rcp60_2060_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp3_rcp60_2060_mean_11PFTs')
ssp3_rcp60_2080_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp3_rcp60_2080_mean_11PFTs')
ssp4_rcp60_2015_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp4_rcp60_2015_mean_11PFTs')
ssp4_rcp60_2060_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp4_rcp60_2060_mean_11PFTs')
ssp4_rcp60_2080_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp4_rcp60_2080_mean_11PFTs')
ssp5_rcp85_2015_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp5_rcp85_2015_mean_11PFTs')
ssp5_rcp85_2060_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp5_rcp85_2060_mean_11PFTs')
ssp5_rcp85_2080_mean_11PFTs = ee.Image('projects/crowtherlab/t3/GCAM_Demeter/HarmonizedLandUse_11PFTs/ssp5_rcp85_2080_mean_11PFTs')

staticCompositeImg = ee.Image('projects/crowtherlab/Composite/CrowtherLab_Composite_30ArcSec').select(composite_vars)

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
nTrees = 500

# Input the name of the property that holds the CV fold assignment
cvFoldString = 'CV_Fold'

# Input the title of the CSV that will hold all of the data that has been given a CV fold assignment
titleOfCSVWithCVAssignments = classProperty+"_wCV_folds_data"

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
# Data processing
####################################################################################################################################################################
# Import the raw CSV
rawPointCollection = pd.read_csv('data/20211026_ECM_diversity_data_sampled.csv', float_precision='round_trip')

# Rename columnto be mapped
rawPointCollection.rename(columns={'myco_diversity': classProperty}, inplace=True)

# Convert factors to integers
rawPointCollection = rawPointCollection.assign(sequencing_platform = (rawPointCollection['sequencing_platform']).astype('category').cat.codes)
rawPointCollection = rawPointCollection.assign(sample_type = (rawPointCollection['sample_type']).astype('category').cat.codes)
rawPointCollection = rawPointCollection.assign(primers = (rawPointCollection['primers']).astype('category').cat.codes)
rawPointCollection = rawPointCollection.assign(target_marker = (rawPointCollection['target_marker']).astype('category').cat.codes)

##################################################################################################################################################################
# Hyperparameter tuning
##################################################################################################################################################################
fcOI = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)

# Define hyperparameters for grid search
varsPerSplit_list = list(range(2,8))
leafPop_list = list(range(4,8))
classifierList = []

# Create list of classifiers
for vps in varsPerSplit_list:
	for lp in leafPop_list:

		model_name = classProperty + '_rf_VPS' + str(vps) + '_LP' + str(lp)

		rf = ee.Feature(ee.Geometry.Point([0,0])).set('cName',model_name,'c',ee.Classifier.smileRandomForest(
		numberOfTrees = nTrees,
		variablesPerSplit = vps,
		minLeafPopulation = lp,
		bagFraction = 0.632,
		seed = 42
		).setOutputMode('REGRESSION'))

		classifierList.append(rf)

try:
	# Grid search results as FC
	grid_search_results = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'grid_search_results')

	# Get top model name
	bestModelName = grid_search_results.limit(1, 'Mean_R2', False).first().get('cName')

	# Get top 10 models
	top_10Models = grid_search_results.limit(10, 'Mean_R2', False).aggregate_array('cName')

	len(grid_search_results.first().getInfo())
except Exception as e:
	# Make a feature collection from the k-fold assignment list
	kFoldAssignmentFC = ee.FeatureCollection(ee.List(kList).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('Fold',n)))

	# Perform grid search
	hyperparameter_tuning = ee.FeatureCollection(list(map(computeCVAccuracyAndRMSE,classifierList)))

	# Export to assets
	gridSearchExport = ee.batch.Export.table.toAsset(
		collection = hyperparameter_tuning,
		description = classProperty+'grid_search_results',
		assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'grid_search_results'
	)
	gridSearchExport.start()

	# !! Break and wait
	count = 1
	while count >= 1:
		taskList = [str(i) for i in ee.batch.Task.list()]
		subsetList = [s for s in taskList if classProperty in s]
		subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
		count = len(subsubList)
		print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Waiting for grid search to complete...')
		time.sleep(normalWaitTime)
	print('Moving on...')

	# Grid search results as FC
	grid_search_results = ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'grid_search_results')

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
# rawPointCollection[rawPointCollection['sampleID'] == 'FMS17564v2']

# platform_id / type_id / primer_id images
# top = ee.Image.constant(0)
# bot = ee.Image.constant(10)
# corelength = ee.Image.constant(10)
target_marker = ee.Image.constant(int(rawPointCollection[rawPointCollection['sample_id'] == 'FMS17564v2']['target_marker']))
sequencing_platform = ee.Image.constant(int(rawPointCollection[rawPointCollection['sample_id'] == 'FMS17564v2']['sequencing_platform']))
sample_type = ee.Image.constant(int(rawPointCollection[rawPointCollection['sample_id'] == 'FMS17564v2']['sample_type']))
primers = ee.Image.constant(int(rawPointCollection[rawPointCollection['sample_id'] == 'FMS17564v2']['primers']))

# constant_imgs = ee.ImageCollection.fromImages([top, bot, corelength, platform_id, marker_id, type_id, primer_id]).toBands().rename(['top', 'bot', 'corelength', 'platform_id', 'marker_id', 'type_id', 'primer_id'])
constant_imgs = ee.ImageCollection.fromImages([target_marker, sequencing_platform, sample_type, primers]).toBands().rename(['target_marker', 'sequencing_platform', 'sample_type', 'primers'])

def finalImageClassification(compositeImg):
	if ensemble == False:
		# Load the best model from the classifier list
		classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', bestModelName).first()).get('c'))

		# Train the classifier with the collection
		trainedClassifer = classifier.train(fcOI, classProperty, covariateList)

		# Classify the image
		classifiedImage = compositeImg.classify(trainedClassifer,classProperty+'_Predicted')

	if ensemble == True:
		def classifyImage(classifierName):
			# Load the best model from the classifier list
			classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', classifierName).first()).get('c'))

			# Train the classifier with the collection
			trainedClassifer = classifier.train(fcOI, classProperty, covariateList)

			# Classify the image
			classifiedImage = compositeImg.classify(trainedClassifer,classProperty+'_Predicted')

			return classifiedImage

		# Classify the images, return mean
		classifiedImage = ee.ImageCollection(top_10Models.map(classifyImage)).mean()

	return classifiedImage

pred_climate_current = staticCompositeImg.addBands(climate_2015).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
pred_futureClimate_rcp26_2050 = staticCompositeImg.addBands(climate_rcp26_2050).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
pred_futureClimate_rcp26_2070 = staticCompositeImg.addBands(climate_rcp26_2070).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
pred_futureClimate_rcp45_2050 = staticCompositeImg.addBands(climate_rcp45_2050).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
pred_futureClimate_rcp45_2070 = staticCompositeImg.addBands(climate_rcp45_2070).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
pred_futureClimate_rcp60_2050 = staticCompositeImg.addBands(climate_rcp60_2050).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
pred_futureClimate_rcp60_2070 = staticCompositeImg.addBands(climate_rcp60_2070).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
pred_futureClimate_rcp85_2050 = staticCompositeImg.addBands(climate_rcp85_2050).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())
pred_futureClimate_rcp85_2070 = staticCompositeImg.addBands(climate_rcp85_2070).addBands(constant_imgs).select(covariateList).reproject(staticCompositeImg.projection())

compositeList = [pred_climate_current,
pred_futureClimate_rcp26_2050,
pred_futureClimate_rcp26_2070,
pred_futureClimate_rcp45_2050,
pred_futureClimate_rcp45_2070,
pred_futureClimate_rcp60_2050,
pred_futureClimate_rcp60_2070,
pred_futureClimate_rcp85_2050,
pred_futureClimate_rcp85_2070]

image_toExport = ee.ImageCollection(list(map(finalImageClassification, compositeList))).toBands().rename(['pred_climate_current',
'pred_futureClimate_rcp26_2050',
'pred_futureClimate_rcp26_2070',
'pred_futureClimate_rcp45_2050',
'pred_futureClimate_rcp45_2070',
'pred_futureClimate_rcp60_2050',
'pred_futureClimate_rcp60_2070',
'pred_futureClimate_rcp85_2050',
'pred_futureClimate_rcp85_2070'])

##################################################################################################################################################################
# Jackkifing
##################################################################################################################################################################

# Set number of repetitions
n_reps = 10
nList = list(range(0,n_reps))

fc_toMap = ee.FeatureCollection(ee.List(nList).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('rep',n)))

# Helper function 2: Spatial Leave One Out cross-validation function:
def LOOcv(f):
	# Get iteration ID
	rep = f.get('rep')

	# Test feature
	testFeature = ee.FeatureCollection(f)

	# Training FeatureCollection: all samples not within geometry of test feature
	trainFC = fcOI.filter(ee.Filter.geometry(testFeature).Not())

	if ensemble == False:
		# Classifier to test: same hyperparameter settings as top model from grid search procedure
		classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', bestModelName).first()).get('c'))

	if ensemble == True:
		# Classifiers to test: top 10 models from grid search hyperparameter tuning
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

# Helper function 3: R2 calculation function
def func2(f):
	rep = f.get('rep')

	# Add the iteration ID to the FC
	fc_toValidate = fcOI.map(lambda f: f.set('rep', rep))

	# Apply leave one out CV function
	predicted = fc_toValidate.map(LOOcv)

	return predicted

# Calculate R2 across range of buffer sizes
loo_cv = fc_toMap.map(func2).flatten()

# Export FC to assets
loo_cv_fc_export = ee.batch.Export.table.toAsset(
	collection = loo_cv,
	description = classProperty+'_jackknifing',
	assetId = 'users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_jackknifing'
)
loo_cv_fc_export.start()

print('Jackkifing started! Moving on...')

# Run the below part when the export is complete (might take >1 day)

# Function to convert GEE FC to pd.DataFrame. Not ideal as it's calling .getInfo(), but does the job
def GEE_FC_to_pd(fc):
	result = []

	values = fc.toList(50000).getInfo()

	BANDS = fc.first().propertyNames().getInfo()

	if 'system:index' in BANDS: BANDS.remove('system:index')

	for item in values:
		values = item['properties']
		row = [str(values[key]) for key in BANDS]
		row = ",".join(row)
		result.append(row)

	df = pd.DataFrame([item.split(",") for item in result], columns = BANDS)
	df.replace('None', np.nan, inplace = True)

	return df

# Jackkifing results
df = GEE_FC_to_pd(ee.FeatureCollection('users/'+usernameFolderString+'/'+projectFolder+'/'+classProperty+'_jackknifing'))
