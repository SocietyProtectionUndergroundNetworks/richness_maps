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

classProperty = 'ECM_diversity'

cvFoldString = 'CV_Fold'
k=10
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


fcOI = ee.FeatureCollection('users/johanvandenhoogen/000_SPUN_temp_log_zeroInflated_wTedersoo/ECM_diversitydistictObs_wProjectVars_wCV_folds_data').limit(100)

# Define hyperparameters for grid search
varsPerSplit_list = list(range(2,5))
leafPop_list = list(range(4,5))

classifierListRegression = []
# Create list of classifiers
for vps in varsPerSplit_list:
	for lp in leafPop_list:

		model_name = classProperty + '_rf_VPS' + str(vps) + '_LP' + str(lp) + '_REGRESSION'

		rf = ee.Feature(ee.Geometry.Point([0,0])).set('cName',model_name,'c',ee.Classifier.smileRandomForest(
		numberOfTrees = 500,
		variablesPerSplit = vps,
		minLeafPopulation = lp,
		bagFraction = 0.632,
		seed = 42
		).setOutputMode('REGRESSION'))

		classifierListRegression.append(rf)
kList = list(range(1,11))
kFoldAssignmentFC = ee.FeatureCollection(ee.List(kList).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('Fold',n)))

# classDfRegression = pd.DataFrame(columns = ['Mean_R2', 'StDev_R2','Mean_RMSE', 'StDev_RMSE','Mean_MAE', 'StDev_MAE', 'cName'])
# classDfClassification = pd.DataFrame(columns = ['Mean_overallAccuracy', 'StDev_overallAccuracy', 'cName'])

def gridSearch(rf):

	# print('Testing model', classifierListRegression.index(rf), 'out of total of', len(classifierListRegression))
	print(rf.get('cName').getInfo())
	# train classifier only on data not equalling zero
	# train classifier only on GlobalFungi data
	fcOI = ee.FeatureCollection('users/johanvandenhoogen/000_SPUN_temp_log_zeroInflated_wTedersoo/ECM_diversitydistictObs_wProjectVars_wCV_folds_data').filter(ee.Filter.neq(classProperty, 0))\
			.filter(ee.Filter.eq('source', 'GlobalFungi'))
	accuracy_feature = ee.Feature(computeCVAccuracyAndRMSE(rf))

	classDfRegression = pd.DataFrame(accuracy_feature.getInfo()['properties'], index = [0])
	# with open('tmp.csv', 'a') as f:
	# 	classDfRegression.to_csv('tmp.csv', columns = ['Mean_R2', 'StDev_R2','Mean_RMSE', 'StDev_RMSE','Mean_MAE', 'StDev_MAE', 'cName'], index=False, mode='a', header=f.tell()==0)

	return classDfRegression

number_of_processes = 12

@contextmanager
def poolcontext(*args, **kwargs):
	"""This just makes the multiprocessing easier with a generator."""
	pool = multiprocessing.Pool(*args, **kwargs)
	yield pool
	pool.terminate()

if __name__ == '__main__':

	with poolcontext(number_of_processes) as pool:

		results = pool.map(gridSearch, classifierListRegression)

	df = pd.concat(results)
	print(df)
