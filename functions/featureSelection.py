import random
import os
import math as m
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid

### Setup the experiment name and the output directory
varToModel = 'C_amount'
today = '20230223'
foldId = 'foldID_6deg'
dataUsed = 'OrgC_2prcOutliersRemoved'
experiment = today + '_' + varToModel + '_' + dataUsed
# if not os.path.exists('../output/' + experiment):
#     os.mkdir('../output/' + experiment)

### Get all the bootstrapped data
# listOfPathsToBootstrappedSamples = sorted([f for f in os.listdir('../../data/bootstrappedSamples/generatingFolds/') if f.endswith('_wFoldIDs.csv')])

### Load all the data
C_type_toPredict = 'organic'
sampled_data = pd.read_csv('../../data/sampled_data.csv', dtype=np.float64, na_values='None').astype({'locationID': np.int32}).drop_duplicates().drop(columns=['longitude','latitude'])
OrgC_data = pd.read_csv('../../data/OrgC_2prcOutliersRemoved.csv', dtype=object, encoding='latin', index_col='uid').astype({'locationID': np.int32,
                                                                                                        'lower_depth': np.float64,
                                                                                                        'upper_depth': np.float64,
                                                                                                        'C_amount': np.float64
                                                                                                        })
blockCV = pd.read_csv('../../data/bootstrappedSamples/generatingFolds/bootstrappedSample_001_wFoldIDs.csv')[['locationID',foldId]].astype({foldId: np.int32})

# Combine the data into one dataframe
OrgC_wCov = (OrgC_data.join(sampled_data.set_index('locationID'), on='locationID').join(blockCV.set_index('locationID'), on='locationID')
                        .assign(depth = lambda x: (x['lower_depth']+x['upper_depth'])/2)
                        .assign(log_C_amount = lambda x: np.log(x['C_amount']+0.01))
                        .query('C_type == @C_type_toPredict')
                        .query('foldID_6deg.notna().values')
                        # .query('intensity.notna()', engine='python')
                        # .sample(2000)
                        .sample(frac=1))

# Define the names of the selected bands
bandNames = [
    'CHELSA_BIO_Annual_Mean_Temperature',
    'CHELSA_BIO_Annual_Precipitation',
    'CHELSA_BIO_Temperature_Seasonality',
    'CrowtherLab_SoilMoisture_Mean_downsampled10km',
    'CHELSA_BIO_Precipitation_Seasonality',
    'CHELSA_BIO_Precipitation_of_Driest_Month',
    'CIFOR_TropicalPeatlandDepth',
    'CSP_Global_Human_Modification',
    'ConsensusLandCoverClass_Snow_Ice',
    'CrowtherLab_IntactLandscapes',
    'CrowtherLab_SoilMoisture_SD_downsampled10km',
    'SG_Clay_Content_015cm',
    'EsaCci_MaxPermafrostActiveLayerThickness_downsampled1km',
    'EarthEnvCloudCover_seasonalCloudConcentration_peakTime',
    'EarthEnvTopoMed_Eastness',
    'EarthEnvTopoMed_Northness',
    'EarthEnvTopoMed_ProfileCurvature',
    'EarthEnvTopoMed_TopoPositionIndex',
    'EarthEnvTopoMed_Elevation',
    'FanEtAl_Depth_to_Water_Table_AnnualMean',
    'FanEtAl_Depth_to_Water_Table_AnnualSD',
    'GHS_Population_Density',
    'GLiM_AcidPlutonicRocks',
    'GLiM_AcidVolcanicRocks',
    'GLiM_BasicPlutonicRocks',
    'GLiM_BasicVolcanicRocks',
    'GLiM_CarbonateSedimentaryRocks',
    'GLiM_Evaporites',
    'GLiM_IntermediatePlutonicRocks',
    'GLiM_IntermediateVolcanicRocks',
    'GLiM_MetamorphicRocks',
    'GLiM_UnconsolidatedSediments',
    'GLiM_Pyroclastics',
    'GLiM_SiliciclasticSedimentaryRocks',
    'GiriEtAl_MangrovesExtent',
    'PEATMAP_GlobalPeatlandExtent',
    'PelletierEtAl_SoilAndSedimentaryDepositThicknesses',
    'SG_Depth_to_bedrock',
    'SG_Sand_Content_015cm',
    'TerraClimate_AridityIndex_downsampled4km',
    'TootchiEtAl_WetlandsRegularlyFlooded',
    'USGS_HighMountainsOrDeepCanyons',
    'USGS_LowHills',
    'USGS_SmoothPlains',
    'WorldClim2_H2OVaporPressure_AnnualSD',
    'WorldClim2_WindSpeed_AnnualSD'
]

# Optional: define a grid of hyper-parameters to search over
parameterGrid = {
    'max_features': [0.333, None],
    'max_samples': [0.632, 0.8, 1],
    'min_samples_leaf': [1, 3, 5, 10, 20]
}


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### Functions

# Define a grid of default parameters of different random forest implementations
defaultParamsGrid = dict({
    'H2O': {# distributed random forest in H2O
        'n_estimators':50,
        'max_depth':20,
        'min_samples_leaf':1,
        'max_features':0.333,
        'max_samples': 0.632,
        'min_impurity_decrease': 1e-05
    },
    'CARET':{# randomForest in CARET
        'n_estimators':50, # normally 500
        'max_depth':None,
        'min_samples_leaf':5,
        'max_features':0.333,
        'max_samples': 0.632
    },
    'smileRF':{# smileRF in Google Earth Engine
        'n_estimators':50, # not specified
        'max_depth':None,
        'max_features':'sqrt',
        'max_samples': 0.5,
        'min_samples_leaf':1,
    },
    'sklearn':{# randomForestRegressor in sklearn
        'n_estimators':50, # default: 100
        'max_depth':None,
        'max_features':1,
        'max_samples': 0.5,
        'min_samples_leaf':1.0
    }
})

# Function that creates a regression matrix and separates the label from the features
def getRegressionMatrix(originalDF, covariates, classProperty, missingValue):
    # Sample the bandNames and additional bands from the original data frame
    regressionMatrix = originalDF[covariates + [classProperty]]
    # Replace the missing values and separate the label from the features
    regressionMatrix = regressionMatrix.fillna(missingValue)
    y = regressionMatrix.loc[:,varToModel]
    X = regressionMatrix.drop(columns = varToModel)
    return X, y

# Function that puts values of a dictionary in a list and assigns them to both estimators
def prepareDict(dict):
    newDict = {}
    for key in dict:
        newDict['sfs__estimator__' + key] = [dict[key]]
        newDict['clf__' + key] = [dict[key]]
    return newDict

# Function that defines a Forward Feature Selection algorithm
def FFS(X, y, cv, grid='sklearn', n_features_to_select='auto', scoring='neg_root_mean_squared_error', tryOut=False, v=2):
    # Reduce the number of bands if you want to try out the algorithm to make it run faster
    if tryOut!=False:
        random.seed(42)
        if isinstance(tryOut, bool):
            # Reduce the number of bandNames to a third
            bN = random.sample(bandNames, m.floor(len(bandNames)*0.333))
        else:
            # Reduce the number of bandNames to what's specified
            bN = random.sample(bandNames, m.floor(tryOut))
        # Reduce the bands
        X = X[bN]

    # Generate the parameter grid
    if 'all' in grid: keys = list(defaultParamsGrid.keys()); grid = [prepareDict(defaultParamsGrid[key]) for key in defaultParamsGrid.keys()]
    if isinstance(grid, list):
        if all(isinstance(s, str) for s in grid): keys = grid; grid = [prepareDict(defaultParamsGrid[key]) for key in grid]
    if isinstance(grid, str) and grid!='all': keys = [grid]; grid = [prepareDict(defaultParamsGrid[grid])]
    if isinstance(grid, dict):
        if all(isinstance(grid[s], list) for s in grid): keys = ['user-defined']; grid = [prepareDict(dict) for dict in list(ParameterGrid(grid))]
    if isinstance(grid, dict):
        if not all(isinstance(grid[s], list) for s in grid): keys = ['user-defined']; grid = [prepareDict(grid)]
    # Define a basic random forest regressor
    clf = RandomForestRegressor(random_state=42)
    # Define a sequential feature addition algorithm
    sfs = SequentialFeatureSelector(estimator=clf,
                                n_features_to_select=n_features_to_select,
                                cv=5,
                                scoring=scoring,
                                tol=0,
                                n_jobs=-1)
    # Run grid search and SFS in a pipeline
    pipeline = Pipeline([('sfs', sfs),('clf', clf)])
    gridSearch = GridSearchCV(estimator=pipeline,
                              param_grid=grid,
                              scoring=scoring,
                              n_jobs=1,
                              cv=cv,
                              verbose=v)
    return gridSearch, X, y, keys


if __name__ == '__main__':
    # Get the regression matrix
    X, y = getRegressionMatrix(OrgC_wCov, bandNames + ['upper_depth', 'lower_depth'], varToModel, missingValue=0)

    # Define the cross validation scheme
    logo = LeaveOneGroupOut()
    cv = list(logo.split(X, y, groups=OrgC_wCov[foldId]))

    # Forward Feature Selection
    # This can be run with default hyper-parameters from 'sklearn','H2O','CARET','smileRF',
    # or all of them when choosing 'all'.
    # Optionally, a grid can be gives as a dictionary of different hyper-parameters (parameterGrid).
    # (WARNING: this might take a long while to compute)
    # The other parameters can be defined as in sklearn.
    ffs, X_fit, y_fit, keys = FFS(X, y, cv, grid='all', n_features_to_select='auto', scoring='neg_root_mean_squared_error', tryOut=False, v=2)
    featureSelectionResults = ffs.fit(X_fit, y_fit)

    # Get the grid search results
    gridSearchResultsDF = pd.DataFrame(featureSelectionResults.cv_results_)
    if keys != 'user-defined': gridSearchResultsDF['default_model_parameters'] = keys
    gridSearchResultsDF.to_csv('../data/bootstrappedSamples/generatingFolds/gridSearchResults.csv')

    # Get the best performing features
    bestFeaturesDF = pd.DataFrame(data={'bestFeatures':featureSelectionResults.best_estimator_['sfs'].get_feature_names_out()})
    bestFeaturesDF.to_csv('../data/bootstrappedSamples/generatingFolds/bestFeatures.csv')



# Show the folds
# from shapely.geometry import Point
# import geopandas as gpd
# from geopandas import GeoDataFrame
# geometry = [Point(xy) for xy in zip(OrgC_wCov['longitude'].astype('float64'), OrgC_wCov['latitude'].astype('float64'))]
# gdf = GeoDataFrame(OrgC_wCov, geometry=geometry)
# gdf[foldId] = gdf[foldId].astype('int')
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# gdf.plot(column=gdf[foldId], ax=world.plot(figsize=(10, 6), color='grey'), marker='o', markersize=15, cmap='tab10', vmin=1, vmax=20)
# plt.show()
