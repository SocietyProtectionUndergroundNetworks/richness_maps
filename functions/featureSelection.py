#!/usr/bin/env python
import os
import math as m
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import PredefinedSplit
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ParameterGrid

# !! Define a random number for the Random Nr Generator
randomNr = 42

### Setup the experiment name and the output directory
varToModel = 'arbuscular_mycorrhizal_richness'
today = '20230223'
foldId = 'CV_Fold_Spatial'
experiment = today + '_' + varToModel 
if not os.path.exists('output/' + experiment):
    os.mkdir('output/' + experiment)

df = pd.read_csv('data/arbuscular_mycorrhizal_richness_training_data.csv')

# Define the names of the selected bands
bandNames = [
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

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### Functions

# Define a grid of default parameters of different random forest implementations
defaultParamsGrid = {
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
        'n_estimators':50, 
        'max_depth':None, # maxNodes
        'max_features':'sqrt', # variablesPerSplit
        'max_samples': 0.5, # bagFraction
        'min_samples_leaf':1, # minLeafPopulation
    },
    'sklearn':{# randomForestRegressor in sklearn
        'n_estimators':50, # default: 100
        'max_depth':None,
        'max_features':1,
        'max_samples': 0.5,
        'min_samples_leaf':1.0
    }
}

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
# def prepareDict(dict):
#     newDict = {}
#     for key in dict:
#         newDict['sfs__estimator__' + key] = [dict[key]]
#         newDict['clf__' + key] = [dict[key]]
#     return newDict

# Function that defines a Forward Feature Selection algorithm
def FFS(X, y, params='sklearn', nParamCombsToTry = None, n_features_to_select='auto', direction = 'forward', scoring=['neg_root_mean_squared_error','r2'], tryOut=False, newOutput=False, v=True):
    global dict
    # Reduce the number of bands if you want to try out the algorithm to make it run faster
    if tryOut!=False:
        if isinstance(tryOut, bool):
            nrToSample = m.floor(len(bandNames)*0.333)
        else:
            nrToSample = m.floor(tryOut)
        # Reduce the bands
        bN = pd.DataFrame(bandNames).sample(nrToSample, random_state=randomNr)[0].values.tolist()
        X_fit = X[bN]
    else:
        X_fit = X

    # Generate the parameter grid to loop over (only FFS)
    if 'all' in params:
        keys = list(defaultParamsGrid.keys()); grid = [defaultParamsGrid[key] for key in defaultParamsGrid.keys()]
    if isinstance(params, str) and not 'all' in params:
        keys = [params]; grid = [defaultParamsGrid[params]]
    if isinstance(params, list) and all(isinstance(s, str) for s in params):
        keys = params; grid = [defaultParamsGrid[key] for key in params]
    if isinstance(params, list) and all(isinstance(s, dict) for s in params):
        keys = ['user-defined grid']; grid = [p for p in params]
    if isinstance(params, dict) and not all(isinstance(params[s], list) for s in params):
        keys = ['user-defined grid']; grid = [params]
    if isinstance(params, dict) and all(isinstance(params[s], list) for s in params):
        keys = ['user-defined grid']; grid = [dic for dic in list(ParameterGrid(params))]
    # Generate the parameter grid to loop over (FFS & grid search)
    # if isinstance(params, ParameterGrid):
    #     keys = ['user-defined grid']; grid = [dic for dic in list(params)]

    # Loop over all the combinations; or alternatively take a sample from them
    if isinstance(nParamCombsToTry, (int, float, complex)) and not isinstance(nParamCombsToTry, bool) and len(grid) > 1:
        combosToTry = pd.DataFrame(range(0,len(grid))).sample(min(m.floor(nParamCombsToTry),len(grid)), random_state=randomNr)[0].values.tolist()
        grid = [grid[i] for i in combosToTry]

    # Setup the output
    if newOutput:
        try: os.remove('output/' + experiment + '/gridSearchResults.csv')
        except FileNotFoundError: pass

    # Load the old output and get the parameter combinations that were already tested
    try:
        prevParams = pd.read_csv('output/' + experiment + '/gridSearchResults.csv')['params']
        for i in prevParams:
            try:
                grid.remove(eval(i))
            except ValueError: pass
    except FileNotFoundError: pass

    # Compute the results
    ind = 0
    if v == True: print('Fitting {0} folds for each of {1} candidates, totalling {2} fits'.format(len(df[foldId].drop_duplicates()),len(grid),len(df[foldId].drop_duplicates())*len(grid)))
    for comboDict in grid:
        ind =+ 1
        t1 = time.time()
        # Define a random forest regressor
        clf = RandomForestRegressor(**comboDict, random_state=randomNr)
        # Define a sequential feature addition algorithm
        sfs = SequentialFeatureSelector(estimator=clf,
                                        n_features_to_select=n_features_to_select,
                                        cv=list(LeaveOneGroupOut().split(X_fit, y, groups=df[foldId])),
                                        scoring='neg_root_mean_squared_error',
                                        direction=direction,
                                        tol=0,
                                        n_jobs=-1).fit(X_fit, y)
        tdiff = time.time() - t1
        # Get the best performing features and the best scoring
        bestFeatures = sorted(sfs.get_feature_names_out())
        X_fit_red = X_fit[bestFeatures]
        # Save the scorings
        scorings = cross_validate(clf, X=X_fit_red, y=y, scoring=scoring, cv=list(LeaveOneGroupOut().split(X_fit, y, groups=df[foldId])), n_jobs=-1)
        scoringDF = pd.DataFrame({#'mean_fit_time': [scorings['fit_time'].mean()],
                                  #'std_fit_time': [scorings['fit_time'].std()],
                                  #'mean_score_time': [scorings['score_time'].mean()],
                                  #'std_score_time': [scorings['score_time'].std()],
                                  'params':[comboDict]
                                  })
        if len(scoring) == 1:
            for i in range(0, len(scorings['test_score'])):
                scoringDF = scoringDF.assign(**{'split' + str(i+1) + '_test_score':  scorings['test_score'][i]})
            scoringDF = scoringDF.assign(**{'mean_test_score': scorings['test_score'].mean(), 'std_test_score':scorings['test_score'].std()})
        else:
            for s in range(0, len(scoring)):
                for i in range(0, len(scorings['test_' + scoring[0]])):
                    scoringDF = scoringDF.assign(**{'split' + str(i+1) + '_test_' + scoring[s]:  scorings['test_' + scoring[s]][i]})
                scoringDF = scoringDF.assign(**{'mean_test_' + scoring[s]: scorings['test_' + scoring[s]].mean(), 'std_test_' + scoring[s]:scorings['test_' + scoring[s]].std()})
        scoringDF.insert(loc=0, column='best_features', value=[bestFeatures])
        # resultsDF = pd.concat([resultsDF, scoringDF], ignore_index=True)
        scoringDF['mean_test_' + scoring[0]]
        if v == True:
            string = ''
            for s in range(0,len(scoring)):
                string += '' + scoring[s] + '=' + str(round(scoringDF['mean_test_' + scoring[s]], 2)[0]) + ' - '
            print('[DONE {0: >2}] - {1}s - {2}{3}'.format(ind, round(tdiff, 1), string, str(comboDict).replace('{','').replace('}','').replace("'",'').replace(": ",'=').replace(",",';')))
        # Write the results to a dataframe
        with open('output/' + experiment + '/gridSearchResults.csv', 'a') as f:
            scoringDF.to_csv(f, mode='a', header=f.tell()==0)


if __name__ == '__main__':
    # Get the regression matrix
    X, y = getRegressionMatrix(df, bandNames, varToModel, missingValue=0)
    
    # Optional: define a grid of hyper-parameters to search over
    hyperParameterGrid = ({
        'max_features': [None], # all features
        'max_samples': [0.632], #bagFraction 
        'min_samples_leaf': list(range(2,14,2)), # minLeafPopulation
        'min_samples_split': list(range(2,10,2)), # minLeafPopulation
        'max_depth':[20, 25, None]
    })

    # Forward Feature Selection (FFS)
    # ref: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html
    # This Sequential Feature Selector adds features in a greedy fashion until scoring is not improving anymore.
    #
    # {params} defines the hyper-parameters. The algorithm can be run with default hyper-parameters
    # from 'sklearn','H2O','CARET','smileRF', or all of them when choosing 'all'.
    # Alternatively, a hyper-parameter grid search including feature selection can be run simultaneously by
    # entering a dictionary (or a list of dictionaries) as 'params'.
    # {nParamCombsToTry} can used to run only a random sample from the specified grid (random grid search).
    # {n_features_to_select} can be set to define the absolute number of features to select (if integer), or the
    # fraction of features to select (if float between 0 and 1). If set to 'auto', the best performing number of
    # features is selected.
    # {scoring} defines the scoring strategy to evaluate the performance. If multiple entries are given, the first one
    # defines the ranking strategy.
    # {tryOut} reduces the number of initial features to debug and run things quickly.
    # It reduces the number of features to the integer specified or to 1/3 if set to True.
    # {newOutput} removes the output file if set to True before running the algorithm.
    FFS(X, y,
        params=hyperParameterGrid,#defaultParamsGrid['smileRF'],
        nParamCombsToTry = False,
        n_features_to_select='auto',
        direction = 'backward',
        scoring=['neg_root_mean_squared_error','r2'],
        tryOut=False,
        newOutput=True)




# References:
# On nested vs. flat cross validation: https://doi.org/10.1016/j.eswa.2021.115222


# Show the folds
# from shapely.geometry import Point
# import geopandas as gpd
# from geopandas import GeoDataFrame
# geometry = [Point(xy) for xy in zip(df['longitude'].astype('float64'), df['latitude'].astype('float64'))]
# gdf = GeoDataFrame(df, geometry=geometry)
# gdf[foldId] = gdf[foldId].astype('int')
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# gdf.plot(column=gdf[foldId], ax=world.plot(figsize=(10, 6), color='grey'), marker='o', markersize=15, cmap='tab10', vmin=1, vmax=20)
# plt.show()

# Rank the results
#     resultsDF['rank_test_score'] = resultsDF['mean_test_' + scoring[0]].rank(method='dense', ascending=False).astype('int')
#     bestFeaturesDF = pd.DataFrame(resultsDF['best_features'][resultsDF['rank_test_score'] == min(resultsDF['rank_test_score'])])