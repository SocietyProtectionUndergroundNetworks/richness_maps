#!/usr/bin/env python

import pandas as pd
import geopandas as gpd
import numpy as np
import multiprocessing
import itertools
from shapely.geometry import Point
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
import datetime
from contextlib import contextmanager

# Constants
classProperty = 'ectomycorrhizal_richness'
df = pd.read_csv('data/20250121_ectomycorrhizal_richness_training_data.csv')#, nrows=20)

today = datetime.date.today().strftime("%Y%m%d")

# Convert DataFrame to GeoDataFrame
geometry = [Point(xy) for xy in zip(df['Pixel_Long'], df['Pixel_Lat'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Set CRS to WGS84
gdf = gdf.set_crs('epsg:4326')

# Variables to include in the model
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

# Create final list of covariates
covariateList = covariateList + project_vars

def run_spatial_loo_cv(buffer_size, rep):
    np.random.seed(rep)

    # Project to a meter-based CRS
    gdf_proj = gdf.to_crs('epsg:3857')

    # Create buffer in meters
    gdf_proj['geometry_buffer'] = gdf_proj.buffer(buffer_size)

    # Initialize LeaveOneOut and classifier
    loo = LeaveOneOut()
    classifier = RandomForestRegressor()

    # Read in the grid search results from GEE
    grid_search_results = pd.read_csv('output/20250121_ectomycorrhizal_richness_grid_search_results.csv')
    VPS = grid_search_results['cName'][rep].split('VPS')[1].split('_')[0]
    LP = grid_search_results['cName'][rep].split('LP')[1].split('_')[0]
    # MN = grid_search_results['cName'][rep].split('MN')[1].split('_')[0]
    # if MN == 'None':
    #     MN = None
    # else:
    #     MN = np.int(MN)

    # Define the hyperparameters
    hyperparameters = {
        'n_estimators': 250, # number of trees
        # 'max_depth': MN, # maxNodes
        'min_samples_split': int(LP), # minLeafPopulation
        'max_features': int(VPS), # variablesPerSplit
        'max_samples': 0.632, # bagFraction
        'random_state': 42 # randomSeed
    }

    # Set hyperparameters
    classifier.set_params(**hyperparameters)

    # Prepare data
    X = gdf_proj[covariateList].values
    y = gdf_proj[classProperty].values

    # predictions = []

    # # Perform spatial Leave-One-Out Cross-Validation
    # n_splits = loo.get_n_splits(X)
    # stop_flag = False
    # for train_idx, test_idx in loo.split(X):

    #     if stop_flag:
    #         break

    #     # Process each test point
    #     for test_point_idx in test_idx:
    #         test_point = gdf_proj.iloc[test_point_idx]['geometry_buffer']
    #         train_points = gdf_proj.copy()
    #         train_points = train_points[train_points['geometry'].disjoint(test_point)]

    #         if not train_points.empty:
    #             classifier.fit(train_points[covariateList].values, train_points[classProperty].values)
    #             predictions.append(classifier.predict(X[test_point_idx].reshape(1, -1))[0])
    #         else:
    #             predictions.append(np.nan)

    #         if len(predictions) == 2500:
    #             stop_flag = True
    #             break

    # # Calculate R-squared value
    # r2 = r2_score(y, predictions)

    # Select 2500 random test points
    test_indices = np.random.choice(len(X), 2500, replace=False)

    predictions = []
    y_selected = []  # Store corresponding true values of y

    for train_idx, test_idx in loo.split(X):
        # Only process if the current test index is in the selected random test indices
        if test_idx[0] in test_indices:
            test_point_idx = test_idx[0]
            test_point = gdf_proj.iloc[test_point_idx]['geometry_buffer']
            train_points = gdf_proj.copy()
            train_points = train_points[train_points['geometry'].disjoint(test_point)]
            
            if not train_points.empty:
                classifier.fit(train_points[covariateList].values, train_points[classProperty].values)
                predictions.append(classifier.predict(X[test_point_idx].reshape(1, -1))[0])
            else:
                predictions.append(np.nan)

            # Collect corresponding y values
            y_selected.append(y[test_point_idx])

    # Ensure we only compute R² if we have valid predictions
    r2 = r2_score(y_selected, predictions)

    output = pd.DataFrame({'r2': r2,
                           'rep': rep,
                           'buffer_size': buffer_size}, index=[0])
    
    output.to_csv('tmp/ectomycorrhizal_SLOO_CV_'+str(buffer_size)+'_'+str(rep)+'.csv')

    return output

@contextmanager
def poolcontext(*args, **kwargs):
    """This just makes the multiprocessing easier with a generator."""
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

if __name__ == '__main__':
    # Define the list of buffer sizes to use 
    buffer_sizes = [1000, 2500, 5000, 10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000]
    reps = list(range(0,10))

    NPROC = 256
    with poolcontext(NPROC) as pool:
        results = pool.starmap(run_spatial_loo_cv, list(itertools.product(buffer_sizes, reps)))
       
        # Combine the results into a single dataframe and save to CSV
        combined = pd.concat(results)
        combined.to_csv("output/"+today+"_ectomycorrhizal_SLOO_CV.csv")
