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
from tqdm import tqdm
from contextlib import contextmanager

# Constants
classProperty = 'arbuscular_mycorrhizal_richness'
df = pd.read_csv('data/arbuscular_mycorrhizal_richness_training_data.csv')#, nrows=20)

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
]

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

# Create final list of covariates
covariateList = covariateList + project_vars

def run_spatial_loo_cv(buffer_size, rep):
    # Project to a meter-based CRS
    gdf_proj = gdf.to_crs('epsg:3857')

    # Create buffer in meters
    gdf_proj['geometry_buffer'] = gdf_proj.buffer(buffer_size)

    # Initialize LeaveOneOut and classifier
    loo = LeaveOneOut()
    classifier = RandomForestRegressor()

    # Read in the grid search results from GEE
    grid_search_results = pd.read_csv('output/20230315_arbuscular_mycorrhizal_richness_grid_search_results_spatialCV.csv')
    VPS = grid_search_results['cName'][rep].split('VPS')[1].split('_')[0]
    LP = grid_search_results['cName'][rep].split('LP')[1].split('_')[0]
 
    # Define the hyperparameters
    hyperparameters = {
        'n_estimators': 25, # number of trees
        'min_samples_split': np.int(LP), # minLeafPopulation
        'max_features': np.int(VPS), # variablesPerSplit
        'max_samples': 0.632, # bagFraction
        'random_state': 42 # randomSeed
    }

    # Set hyperparameters
    classifier.set_params(**hyperparameters)

    # Prepare data
    X = gdf_proj[covariateList].values
    y = gdf_proj[classProperty].values

    predictions = []

    # Perform spatial Leave-One-Out Cross-Validation
    n_splits = loo.get_n_splits(X)
    for train_idx, test_idx in tqdm(loo.split(X), total=n_splits, desc=f"Spatial LOO CV (buffer={buffer_size})"):
        test_point = gdf_proj.iloc[test_idx[0]]['geometry_buffer']
        train_points = gdf_proj#.iloc[train_idx]
        train_points = train_points[train_points['geometry'].disjoint(test_point)]

        if not train_points.empty:
            classifier.fit(train_points[covariateList].values, train_points[classProperty].values)
            predictions.append(classifier.predict(X[test_idx])[0])
        else:
            predictions.append(np.nan)

    # Calculate R-squared value
    r2 = r2_score(y, predictions)

    output = pd.DataFrame({'r2': r2,
                           'rep': rep,
                           'buffer_size': buffer_size}, index=[0])

    return output

@contextmanager
def poolcontext(*args, **kwargs):
    """This just makes the multiprocessing easier with a generator."""
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

if __name__ == '__main__':
    # Define the list of buffer sizes to use 
    buffer_sizes = [1000, 2500, 5000, 10000, 50000, 100000, 250000, 500000, 750000, 1000000]
    reps = list(range(0,10))
        # Define the list of buffer sizes to use
    buffer_sizes = [10000, 250000, 1000000]
    reps = list(range(0,3))

    NPROC = 72
    with poolcontext(NPROC) as pool:
        results = pool.starmap(run_spatial_loo_cv, list(itertools.product(buffer_sizes, reps)))
       
        # Combine the results into a single dataframe and save to CSV
        combined = pd.concat(results)
        combined.to_csv("output/SLOO_CV_AM.csv")
