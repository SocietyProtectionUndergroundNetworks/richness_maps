#!/usr/bin/env python

import pandas as pd
import geopandas as gpd
import numpy as np
import multiprocessing
import os
import itertools
from shapely.geometry import Point
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
import datetime
from contextlib import contextmanager

# Constants
classProperty = 'ectomycorrhizal_richness'
df = pd.read_csv('data/20250121_ectomycorrhizal_richness_training_data.csv')# nrows=20)

today = datetime.date.today().strftime("%Y%m%d")

# Convert DataFrame to GeoDataFrame
geometry = [Point(xy) for xy in zip(df['Pixel_Long'], df['Pixel_Lat'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Set CRS to WGS84
gdf = gdf.set_crs('epsg:4326')

# Project to a meter-based CRS
gdf_proj = gdf.to_crs('epsg:3857')

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

grid_search_results = pd.read_csv('output/20250121_ectomycorrhizal_richness_grid_search_results.csv')

# Initialize LeaveOneOut and classifier
loo = LeaveOneOut()
classifier = RandomForestRegressor()

# Create final list of covariates
covariateList = covariateList + project_vars

def spatial_loo_cv(buffer_size, rep, test_index):
    """ 
    Perform spatial Leave-One-Out Cross-Validation (LOO) for a given buffer size, repetition, and test indices.
    
    Parameters:
        buffer_size (int): Buffer size for spatial exclusion (not used explicitly in this function, but included for parallel execution).
        rep (int): Index of the hyperparameter configuration.
        test_indices (list): List of test indices to evaluate.

    Returns:
        pd.DataFrame: A DataFrame containing (test_index, rep, buffer_size, y_true, y_pred)
    """

    # Read in the grid search results from GEE
    VPS = grid_search_results['cName'][rep].split('VPS')[1].split('_')[0]
    LP = grid_search_results['cName'][rep].split('LP')[1].split('_')[0]

    # Define the hyperparameters
    hyperparameters = {
        'n_estimators': 250,  # Number of trees
        'min_samples_split': int(LP),  # minLeafPopulation
        'max_features': int(VPS),  # variablesPerSplit
        'max_samples': 0.632,  # bagFraction
        'random_state': 123  # randomSeed
    }

    # Set hyperparameters
    classifier.set_params(**hyperparameters)

    # Prepare data
    X = gdf_proj[covariateList].values
    y = gdf_proj[classProperty].values

    # Create buffer in meters
    gdf_proj['geometry_buffer'] = gdf_proj.buffer(buffer_size)

    # Extract test points
    test_point = gdf_proj.iloc[test_index]['geometry_buffer']
    
    # Train on spatially disjoint data
    train_points = gdf_proj
    train_points = train_points[train_points['geometry'].disjoint(test_point)]

    if not train_points.empty:
        classifier.fit(train_points[covariateList].values, train_points[classProperty].values)
        prediction = classifier.predict(X[test_index].reshape(1, -1))[0]
    else:
        prediction = np.nan

    # Get corresponding y values for the selected test indices
    y_selected = y[test_index]

    # Return results as a DataFrame
    result_df = pd.DataFrame([{
        "test_index": test_index,
        "rep": rep,
        "buffer_size": buffer_size,
        "y_true": y_selected,
        "y_pred": prediction
    }])

    # result_df.to_csv('output/tmp/sloo_cv'+str(buffer_size)+'_'+str(rep)+'_'+str(test_index)+'.csv')
    with open('output/merged/ectomycorrhizal_SLOO_CV_'+str(buffer_size)+'.csv', 'a') as f:
        result_df.to_csv(f, header=f.tell()==0, index=False)

    return result_df

@contextmanager
def poolcontext(*args, **kwargs):
    """This just makes the multiprocessing easier with a generator."""
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def calc_r2(pd_series):
    """Calculate R-squared."""
    return r2_score(pd_series['y_true'], pd_series['y_pred'])

if __name__ == '__main__':
    buffer_sizes = [1000, 2500, 5000, 10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000]
    reps = list(range(0,10))

    test_idx_list = [test_idx[0] for _, test_idx in loo.split(gdf_proj)] 

    # Compute total expected unique combinations
    total_combinations = set(itertools.product(buffer_sizes, reps, test_idx_list))

    completed_combinations = set()

    print("Checking completed combinations in CSVs...")

    # Read existing completed combinations
    for buffer_size in buffer_sizes:
        merged_file = f'output/merged/ectomycorrhizal_SLOO_CV_{buffer_size}.csv'

        if os.path.exists(merged_file):
            finished = pd.read_csv(merged_file, dtype=str, on_bad_lines='skip', header=0)

            if {'rep', 'buffer_size', 'test_index'}.issubset(finished.columns):
                finished = finished[['buffer_size', 'rep', 'test_index']].apply(pd.to_numeric, errors='coerce').dropna()

                finished_records = set(zip(
                    finished['buffer_size'].astype(int),
                    finished['rep'].astype(int),
                    finished['test_index'].astype(int)
                ))

                completed_combinations.update(finished_records)

    print(f"Found {len(completed_combinations)} completed combinations.")

    # Ensure unique remaining combinations
    remaining_combinations = total_combinations - completed_combinations

    print("Checking uniqueness before execution...")
    remaining_list = list(remaining_combinations)

    # Check for duplicates before execution
    if len(remaining_list) != len(set(remaining_list)):
        print("Warning: Duplicates detected in remaining_combinations before processing!")

    print(f"Total unique combinations to process: {len(remaining_combinations)}")

    # Check if (5000, 9, 12472) is in remaining_combinations
    # print((5000, 9, 12472) in remaining_combinations)

    # Run in parallel
    NPROC = 256
    with poolcontext(NPROC) as pool:
        results = pool.starmap(spatial_loo_cv, remaining_list)

        # Combine results
        combined = pd.concat([r for r in results if r is not None])

        # Calculate R-squared values
        r2_values = combined.groupby(['buffer_size', 'rep'], group_keys=False)[['y_true', 'y_pred']].apply(calc_r2)

        # Rename and save results
        r2_values = r2_values.reset_index()
        r2_values.columns = ['buffer_size', 'rep', 'r2']
        r2_values.to_csv(f"output/{today}_ectomycorrhizal_SLOO_CV_v2.csv", index=False)