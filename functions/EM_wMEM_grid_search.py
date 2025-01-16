#!/usr/bin/env python
import numpy as np
import pandas as pd
import datetime
import ee
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from contextlib import contextmanager
from functools import partial
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import multiprocessing

ee.Initialize()

### Setup the experiment name and the output directory
classProperty = 'ectomycorrhizal_richness'
today = datetime.date.today().strftime("%Y%m%d")

# Read in the data
df = pd.read_csv('data/ectomycorrhizal_richness_training_data_wMEMs.csv')

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
]

# Spatial predictors
EM_spatial_ic = ee.ImageCollection('users/johanvandenhoogen/000_SPUN/MEM_EMF')
EM_spatial = EM_spatial_ic.toBands().rename(EM_spatial_ic.aggregate_array('system:index'))

# Get bandNames of spatial predictors
spatial_preds = EM_spatial_ic.aggregate_array('system:index').getInfo()

# Create list of all covariates
covariateList = covariateList + project_vars + spatial_preds

def gridSearch(params, X, y, df, cv_col, nTrees=250, random_seed=42):
    max_features, min_samples_leaf = params

    # Initialize the RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=nTrees, random_state=random_seed, max_features=max_features, min_samples_leaf=min_samples_leaf)

    unique_folds = df[cv_col].unique()
    cv_folds = [(np.where(df[cv_col] != fold)[0], np.where(df[cv_col] == fold)[0]) for fold in unique_folds]

    # Perform cross-validation with custom folds
    scores = cross_validate(
        rf, X, y,
        cv = cv_folds,
        scoring={
            'r2': make_scorer(r2_score),
            'rmse': make_scorer(mean_squared_error, squared=False),
            'mae': make_scorer(mean_absolute_error)
        },
        return_train_score=False,
        n_jobs=1
    )

    # Prepare the results DataFrame
    model_name = f"{classProperty}_rf_VPS{max_features}_LP{min_samples_leaf}_REGRESSION"
    suffix = "_Random" if cv_col == "CV_Fold_Random" else "_Spatial"
    results = {
        f'Mean_R2{suffix}': np.mean(scores['test_r2']),
        f'StDev_R2{suffix}': np.std(scores['test_r2']),
        f'Mean_RMSE{suffix}': np.mean(scores['test_rmse']),
        f'StDev_RMSE{suffix}': np.std(scores['test_rmse']),
        f'Mean_MAE{suffix}': np.mean(scores['test_mae']),
        f'StDev_MAE{suffix}': np.std(scores['test_mae']),
        'cName': model_name
    }
    print(np.mean(scores['test_r2']))
    return pd.DataFrame([results])

@contextmanager
def poolcontext(*args, **kwargs):
		"""This just makes the multiprocessing easier with a generator."""
		pool = multiprocessing.Pool(*args, **kwargs)
		yield pool
		pool.terminate()

if __name__ == '__main__':
    # Get the regression matrix
    X = df[covariateList]
    y = df[classProperty]
    
    # Define hyperparameters for grid search
    param_grid = {
        'max_features': list(range(2, 14, 2)),
        'min_samples_leaf': list(range(2, 14, 2))
    }

    # Create a list of all combinations of hyperparameters
    all_params = list(product(param_grid['max_features'], param_grid['min_samples_leaf']))

    results_list = []

    for cv_col in ["CV_Fold_Random", "CV_Fold_Spatial"]:
        with poolcontext(processes=7) as pool:
            results = pool.map(partial(gridSearch, X=X, y=y, df=df, cv_col=cv_col), all_params)
            results_list.extend(results)

    # Combine results into a single DataFrame and save to CSV
    results_df = pd.concat(results_list, ignore_index=True)
    results_df.to_csv("output/" + today + "_grid_search_results_Regression_wMEM.csv")


