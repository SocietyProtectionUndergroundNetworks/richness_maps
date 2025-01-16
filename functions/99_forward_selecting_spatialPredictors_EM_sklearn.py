import multiprocessing
import pandas as pd
import numpy as np
import datetime
from contextlib import contextmanager
from sklearn.ensemble import RandomForestRegressor
import ee

ee.Initialize()

today = datetime.date.today().strftime("%Y%m%d")

guild = 'ectomycorrhizal'

iterList = list(range(0, 66))

def get_prebObs(iteration):
    print(iteration)
    # Read in the training data
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

    # List of the project variables to use
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

    # List of the spatial predictors to use
    # Spatial predictors
    EM_spatial_ic = ee.ImageCollection('users/johanvandenhoogen/000_SPUN/MEM_EMF')
    EM_spatial = EM_spatial_ic.toBands().rename(EM_spatial_ic.aggregate_array('system:index'))

    # Get bandNames of spatial predictors
    spatial_preds = EM_spatial_ic.aggregate_array('system:index').getInfo()

    # Add the spatial predictors to the covariate list
    covariateList = covariateList + project_vars + spatial_preds[0:iteration]

    # Set the target variable and features
    y = 'ectomycorrhizal_richness'
    X = df[covariateList]
    
    # Create and train the random forest regressor
    rf = RandomForestRegressor(
        n_estimators=250,
        max_features=12,
        min_samples_leaf=4,
        random_state=42
    )
    rf.fit(X, df[y])

    # Make predictions and calculate residuals
    predictions = rf.predict(X)
    residuals = df[y] - predictions

    # Create a results dataFrame
    results = df[[y, 'Pixel_Lat', 'Pixel_Long']].copy()
    results[y + '_Predicted'] = predictions
    results['residuals'] = residuals
    results['number_of_spatialpredictors'] = iteration

    return results

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

if __name__ == '__main__':
    NPROC = 7
    with poolcontext(NPROC) as pool:
        results = pool.map(get_prebObs, iterList)
        results = pd.concat(results)
        results.to_csv('spatial_predictors/' + today + '_EM_predObs_forwardselected_spatialpredictors.csv')
