import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import multiprocessing
from contextlib import contextmanager

# Constants
classProperty = 'ectomycorrhizal_richness'
df = pd.read_csv('data/ectomycorrhizal_richness_training_data.csv')

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

# Create final list of covariates
covariateList = covariateList + project_vars

# Load data and labels
X = df[covariateList]
y = df[classProperty]
rep = 1
# Train Random Forest models and calculate SHAP values
# Train Random Forest models and calculate SHAP values
def calculate_shap_values(rep):
    grid_search_results = pd.read_csv('output/20230328_ectomycorrhizal_richness_grid_search_results_Regression_zeroInflated.csv')
    VPS = grid_search_results['cName'][rep].split('VPS')[1].split('_')[0]
    LP = grid_search_results['cName'][rep].split('LP')[1].split('_')[0]

    hyperparameters = {
        'n_estimators': 250,
        'min_samples_split': int(LP),
        'max_features': int(VPS),
        'max_samples': 0.632,
        'random_state': 42
    }

    classifier = RandomForestRegressor()
    classifier.set_params(**hyperparameters)

    classifier.fit(X, y)

    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer(df[covariateList])

    return shap_values.values


@contextmanager
def poolcontext(*args, **kwargs):
		"""This just makes the multiprocessing easier with a generator."""
		pool = multiprocessing.Pool(*args, **kwargs)
		yield pool
		pool.terminate()

NPROC = 10

if __name__ == '__main__':
    reps = list(range(0, 10))
    with poolcontext(NPROC) as pool:
        shap_values_list = pool.map(calculate_shap_values, reps)

        plt.figure()
        shap.summary_plot(np.mean(shap_values_list, axis=0), pd.DataFrame(data=df, columns=covariateList), show = False, sort = True)
        plt.xlabel('Mean absolute SHAP value')
        plt.tight_layout()
        plt.savefig('figures/20230417_ectomycorrhizal_richness_shap_summary_plots.png', dpi=300)

        plt.figure()
        shap.summary_plot(np.mean(shap_values_list, axis=0), pd.DataFrame(data=df, columns=covariateList), plot_type = "bar", show = False, sort = True)
        plt.xlabel('Mean absolute SHAP value')
        plt.tight_layout()
        plt.savefig('figures/20230417_ectomycorrhizal_richness_shap_featureImp_plots.png', dpi=300)


        explanation = shap.Explanation(values=np.mean(shap_values_list, axis=0),
                                    # base_values=shap_values_list[0].base_values,
                                    data=pd.DataFrame(data=df, columns=covariateList),
                                    feature_names=list(df.columns))
        
        # Get the top 5 most important features
        importance = np.abs(explanation.values).mean(0)
        top_5 = np.argsort(-importance)[:6]

        column_names = {'ConsensusLandCover_Human_Development_Percentage': 'Human Development (%)',
                        'SG_SOC_Content_005cm': 'SOC at 5cm (g/kg)',
                        'CHELSA_BIO_Annual_Mean_Temperature': 'Annual Mean Temperature (°C)',
                        'CGIAR_PET': 'Potential Evapotranspiration',
                        'CHELSA_BIO_Max_Temperature_of_Warmest_Month': 'Max Temperature of Warmest Month',
                        'CHELSA_BIO_Precipitation_Seasonality': 'Precipitation Seasonality'}

        # Rename the columns in X using the dictionary
        X = X.rename(columns=column_names)

        # Create a multipanelled figure of the top 5 features
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

        for i, feature_idx in enumerate(top_5):
            shap.dependence_plot(X.columns[feature_idx], explanation.values, X, ax=axes[i // 3, i % 3], show=False)
            plt.tight_layout()

        plt.savefig('figures/20230417_ectomycorrhizal_richness_shap_dependence_plots.png', dpi=300)
