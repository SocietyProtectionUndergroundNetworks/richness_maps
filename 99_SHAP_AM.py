import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import multiprocessing
from contextlib import contextmanager

# Constants
classProperty = 'arbuscular_mycorrhizal_richness'
df = pd.read_csv('data/arbuscular_mycorrhizal_richness_training_data.csv')

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

# Rename variables in covariateList to increase readability
covariateListRenamed = [
    'Potential Evapotranspiration',
    'Annual Mean Temperature',
    'Annual Precipitation',
    'Max Temperature of Warmest Month',
    'Precipitation Seasonality',
    'Human Development',
    'Coef. of Var. EVI',
    'Correlation EVI',
    'Homogeneity EVI',
    'Aspect Cosine',
    'Aspect Sine',
    'Elevation',
    'Slope',
    'Topo. Position Index',
    'Burnt Areas Probability',
    'Population Density',
    'Above Ground Biomass',
    'Net Primary Productivity',
    'Depth to Bedrock',
    'Sand Content at 5cm',
    'SOC at 5cm',
    'Soil pH at 5cm',
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

# Load data and labels
X = df[covariateList]
y = df[classProperty]

# Train Random Forest models and calculate SHAP values
def calculate_shap_values(rep):
    grid_search_results = pd.read_csv('output/20230328_arbuscular_mycorrhizal_richness_grid_search_results.csv')
    VPS = int(grid_search_results['cName'][rep].split('VPS')[1].split('_')[0])
    LP = int(grid_search_results['cName'][rep].split('LP')[1].split('_')[0])

    hyperparameters = {
        'n_estimators': 25,
        'min_samples_split': LP,
        'max_features': VPS,
        'max_samples': 0.632,
        'random_state': 42
    }

    # Create 
    classifier = RandomForestRegressor()
    classifier.set_params(**hyperparameters)

    classifier.fit(X, y)

    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer(df[covariateList])

    return shap_values.values

# # Calculate SHAP values
# shap_values_list = [shap_values.values, shap_values.values, shap_values.values]

# # Calculate mean SHAP values
# mean_shap_values = np.mean(shap_values_list, axis=0)

# # Sum 'project_vars' SHAP values together
# project_shap_values = np.sum(mean_shap_values[:, len(covariateList) - len(project_vars):], axis=1).reshape(-1, 1)

# # Get SHAP values for other features
# other_shap_values = mean_shap_values[:, :len(covariateList) - len(project_vars)]

# # Combine 'project_vars' SHAP values with other features
# combined_shap_values = np.hstack([other_shap_values, project_shap_values])

# # Create new feature names list
# new_feature_names = covariateListRenamed + ["project_vars"]

# # Create a DataFrame for SHAP values
# df_shap_values = pd.DataFrame(data=combined_shap_values, columns=new_feature_names)

# # Plot
# plt.figure()
# shap.summary_plot(combined_shap_values, df_shap_values, show=False, sort=True)
# plt.xlabel('Mean absolute SHAP value')
# plt.tight_layout()
# plt.show()

# # Plot
# plt.figure()
# shap.summary_plot(other_shap_values, pd.DataFrame(data = other_shap_values, columns = covariateListRenamed), show=False, sort=True)
# plt.xlabel('Mean absolute SHAP value')
# plt.tight_layout()
# plt.show()


@contextmanager
def poolcontext(*args, **kwargs):
		"""This just makes the multiprocessing easier with a generator."""
		pool = multiprocessing.Pool(*args, **kwargs)
		yield pool
		pool.terminate()

NPROC = 7

if __name__ == '__main__':
    reps = list(range(0, 10))
    with poolcontext(NPROC) as pool:
        shap_values_list = pool.map(calculate_shap_values, reps)
        
        # plt.figure()
        # shap.summary_plot(np.mean(shap_values_list, axis=0), pd.DataFrame(data=df, columns=covariateList), show = False, sort = True)
        # plt.xlabel('Mean absolute SHAP value')
        # plt.tight_layout()
        # plt.savefig('figures/20230330_arbuscular_mycorrhizal_richness_shap_summary_plots.png', dpi=300)

        # Calculate mean SHAP values
        mean_shap_values = np.mean(shap_values_list, axis=0)

        # Sum 'project_vars' SHAP values together
        project_shap_values = np.sum(mean_shap_values[:, len(covariateList) - len(project_vars):], axis=1).reshape(-1, 1)

        # Get SHAP values for environmental features
        env_shap_values = mean_shap_values[:, :len(covariateList) - len(project_vars)]

        # Combine 'project_vars' SHAP values with other features
        combined_shap_values = np.hstack([env_shap_values, project_shap_values])

        # Create new feature names list
        new_feature_names = covariateListRenamed + ["project_vars"]

        # Create a DataFrame for SHAP values
        df_shap_values = pd.DataFrame(data=combined_shap_values, columns=new_feature_names)

        # Plot
        plt.figure()
        shap.summary_plot(combined_shap_values, df_shap_values, show=False, sort=True)
        plt.xlabel('Mean absolute SHAP value')
        plt.tight_layout()
        plt.savefig('figures/20230616_arbuscular_mycorrhizal_richness_shap_summary_plots_projectGrouped.png', dpi=300)

        # Plot
        plt.figure()
        shap.summary_plot(env_shap_values, pd.DataFrame(data = env_shap_values, columns = covariateListRenamed), show=False, sort=True)
        plt.xlabel('Mean absolute SHAP value')
        plt.tight_layout()
        plt.savefig('figures/20230616_arbuscular_mycorrhizal_richness_shap_summary_plots_projectRemoved.png', dpi=300)

        # Create a SHAP explanation object
        explanation = shap.Explanation(values=np.mean(shap_values_list, axis=0),
                                    # base_values=shap_values_list[0].base_values,
                                    data=pd.DataFrame(data=df, columns=covariateList),
                                    feature_names=list(df.columns))
        
        # Get the top 5 most important features
        importance = np.abs(explanation.values).mean(0)
        top_5 = np.argsort(-importance)[:6]

        column_names = {'ConsensusLandCover_Human_Development_Percentage': 'Human Development',
                        'SG_SOC_Content_005cm': 'SOC at 5cm',
                        'CHELSA_BIO_Annual_Mean_Temperature': 'Annual Mean Temperature',
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

        plt.savefig('figures/20230616_arbuscular_mycorrhizal_richness_shap_dependence_plots.png', dpi=300)
