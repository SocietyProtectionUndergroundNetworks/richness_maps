import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from itertools import repeat, chain
import matplotlib.pyplot as plt
import multiprocessing
from contextlib import contextmanager
import datetime

today = datetime.date.today().strftime("%Y%m%d")
# Constants
classProperty = 'rwr'
df = pd.read_csv('data/20250116_rwr_training_data.csv')

# Variables to include in the model
envCovariateList = [
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
envCovariateListRenamed = [
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
    'Aboveground Biomass',
    'Net Primary Productivity',
    'Depth to Bedrock',
    'Sand Content at 5cm',
    'SOC at 5cm',
    'Soil pH at 5cm'
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

# Rename variables in df to increase readability
column_names = dict(zip(envCovariateList, envCovariateListRenamed))
df = df.rename(columns=column_names)

# Create final list of covariates
covariateList = envCovariateListRenamed + project_vars 

# Subset columns from df
df = df[covariateList + [classProperty]]

# Set categorical variables
for cat in project_vars:
    df[cat] = df[cat].astype('category')

# Load data and labels
X = df[covariateList]
y = df[classProperty]

# Train Random Forest models and calculate SHAP values
def calculate_shap_values(rep):
    grid_search_results = pd.read_csv('output/20240620_arbuscular_mycorrhizal_rwr_grid_search_results.csv')
    VPS = int(grid_search_results['cName'][rep].split('VPS')[1].split('_')[0])
    LP = int(grid_search_results['cName'][rep].split('LP')[1].split('_')[0])

    hyperparameters = {
        'n_estimators': 250,
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
        try:
            with np.load('shap_values_AM_rwr.npz') as data:
                shap_values_list = [data[f'arr_{i}'] for i in range(len(data.keys()))]
        except Exception as e:
            shap_values_list = pool.map(calculate_shap_values, reps)
              
            # Save SHAP values to file
            np.savez('shap_values_AM_rwr.npz', *shap_values_list)

    # # Read SHAP values from file
    # with np.load('shap_values_AM.npz') as data:
    #       shap_values_list = [data[f'arr_{i}'] for i in range(len(data.keys()))]

    # Plot 1: SHAP summary plot, with all features
    plt.figure()
    shap.summary_plot(np.mean(shap_values_list, axis=0), pd.DataFrame(data=df, columns=covariateList), show = False, sort = True)
    plt.xlabel('Mean absolute SHAP value')
    plt.tight_layout()
    # plt.show()
    plt.savefig('figures/shap/'+today+'_arbuscular_mycorrhizal_rwr_shap_summary_plots_full.png', dpi=300)

    # Plot 2: SHAP summary plot, with project_vars removed
    # Calculate mean SHAP values
    mean_shap_values = np.mean(shap_values_list, axis=0)

    # Get the indices of the features to drop
    drop_indices = [i for i, feat in enumerate(covariateList) if feat in project_vars]

    # Create a mask where only the features not in project_vars are True
    mask = np.ones(len(covariateList), dtype=bool)
    mask[drop_indices] = False

    # Create a new dataframe without the features to drop
    df_filtered = df[envCovariateListRenamed]

    # Filter the mean SHAP values
    mean_shap_values_filtered = mean_shap_values[:, mask]

    # Plot and save figure to file
    plt.figure()
    shap.summary_plot(mean_shap_values_filtered, df_filtered, show=False, sort=True)
    plt.xlabel('Mean absolute SHAP value')
    plt.tight_layout()
    # plt.show()
    plt.savefig('figures/shap/'+today+'_arbuscular_mycorrhizal_rwr_shap_summary_plots_projectRemoved.png', dpi=300)

    # Plot 3: SHAP summary plot, with project_vars grouped together
    # Sum 'project_vars' SHAP values together
    project_shap_values = np.sum(mean_shap_values[:, len(covariateList) - len(project_vars):], axis=1).reshape(-1, 1)

    # Get SHAP values for other features
    other_shap_values = mean_shap_values[:, :len(covariateList) - len(project_vars)]

    # Combine 'project_vars' SHAP values with other features
    combined_shap_values = np.hstack([other_shap_values, project_shap_values])

    # Create new feature names list
    new_feature_names = envCovariateListRenamed + ["project_vars"]

    # Create a df where project vars are Nan
    df_project_vars_grouped = df[envCovariateListRenamed]
    df_project_vars_grouped.loc[:, 'Project Variables'] = np.NaN

    plt.figure()
    shap.summary_plot(combined_shap_values, features = df_project_vars_grouped, sort=True, show = False)
    plt.xlabel('Mean absolute SHAP value')
    plt.tight_layout()
    # plt.show()
    plt.savefig('figures/shap/'+today+'_arbuscular_mycorrhizal_rwr_shap_summary_plots_projectGrouped.png', dpi=300)

    # # Plot 4: SHAP dependence plots for the top 6 features
    # # Create SHAP explanation object        
    # explanation = shap.Explanation(values=mean_shap_values_filtered,
    #             # base_values=shap_values_list[0].base_values,
    #             data=pd.DataFrame(data=df[envCovariateListRenamed + ['amf_sampling_density'] + [classProperty]], columns=envCovariateListRenamed + [classProperty]),
    #             feature_names=list(df[envCovariateListRenamed + ['amf_sampling_density'] + [classProperty]].columns))
    
    # # Get the top 6 most important features
    # importance = np.abs(explanation.values).mean(0)
    # top_6 = np.argsort(-importance)[:6]

    # # Create a multipanelled figure of the top 6 features
    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

    # # Plot
    # for i, feature_idx in enumerate(top_6):
    #     shap.dependence_plot(list(envCovariateListRenamed+ ['amf_sampling_density'])[feature_idx], explanation.values, X[envCovariateListRenamed + ['amf_sampling_density']], ax=axes[i // 3, i % 3], interaction_index = 'auto', show=False)
    #     plt.tight_layout()

    # # Save figure to file
    # plt.savefig('figures/20240620_arbuscular_mycorrhizal_rwr_shap_scatter_plots_wInteraction.png', dpi=300)

    # # Plot 5: SHAP dependence plots for the top 6 features, without interaction
    # # Create a multipanelled figure of the top 6 features
    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

    # # Plots without interaction
    # for i, feature_idx in enumerate(top_6):
    #     shap.dependence_plot(list(envCovariateListRenamed+ ['amf_sampling_density'])[feature_idx], explanation.values, X[envCovariateListRenamed + ['amf_sampling_density']], ax=axes[i // 3, i % 3], interaction_index = None, show=False)
    #     plt.tight_layout()

    # # Save figure to file
    # plt.savefig('figures/20240620_arbuscular_mycorrhizal_rwr_shap_scatter_plots.png', dpi=300)

    # # Plot 6: SHAP bar plot for the top 12 features, with project_vars grouped together
    # plt.figure()
    # shap.summary_plot(combined_shap_values, features = df_project_vars_grouped, plot_type = 'bar', sort=True, show = False, max_display=12)
    # plt.xlabel('Mean absolute SHAP value')
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('figures/20240620_arbuscular_mycorrhizal_rwr_shap_bar_plots_projectGrouped.png', dpi=300)

    mean_shap_values = np.mean(np.abs(combined_shap_values), axis=0)

    # Create a DataFrame with feature names from df_project_vars_grouped and their corresponding mean SHAP values
    df_mean_shap_values = pd.DataFrame({
        'Feature': df_project_vars_grouped.columns,
        'Mean SHAP Value': mean_shap_values
    })

    # Sort by absolute mean SHAP value
    df_mean_shap_values = df_mean_shap_values.reindex(df_mean_shap_values['Mean SHAP Value'].sort_values(ascending=False).index)

    # Write to file
    df_mean_shap_values.to_csv('figures/shap/'+today+'_'+'arbusucular_mycorrhizal_rwr_mean_shap_values.csv', index=False)
