import pandas as pd
import numpy as np
import shap        
from shap.plots import colors
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import multiprocessing
from contextlib import contextmanager
import fasttreeshap

# Constants
classProperty = 'ectomycorrhizal_richness'
df = pd.read_csv('data/ectomycorrhizal_richness_training_data.csv')

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
    'Soil pH at 5cm',
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
    grid_search_results = pd.read_csv('output/20230922_ectomycorrhizal_richness_grid_search_results_Regression_kNNDMW_guildsFixed.csv')
    VPS = int(grid_search_results['cName'][rep].split('VPS')[1].split('_')[0])
    LP = int(grid_search_results['cName'][rep].split('LP')[1].split('_')[0])

    hyperparameters = {
        'n_estimators': 250,
        'min_samples_split': LP,
        'max_features': VPS,
        'max_samples': 0.632,
        'random_state': 42
    }

    classifier = RandomForestRegressor()
    classifier.set_params(**hyperparameters)

    classifier.fit(X, y)

    explainer = fasttreeshap.TreeExplainer(classifier)

    shap_values = explainer(df[covariateList])

    return shap_values.values

with np.load('shap_values_EM_fasttree.npz') as data:
    shap_values_list = [data[f'arr_{i}'] for i in range(len(data.keys()))]

# Calculate mean SHAP values
mean_shap_values = np.mean(shap_values_list, axis=0)
mean_shap_values.shape
# Plot 1: SHAP summary plot, with all features
plt.figure()
shap.summary_plot(mean_shap_values, pd.DataFrame(data=df, columns=covariateList), show = False, sort = True)
plt.xlabel('Mean absolute SHAP value')
plt.tight_layout()
# plt.show()
plt.savefig('figures/20230922_ectomycorrhizal_richness_shap_summary_plots_full.png', dpi=300)

# Plot 2: SHAP summary plot, with project_vars removed
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
plt.savefig('figures/20230922_ectomycorrhizal_richness_shap_summary_plots_projectRemoved.png', dpi=300)

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
plt.savefig('figures/20230922_ectomycorrhizal_richness_shap_summary_plots_projectGrouped.png', dpi=300)

# Plot 4: SHAP dependence plots for the top 6 features
# Create SHAP explanation object        
explanation = shap.Explanation(values=mean_shap_values_filtered,
            # base_values=shap_values_list[0].base_values,
            data=pd.DataFrame(data=df[envCovariateListRenamed + [classProperty]], columns=envCovariateListRenamed + [classProperty]),
            feature_names=list(df[envCovariateListRenamed + [classProperty]].columns))

# Get the top 6 most important features
importance = np.abs(explanation.values).mean(0)
top_6 = np.argsort(-importance)[:6]

# Create a multipanelled figure of the top 6 features
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

# Plot
for i, feature_idx in enumerate(top_6):
    shap.dependence_plot(envCovariateListRenamed[feature_idx], explanation.values, X[envCovariateListRenamed], ax=axes[i // 3, i % 3], interaction_index = 'auto', show=False)
    plt.tight_layout()

# Save figure to file
plt.savefig('figures/20230922_ectomycorrhizal_richness_shap_scatter_plots_wInteraction.png', dpi=300)

# Plot 5: SHAP dependence plots for the top 6 features, without interaction
# Create a multipanelled figure of the top 6 features
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

# Plots without interaction
for i, feature_idx in enumerate(top_6):
    shap.dependence_plot(envCovariateListRenamed[feature_idx], explanation.values, X[envCovariateListRenamed], ax=axes[i // 3, i % 3], interaction_index = None, show=False)
    plt.tight_layout()

# Save figure to file
plt.savefig('figures/20230922_ectomycorrhizal_richness_shap_scatter_plots.png', dpi=300)

# Plot 6: SHAP bar plot for the top 12 features, with project_vars grouped together
plt.figure()
shap.summary_plot(combined_shap_values, features = df_project_vars_grouped, plot_type = 'bar', sort=True, show = False, max_display=12)
plt.xlabel('Mean absolute SHAP value')
plt.tight_layout()
plt.show()
plt.savefig('figures/20240118_ectomycorrhizal_richness_shap_bar_plots_projectGrouped.png', dpi=300)

mean_shap_values = np.mean(np.abs(combined_shap_values), axis=0)

# Create a DataFrame with feature names from df_project_vars_grouped and their corresponding mean SHAP values
df_mean_shap_values = pd.DataFrame({
    'Feature': df_project_vars_grouped.columns,
    'Mean SHAP Value': mean_shap_values
})

# Sort by absolute mean SHAP value
df_mean_shap_values = df_mean_shap_values.reindex(df_mean_shap_values['Mean SHAP Value'].sort_values(ascending=False).index)

# Write to file
df_mean_shap_values.to_csv('output/20240118_ectomycorrhizal_richness_mean_shap_values.csv', index=False)
