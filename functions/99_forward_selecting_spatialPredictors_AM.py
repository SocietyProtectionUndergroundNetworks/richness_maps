import ee
import multiprocessing
import pandas as pd
import numpy as np
import datetime
from contextlib import contextmanager

ee.Initialize()

today = datetime.date.today().strftime("%Y%m%d")

guild = 'arbuscular_mycorrhizal'

iterList = list(range(0,21))

def get_prebObs(iteration):
    # Function to convert GEE FC to pd.DataFrame. Not ideal as it's calling .getInfo(), but does the job
    def GEE_FC_to_pd(fc):
        result = []

        values = fc.toList(500000).getInfo()

        BANDS = fc.first().propertyNames().getInfo()

        if 'system:index' in BANDS: BANDS.remove('system:index')

        for item in values:
            values_item = item['properties']
            row = [values_item[key] for key in BANDS]
            result.append(row)

        df = pd.DataFrame([item for item in result], columns = BANDS)
        df.replace('None', np.nan, inplace = True)

        return df

    # Add point coordinates to FC as properties
    def addLatLon(f):
        lat = f.geometry().coordinates().get(1)
        lon = f.geometry().coordinates().get(0)
        return f.set(latString, lat).set(longString, lon)

    ####################################################################################################################################################################
    # Configuration
    ####################################################################################################################################################################
    # Input the name of the classification property
    classProperty = guild + '_richness'

    # Specify the column names where the latitude and longitude information is stored
    latString = 'Pixel_Lat'
    longString = 'Pixel_Long'

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

    # List of the spatial predictors to use
    spatial_preds = ['MEM1', 'MEM2', 'MEM4', 'MEM5', 'MEM7', 'MEM9', 'MEM12', 'MEM13', 'MEM17', 'MEM19', 'MEM21', 'MEM22', 'MEM23', 'MEM25', 'MEM26', 'MEM27', 'MEM28', 'MEM31', 'MEM32', 'MEM33', 'MEM34', 'MEM35', 'MEM36', 'MEM37', 'MEM39', 'MEM40', 'MEM41', 'MEM46', 'MEM49', 'MEM50', 'MEM51', 'MEM53', 'MEM55', 'MEM56', 'MEM57', 'MEM61', 'MEM64', 'MEM72', 'MEM73', 'MEM75', 'MEM77', 'MEM79', 'MEM85', 'MEM91', 'MEM95', 'MEM97', 'MEM99', 'MEM100', 'MEM108', 'MEM111', 'MEM114', 'MEM117', 'MEM132', 'MEM142', 'MEM172', 'MEM208', 'MEM224', 'MEM227', 'MEM231', 'MEM261', 'MEM272', 'MEM294', 'MEM308', 'MEM310', 'MEM331', 'MEM386']
    
    # Add the spatial predictors to the covariate list
    covariateList = covariateList + project_vars + spatial_preds[0:iteration]
    
    ##################################################################################################################################################################
    # Predicted - Observed
    ##################################################################################################################################################################
    fcOI = ee.FeatureCollection('users/johanvandenhoogen/000_SPUN_GFv4_9/arbuscular_mycorrhizalwMEM/arbuscular_mycorrhizal_richness_training_data_wMEMs')

    classifier = ee.Classifier.smileRandomForest(
            numberOfTrees = 250,
            variablesPerSplit = 10,
            minLeafPopulation = 4,
            bagFraction = 0.632,
            seed = 42
            ).setOutputMode('REGRESSION')

    # Train the classifier with the collection
    trainedClassifier = classifier.train(fcOI, classProperty, covariateList)

    # Set reference level
    fcOIforClassification = fcOI.map(lambda f: f.set('sequencing_platform454Roche', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('sequencing_platformIllumina', 1)) # <- This is the reference level
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('sample_typerhizosphere_soil', 1)) # <- This is the reference level
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('sample_typesoil', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('sample_typetopsoil', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersAML1_AML2_then_AMV4_5NF_AMDGR', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersAML1_AML2_then_NS31_AM1', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersAML1_AML2_then_nu_SSU_0595_5__nu_SSU_0948_3_', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersAMV4_5F_AMDGR', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersAMV4_5NF_AMDGR', 1)) # <- This is the reference level
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersGeoA2_AML2_then_NS31_AMDGR', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersGeoA2_NS4_then_NS31_AML2', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersGlomerWT0_Glomer1536_then_NS31_AM1A_and_GlomerWT0_Glomer1536_then_NS31_AM1B', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersGlomerWT0_Glomer1536_then_NS31_AM1A__GlomerWT0_Glomer1536_then_NS31_AM1B', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersNS1_NS4_then_AML1_AML2', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersNS1_NS4_then_AMV4_5NF_AMDGR', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersNS1_NS4_then_NS31_AM1', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersNS1_NS41_then_AML1_AML2', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersNS31_AM1', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersNS31_AML2', 0))
    fcOIforClassification = fcOIforClassification.map(lambda f: f.set('primersWANDA_AML2', 0))

    # Classify the FC
    classifiedFC = fcOIforClassification.classify(trainedClassifier,classProperty+'_Predicted')

    # Add coordinates to FC
    predObs = classifiedFC.map(addLatLon)

    # Add residuals to FC
    predObs_wResiduals = predObs.map(lambda f: f.set('residuals', ee.Number(f.get(classProperty)).subtract(f.get(classProperty+'_Predicted'))))

    # Convert to pd
    predObs_df = GEE_FC_to_pd(predObs_wResiduals)

    # Add number of spatial predictors to df
    predObs_df['number_of_spatialpredictors'] = iteration

    return predObs_df

@contextmanager
def poolcontext(*args, **kwargs):
		"""This just makes the multiprocessing easier with a generator."""
		pool = multiprocessing.Pool(*args, **kwargs)
		yield pool
		pool.terminate()

if __name__ == '__main__':
		NPROC = 20
		with poolcontext(NPROC) as pool:
				results = pool.map(get_prebObs, iterList)
				results = pd.concat(results)
				results.to_csv('spatial_predictors/'+today+'_AM_predObs_forwardselected_spatialpredictors.csv')

