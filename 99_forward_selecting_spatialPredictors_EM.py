import ee
import multiprocessing
import pandas as pd
import numpy as np
import datetime
from contextlib import contextmanager

ee.Initialize()

today = datetime.date.today().strftime("%Y%m%d")

guild = 'ectomycorrhizal'

iterList = list(range(0,66))

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
    
    ##################################################################################################################################################################
    # Predicted - Observed
    ##################################################################################################################################################################
    fcOI = ee.FeatureCollection('users/johanvandenhoogen/000_SPUN_GFv4_9/ectomycorrhizalwMEM/ectomycorrhizal_richness_training_data_wMEMs')

    classifier = ee.Classifier.smileRandomForest(
            numberOfTrees = 250,
            variablesPerSplit = 12,
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
				results.to_csv('spatial_predictors/'+today+'_EM_predObs_forwardselected_spatialpredictors.csv')

