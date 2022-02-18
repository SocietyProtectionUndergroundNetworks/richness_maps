# Import the modules of interest
import pandas as pd
import numpy as np
import ee

ee.Initialize()

AMF = ee.FeatureCollection('users/johanvandenhoogen/000_SPUN/diversity/AMF_diversity_wCV_folds_data')
pred = ee.Image('users/johanvandenhoogen/000_SPUN/diversity/AMF_diversityclassifiedImg_wFuturePreds')

fCOfResults = pred.addBands(ee.Image.pixelLonLat()).sampleRegions(collection = AMF,
 								 geometries = False)

residualsFC = fCOfResults.map(lambda f: f.set('AbsResidual', ee.Number(f.get('AMF_diversity')).subtract(f.get('pred_climate_current')).abs()))

# Function to convert GEE FC to pd.DataFrame. Not ideal as it's calling .getInfo(), but does the job
def GEE_FC_to_pd(fc):
    result = []

    values = fc.toList(50000).getInfo()

    BANDS = fc.first().propertyNames().getInfo()

    if 'system:index' in BANDS: BANDS.remove('system:index')

    for item in values:
        values = item['properties']
        row = [str(values[key]) for key in BANDS]
        row = ",".join(row)
        result.append(row)

    df = pd.DataFrame([item.split(",") for item in result], columns = BANDS)
    df.replace('None', np.nan, inplace = True)

    return df

residualsDf = GEE_FC_to_pd(residualsFC)[['AMF_diversity','pred_climate_current','AbsResidual','latitude','longitude']]
residualsDf
residualsDf.to_csv('output/AMF_v1_residuals.csv')
