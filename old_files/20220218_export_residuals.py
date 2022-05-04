# Import the modules of interest
import pandas as pd
import numpy as np
import ee

ee.Initialize()

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

# Taxa
taxa = ['AMF', 'ECM']

for taxon in taxa:

    # Training data
    training_data = ee.FeatureCollection('users/johanvandenhoogen/000_SPUN/diversity/'+taxon+'_diversity_wCV_folds_data')

    # Predicted
    predicted = ee.Image('users/johanvandenhoogen/000_SPUN/diversity/'+taxon+'_diversityclassifiedImg_wFuturePreds')

    fCOfResults = predicted.addBands(ee.Image.pixelLonLat()).sampleRegions(collection = training_data, geometries = False)

    residualsFC = fCOfResults.map(lambda f: f.set('AbsResidual', ee.Number(f.get(taxon+'_diversity')).subtract(f.get('pred_climate_current')).abs()))

    # Convert to pandas
    residualsDf = GEE_FC_to_pd(residualsFC)[[taxon+'_diversity','pred_climate_current','AbsResidual','latitude','longitude']]
    # Write to file
    residualsDf.to_csv('output/'+taxon+'_v1_residuals.csv')
