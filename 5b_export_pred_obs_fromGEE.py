import pandas as pd
import numpy as np

import ee

ee.Initialize()

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

df = pd.DataFrame()

# Import the data
# for i in list(range(1,11)):
for i in [0,1, 5,7,8,9]:
    fc = ee.FeatureCollection('users/johanvandenhoogen/000_SPUN_GFv4_9/ectomycorrhizal/ectomycorrhizal_richness_pred_obs_rep_'+str(i))
    df = pd.concat([df, GEE_FC_to_pd(fc)])

df = df.groupby('sample_id').mean().reset_index()
df.to_csv('output/20230501_ectomycorrhizal_richness_pred_obs.csv', index = False)