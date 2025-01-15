# Import the modules of interest
import pandas as pd
import numpy as np
import ee

# Authenticate and initialize
ee.Initialize()

buffer_sizes = [1000, 2500, 5000, 10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000]

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

varList = [
'AM', 'EM'
]


for var in varList:
	if var == 'AM':
		guild = 'arbuscular_mycorrhizal'
	if var == 'EM':
		guild = 'ectomycorrhizal'
	df = pd.DataFrame()
	print(var)
	for buffer in buffer_sizes:
		for rep in range(1,11):
			fc = ee.FeatureCollection('projects/crowtherlab/johan/SPUN/'+var+'_sloo_cv/'+guild+'_richness_sloo_cv_results_wExtrapolation_'+str(buffer)+'_rep_'+str(rep))

			try:
				df = pd.concat([df, GEE_FC_to_pd(fc)])
				df.to_csv('/Users/johanvandenhoogen/SPUN/richness_maps/output/'+guild+'_sloo_cv_results_wExtrapolation.csv')
			except Exception as e:
				pass
