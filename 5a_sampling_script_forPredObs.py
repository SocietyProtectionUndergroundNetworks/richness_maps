import ee
from time import sleep
import multiprocessing
import pandas as pd
import numpy as np

from functools import partial
from contextlib import contextmanager

ee.Initialize()

arbuscular_mycorrhizal_richness_predicted_regular = ee.Image('users/johanvandenhoogen/000_SPUN_GFv4_8/arbuscular_mycorrhizal/arbuscular_mycorrhizal_richness_ClassifiedImage').rename('arbuscular_mycorrhizal_richness_predicted_regular')
arbuscular_mycorrhizal_richness_predicted_wSpatialPredictors = ee.Image('users/johanvandenhoogen/000_SPUN_GFv4_8/arbuscular_mycorrhizal_wSpatialPreds/arbuscular_mycorrhizal_richness_ClassifiedImage').rename('arbuscular_mycorrhizal_richness_predicted_wSpatialPredictors')

compositeToUse = ee.Image.cat(arbuscular_mycorrhizal_richness_predicted_regular, arbuscular_mycorrhizal_richness_predicted_wSpatialPredictors, ee.Image.pixelLonLat().reproject(arbuscular_mycorrhizal_richness_predicted_regular.projection()))

# FeatureCollection to sample
points = ee.FeatureCollection('users/johanvandenhoogen/000_SPUN_GFv4_8/arbuscular_mycorrhizal_wSpatialPreds/arbuscular_mycorrhizal_richness_training_data').select(['arbuscular_mycorrhizal_richness', 'sample_id'])


# compositeToUse = ee.Image('users/johanvandenhoogen/000_SPUN_GFv4_6/ectomycorrhizal/ectomycorrhizal_richnessclassifiedImage_zeroInflated').select('ectomycorrhizal_richness_Predicted')
# compositeToUse = compositeToUse.addBands(ee.Image.pixelLonLat().reproject(compositeToUse.projection()))

# # FeatureCollection to sample
# points = ee.FeatureCollection('users/johanvandenhoogen/000_SPUN_GFv4_7/ectomycorrhizal/ectomycorrhizal_richness_training_data').select('ectomycorrhizal_richness')

# points = points.map(lambda f: f.set('ectomycorrhizal_richness', ee.Number(f.get('ectomycorrhizal_richness')).exp().subtract(1)))


def FCDFconv(fc):
        features = fc.getInfo()['features']
        dictarray = []

        for f in features:
                dict = f['properties']
                dictarray.append(dict)

        df = pd.DataFrame(dictarray)

        return df

origCols = list(FCDFconv(points.limit(1)).columns)

# Bands to sample. Default: all bands plus variables present in original FC
BANDS = compositeToUse.bandNames().getInfo()+origCols

print('Number of features to sample:', points.size().getInfo())

def extract_grid(region, points):
        """
        Extracts a single point.
        This handles the too-many-requests error by idling the worker with backoff.
        """
        success = False
        idle = 0

        result = []
        while not success:
                try:
                    values = (compositeToUse.reduceRegions(collection = points.filterBounds(ee.Feature(region).geometry()),
                                                                                                reducer = ee.Reducer.first(),
                                                                                                scale = compositeToUse.projection().nominalScale().getInfo(),
                                                                                                tileScale = 16)
                                        .toList(50000)
                                        .getInfo())

                    for item in values:
                            values = item['properties']
                            row = [str(values[key]) for key in BANDS]
                            row = ",".join(row)
                            result.append(row)

                    return result

                except Exception as e:
                            print(e)
                            success = False
                            idle = (1 if idle > 5 else idle + 1)
                            print("idling for %d" % idle)
                            sleep(idle)

def extract_and_write_grid(n, grids, points, region):
	region = grids.get(n)
	results = extract_grid(region, points)

	if len(results) > 0:
		print("Processed %d features" % len(results))

		df = pd.DataFrame([item.split(",") for item in results], columns = BANDS)
		df.replace('None', np.nan, inplace = True)
		return df
	else:
		pass

def generateGrid(region, size):
	"""Generate a grid covering the region with size*size rectangles"""
	bins = ee.Number(size)
	coords = ee.List(region.coordinates().get(0))
	xs = coords.map(lambda l : ee.List(l).get(0))
	ys = coords.map(lambda l : ee.List(l).get(1))

	xmin = ee.Number(xs.reduce(ee.Reducer.min()))
	xmax = ee.Number(xs.reduce(ee.Reducer.max()))
	ymin = ee.Number(ys.reduce(ee.Reducer.min()))
	ymax = ee.Number(ys.reduce(ee.Reducer.max()))

	dx = xmax.subtract(xmin).divide(bins)
	dy = ymax.subtract(ymin).divide(bins)

	def f1(n):
		def f2(m):
			x1 = xmin.add(dx.multiply(n))
			y1 = ymin.add(dy.multiply(m))
			return ee.Geometry.Rectangle([x1, y1, x1.add(dx), y1.add(dy)], None, False)
		return ee.List.sequence(0, bins.subtract(1)).map(f2)
	grid = ee.List.sequence(0, bins.subtract(1)).map(f1).flatten().flatten()
	return grid

@contextmanager
def poolcontext(*args, **kwargs):
		"""This just makes the multiprocessing easier with a generator."""
		pool = multiprocessing.Pool(*args, **kwargs)
		yield pool
		pool.terminate()

if __name__ == '__main__':
		unboundedGeo = ee.Geometry.Polygon([[[-180, 88], [180, 88], [180, -88], [-180, -88]]], None, False);

		GRID_WIDTH = 15  # How many grid cells to use.
		grids = generateGrid(unboundedGeo, GRID_WIDTH)
		size = grids.size().getInfo()
		print("Grid size: %d " % size)

		# How many concurrent processors to use.  If you're hitting lots of
		# "Too many aggregation" errors (more than ~10/minute), then make this
		# number smaller.  You should be able to always use at least 20.
		NPROC = 40
		with poolcontext(NPROC) as pool:
				results = pool.map(
						partial(extract_and_write_grid, grids=grids, points=points, region=unboundedGeo),
						range(0, size))
				results = pd.concat(results)
				results.to_csv("data/202300221_AM_predObs_sampled.csv")
