import ee
from time import sleep
import multiprocessing
import math
import time
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial
from contextlib import contextmanager
from ctypes import c_int
from multiprocessing import Value, Lock, Process

ee.Initialize()

# Define distance in meters to gapfill (default = 10000)
distanceToFillInMeters = 10000

sampling_density = ee.Image('users/johanvandenhoogen/000_SPUN_GFv4_10/ecm_sampleintensity_5degrees_scaled').rename('ecm_sampling_density')

compositeToUse = ee.Image('projects/crowtherlab/Composite/CrowtherLab_Composite_30ArcSec').addBands(sampling_density)

# Function to gapfill missing pixels
def gapFillAndExtendBounds(compositeToUse):
	# N.B., This script is based on code from Matt Hancher and includes annotations
	# from his original example code.
	# Use a helper function to fill holes and boundaries with the nearest value
	def fillHolesWithNearestValue(imageToFill):
		source = imageToFill.mask();
		# Measure 1000000x the distance to the nearest valid pixel. We pick
		# a large cost (here 1000000) so that the least-cost path will not be
		# influenced later by the border pixels.
		# originally: 1000000
		cost0 = ee.Image(1000000).where(source, 0).cumulativeCost(source, distanceToFillInMeters);

		# Measure the distance to the nearest pixel plus the half-pixel
		# traversal to the center of a valid border pixel, which may be
		# 1/2 or 1/sqrt(2).
		cost1 = ee.Image(1000000).where(source, 1).cumulativeCost(source, distanceToFillInMeters);

		# Measure the distance to the nearest pixel plus the half-pixel
		# traversal to center of a valid pixel, where the valid pixel
		# has a cost equal to its original value.
		cost2 = imageToFill.unmask(1000000).cumulativeCost(source, distanceToFillInMeters);

		# Finally we can compute the original value of the nearest
		# unmasked pixel.
		fill = cost2.subtract(cost0).divide(cost1.subtract(cost0));

		# Fill in the masked pixels.
		filled = imageToFill.unmask(0).add(fill);

		return filled.copyProperties(imageToFill);

	# Use a helper function to convert an image collection to a multiband image
	def icToImage(imageCollection):

		# Create an empty image to fill
		emptyImage = ee.Image(0).rename('BAND_TO_REMOVE');

		# Iterate through the collection to make the new multiband image
		def functionToIterate(image, result):
			return ee.Image(result).addBands(image)

		multibandImageToSub = ee.Image(imageCollection.iterate(functionToIterate, emptyImage))
		multibandImageSelected = multibandImageToSub.select(multibandImageToSub.bandNames().remove('BAND_TO_REMOVE'))

		return multibandImageSelected;

	# Turn the multiband image of interest into an image collection
	imageCollection = ee.ImageCollection(
		compositeToUse.bandNames().map(lambda bandName: compositeToUse.select([bandName]).set('imageName', bandName)))
	# print('Image Collection from Composite',imageCollection)

	# Separate out the images that shouldn't be filled
	imagesNotToFill = imageCollection.filter(ee.Filter.inList('imageName', covariateList).Not())
	# print('Images not to Fill',imagesNotToFill);

	imagesNotFilled = icToImage(imagesNotToFill)
	# print('Images not Filled',imagesNotFilled);

	imagesToFill = imageCollection.filter(ee.Filter.inList('imageName', covariateList))
	# print('Images to Fill',imagesToFill);

	imagesFilled = imagesToFill.map(fillHolesWithNearestValue)
	# print('Gap Filled Images',imagesFilled);

	multiBandImageFilled = icToImage(imagesFilled)
	# print('Multiband image filled',multiBandImageFilled);

	imageToReturn = ee.Image.cat(multiBandImageFilled, imagesNotFilled)
	# print('Filled Composite',stackedCompositeFilled);

	return imageToReturn

covariateList = compositeToUse.bandNames().getInfo()
compositeToUse = gapFillAndExtendBounds(compositeToUse)

# FeatureCollection to sample
points = ee.FeatureCollection("users/johanvandenhoogen/000_SPUN_GFv4_10/20231001_EM_richness_rarefied")

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
				results.to_csv("data/20231001_EM_richness_rarefied_sampled.csv")
