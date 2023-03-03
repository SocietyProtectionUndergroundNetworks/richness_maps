# Import the modules of interest
import pandas as pd
import numpy as np
import subprocess
import multiprocessing
from contextlib import contextmanager
from functools import partial
import datetime
import ee
ee.Initialize()


today = datetime.date.today().strftime("%Y%m%d")

imageOfInterest = ee.Image('users/johanvandenhoogen/000_SPUN_GFv4_6/ectomycorrhizal/ectomycorrhizal_richnessclassifiedImage_zeroInflated').select('ectomycorrhizal_richness_Predicted')

scale = imageOfInterest.projection().nominalScale().getInfo()
imageOfInterest = ee.Image('users/camillefournier-de-lauriere/RPRS_pilotmaps2').select('Jenkins2013_MammalsRichness')

latitude_list = ee.List.sequence(-60,80,1).getInfo()
latitude_list = [int(el*10)/10 for el in latitude_list]

unboundedGeo = ee.Geometry.Polygon([[[-180, 88], [180, 88], [180, -88], [-180, -88]]], None, False)
minMaxDict = imageOfInterest.reduceRegion(**{
'reducer': ee.Reducer.minMax(),
'geometry': unboundedGeo,
'scale': scale*10,
'maxPixels': int(1e13),
'tileScale': 16
})

imageOfInterest = imageOfInterest.unitScale(ee.Number(minMaxDict.get('Jenkins2013_MammalsRichness_min')), ee.Number(minMaxDict.get('Jenkins2013_MammalsRichness_max')))

def get_lat_summary(n):
    latitude = latitude_list[n]
    mask = ee.Image.pixelLonLat().select('latitude').lt(latitude + 2).updateMask(ee.Image.pixelLonLat().select('latitude').gte(latitude)).selfMask()
    maskedImg = imageOfInterest.updateMask(mask)

    # Workaround for combined reducer: duplicate bands
    maskedImg = maskedImg.addBands(maskedImg)

    reduced = maskedImg.reduceRegion(reducer = ee.Reducer.mean().combine(ee.Reducer.percentile([2.5,97.5],['lower','upper'])),
                                     geometry = unboundedGeo, 
                                     scale = scale*10, 
                                     maxPixels = int(1e13), 
                                     tileScale = 16)

    df = pd.DataFrame.from_dict(reduced.getInfo(), orient = 'index').T
    df['latitude'] = latitude
    print(latitude)
    return df

@contextmanager
def poolcontext(*args, **kwargs):
        """This just makes the multiprocessing easier with a generator."""
        pool = multiprocessing.Pool(*args, **kwargs)
        yield pool
        pool.terminate()

if __name__ == '__main__':
        # How many concurrent processors to use.  If you're hitting lots of
        # "Too many aggregation" errors (more than ~10/minute), then make this
        # number smaller.  You should be able to always use at least 20.
        NPROC = 20
        with poolcontext(NPROC) as pool:
                results = pool.map(
                        partial(get_lat_summary),
                        range(0, len(latitude_list)))
                results = pd.concat(results)
                results.to_csv("output/"+today+"_Jenkins2013_MammalsRichness_latitude_summary.csv")
