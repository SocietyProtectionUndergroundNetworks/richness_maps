## Load packages
from geemap import *
import time
import requests

ee.Initialize()

# Define the list of years in the HYDE dataset 
listOfYears = ee.List([
  "_10000BC",
  "_9000BC",
  "_8000BC",
  "_7000BC",
  "_6000BC",
  "_5000BC",
  "_4000BC",
  "_3000BC",
  "_2000BC",
  "_1000BC",
  "_0AD",
  "_100AD",
  "_200AD",
  "_300AD",
  "_400AD",
  "_500AD",
  "_600AD",
  "_700AD",
  "_800AD",
  "_900AD",
  "_1000AD",
  "_1100AD",
  "_1200AD",
  "_1300AD",
  "_1400AD",
  "_1500AD",
  "_1600AD",
  "_1700AD",
  # "_1710AD",
  # "_1720AD",
  # "_1730AD",
  # "_1740AD",
  "_1750AD",
  # "_1760AD",
  # "_1770AD",
  # "_1780AD",
  # "_1790AD",
  "_1800AD",
  # "_1810AD",
  # "_1820AD",
  # "_1830AD",
  # "_1840AD",
  "_1850AD",
  # "_1860AD",
  # "_1870AD",
  # "_1880AD",
  # "_1890AD",
  "_1900AD",
  "_1910AD",
  "_1920AD",
  "_1930AD",
  "_1940AD",
  "_1950AD",
  "_1960AD",
  "_1970AD",
  "_1980AD",
  "_1990AD",
  "_2000AD",
  # "_2001AD",
  # "_2002AD",
  # "_2003AD",
  # "_2004AD",
  "_2005AD",
  # "_2006AD",
  # "_2007AD",
  # "_2008AD",
  # "_2009AD",
  "_2010AD",
  # "_2011AD",
  # "_2012AD",
  # "_2013AD",
  # "_2014AD",
  "_2015AD",
  # "_2016AD",
  # "_2017AD",
  "_2020AD",
])
indexOfYears = ee.List.sequence(0,listOfYears.length().subtract(1))

# Load the hyde dataset 
hyde = ee.ImageCollection('users/crowtherlab/Composite/HYDE33_BaselineScenario')

# Loop over the years to create a composite image of land use 
def makeLandUseImage(index):
    # Get the year and corresponding image
    year = listOfYears.get(ee.Number(index).int())
    year_image = hyde.filterMetadata('system:index', 'contains', ee.String(year)).first()

    # Create land use image by combining selected bands and normalizing
    land_use_image = (year_image.select('uopp')
                      .add(year_image.select('cropland'))
                      .add(year_image.select('grazing'))
                      .divide(year_image.select('maxln'))
                      .rename(ee.String('HYDE_LandUse').cat(ee.String(year))))

    # Apply mask and handle missing values
    land_use_image = land_use_image.unmask(-0.2)
    mask = land_use_image.gt(0.01)
    land_use_image = land_use_image.updateMask(mask)
    return land_use_image
    
landUse = ee.ImageCollection(indexOfYears.map(makeLandUseImage))

# Make a base for the gif 
baseImage = hyde.first().select(0).mask().visualize(bands='popc',min=0,max=1,palette=['ECECEC','B2B2B2'])

# Produce an equal area projection from a WKT string
# https://epsg.io/6933#
wkt = 'PROJCS["unnamed", \
    GEOGCS["WGS 84", \
        DATUM["WGS_1984", \
            SPHEROID["WGS 84",6378137,298.257223563, \
                AUTHORITY["EPSG","7030"]], \
            TOWGS84[0,0,0,0,0,0,0], \
            AUTHORITY["EPSG","6326"]], \
        PRIMEM["Greenwich",0, \
            AUTHORITY["EPSG","8901"]], \
        UNIT["degree",0.0174532925199433, \
            AUTHORITY["EPSG","9108"]], \
        AUTHORITY["EPSG","4326"]], \
    PROJECTION["Cylindrical_Equal_Area"], \
    PARAMETER["standard_parallel_1",30], \
    PARAMETER["central_meridian",0], \
    PARAMETER["false_easting",0], \
    PARAMETER["false_northing",0], \
    UNIT["Meter",1], \
    AUTHORITY["epsg","6933"]]';
equalAreaProjGlobe = ee.Projection(wkt).atScale(1000)

# Define RGB visualization parameters
palette = ["FDE686", "FDD271", "FDBF5D", "FDAC49", "FE9430", "FE7B15", "FD5C00", "F82E00", "F40000"]
# Create RGB visualization images for use as animation frames
def blendImages(img):
    img = img.rename('b1')
    return baseImage.blend(img.visualize(bands='b1', min=0, max=1, palette=palette))
    
rgbVis = ee.List(landUse.map(blendImages).toList(1000))

# Gif Parameters
gifParams = {
  'region': ee.Geometry.Polygon([-180, 90, 0, 90, 180, 90, 180, -60, 0, -60, -180, -60], None, False),
  'dimensions': 2000,
  'crs': equalAreaProjGlobe
};

# Download the images
for i in range(0, rgbVis.size().getInfo()):
    url = ee.Image(rgbVis.get(int(i))).getThumbURL(gifParams)
    time.sleep(1)
    # download_from_url(url, out_file_name='HYDE_'+str(int(i)), out_dir='/Users/Thomas/GoogleDrive/CrowtherLab/FocusTerra/00_FinalTable/Introduction', unzip=True, verbose=False)
    out_dir='./HYDEimages'
    response = requests.get(url)
    with open(out_dir + '/HYDE_'+str(int(i))+'.jpeg', 'wb') as file:
        file.write(response.content)
