import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.neighbors import BallTree
from sklearn.metrics import DistanceMetric
from scipy.stats import ks_2samp

# pathToFoldAssignmentData = '/Users/johanvandenhoogen/SPUN/richness_maps/data/arbuscular_mycorrhizal_richness_training_data_wSpatialFolds_H3.csv'
pathToFoldAssignmentData = '/Users/johanvandenhoogen/SPUN/richness_maps/data/arbuscular_mycorrhizal_richness_training_data_wSpatialFolds_Rectangle.csv'

# Define a function to naturally sort
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# The average edge length (km) of the H3 grid
h3_resolutions = dict({
    'H3res0':'H3res0 (~1108km)',
    'H3res1':'H3res1 (~419km)',
    'H3res2':'H3res2 (~158km)',
    'H3res3':'H3res3 (~60km)',
    'H3res4':'H3res4 (~23km)',
})

if __name__ == '__main__':

    # Define the distance metric to use
    dist = DistanceMetric.get_metric('haversine')

    # Read the fold assignments
    foldAssignments = pd.read_csv(pathToFoldAssignmentData)
    nrOfPoints = len(foldAssignments)
    if nrOfPoints > 10000:
        print('You run this code with >10,000 observations. This can take a while. '
                                 'Consider using a bootstrap approach for your analysis.')
    foldAssignments['Pixel_Lat'] = np.radians(foldAssignments['Pixel_Lat'])
    foldAssignments['Pixel_Long'] = np.radians(foldAssignments['Pixel_Long'])
    listOfFoldAssignments = [c for c in foldAssignments.columns if "foldID" in c]

    # Compute blockCV distances
    # Calculate the distances using k-d tree with haversine distance metric
    for foldAssigment in listOfFoldAssignments:
        hList = []
        for fold in sorted(foldAssignments[foldAssigment].drop_duplicates()):
            inFold = foldAssignments.loc[foldAssignments[foldAssigment] == fold][['Pixel_Lat', 'Pixel_Long']]
            outFold = foldAssignments.loc[foldAssignments[foldAssigment] != fold][['Pixel_Lat', 'Pixel_Long']]
            tree = BallTree(outFold, metric=dist)
            dists, ilocs = tree.query(inFold)
            distsInKm = dists.flatten() * 6367
            inFold['distance'] = distsInKm
            hList.append(inFold['distance'])
        foldAssignments['distance_' + foldAssigment] = pd.concat(hList)

    # Compute sample2sample
    tree = BallTree(foldAssignments[['Pixel_Lat', 'Pixel_Long']], metric=dist)
    dists, ilocs = tree.query(foldAssignments[['Pixel_Lat', 'Pixel_Long']], k=2)
    distToItself,nndist = np.array(list(zip(*dists)))
    sample2sample = pd.DataFrame({'sample-to-sample': nndist * 6367}, columns=['sample-to-sample'])
    # dist = DistanceMetric.get_metric('haversine')
    # distMatrix = dist.pairwise(foldAssignments[['Pixel_Lat', 'Pixel_Long']].to_numpy()) * 6373
    # np.fill_diagonal(distMatrix, np.nan)
    # sample2sample = np.nanmin(distMatrix, axis=1)

    # Compute sample2pred
    randomPoints = pd.read_csv('data/randomPoints.csv').sample(min(10000, nrOfPoints))
    randomPoints['Pixel_Lat'] = np.radians(randomPoints['Pixel_Lat'])
    randomPoints['Pixel_Long'] = np.radians(randomPoints['Pixel_Long'])
    dist = DistanceMetric.get_metric('haversine')
    tree = BallTree(foldAssignments[['Pixel_Lat', 'Pixel_Long']], metric=dist)
    dists, ilocs = tree.query(randomPoints[['Pixel_Lat', 'Pixel_Long']])
    sample2pred = pd.DataFrame({'sample-to-prediction': dists.flatten() * 6367}, columns=['sample-to-prediction'])

    # Concatenate the data
    distanceDF = pd.concat([sample2sample.reset_index(drop=True), sample2pred.reset_index(drop=True)], axis=1)
    cvDistances = ['distance_' + cv for cv in listOfFoldAssignments]
    distanceDF = pd.concat([distanceDF.reset_index(drop=True), foldAssignments[cvDistances].reset_index(drop=True)], axis=1)
    distanceDF.rename(columns=lambda cN: cN.replace('distance_foldID','blockCV'), inplace=True)

    # Compare distributions using Kolmogorov-Smirnov-Test
    listOfKolSmResults = []
    for distribution in natural_sort([d for d in distanceDF.columns if d != 'sample-to-prediction']):
        dist, stat = distribution, ks_2samp(distanceDF['sample-to-prediction'], distanceDF[distribution]).statistic
        listOfKolSmResults.append((dist, stat))
    kolSmStats = pd.DataFrame(listOfKolSmResults, columns=['foldID','Kolmogorov-Smirnov statistic']).sort_values('Kolmogorov-Smirnov statistic').reset_index(drop=True)
    kolSmDF = kolSmStats; kolSmDF["foldID"] = kolSmDF["foldID"].str.replace("blockCV_","foldID_")
    kolSmDF.to_csv('data/blockSizesForCV.csv', index=False)
    bestSize = kolSmDF['foldID'][0].replace('foldID_','')
    if 'H3' in bestSize: bestSize = h3_resolutions[bestSize]
    print('Best fitting CV assignment block size: ', bestSize)

    # Plot the best fitting distribution
    dataToShow = distanceDF[['sample-to-sample','sample-to-prediction',kolSmStats['foldID'][0].replace('foldID','blockCV')]].rename(columns={kolSmStats['foldID'][0].replace('foldID','blockCV'):'blockCV'}).replace(0, 0.01, inplace=False)
    color = [sns.color_palette('colorblind')[4], sns.color_palette('colorblind')[1]] + list(sns.color_palette("mako"))
    customPalette = sns.set_palette(sns.color_palette(color))
    kdeplot = sns.kdeplot(data=dataToShow, bw_adjust=1.5, bw_method='silverman', clip=(0, distanceDF.quantile(0.99).max()),
                log_scale=True, common_norm=False, shade=True, palette=customPalette)
    kdeplot.set(xlabel='Geographic distance (km)', title='Nearest neighbor distance distributions')
    figure = kdeplot.get_figure()
    figure.savefig('figures/NNDD_bestFoldAssignment.png', dpi=400)
    figure.show()
    kdeplot.get_figure().clf()

    # Plot all distributions
    color = [sns.color_palette('colorblind')[4], sns.color_palette('colorblind')[1]]
    for blockCvSize in range(0,len(distanceDF.columns)-2):
        color.append(sns.color_palette("mako", as_cmap=True)(blockCvSize/(len(distanceDF.columns)-2)))
    customPalette = sns.set_palette(sns.color_palette(color))
    kdeplot = sns.kdeplot(data=distanceDF.replace(0, 0.01, inplace=False), bw_adjust=1.5, bw_method='silverman', clip=(0, distanceDF.quantile(0.99).max()),
                log_scale=True, common_norm=False, shade=True, palette=customPalette)
    kdeplot.set(xlabel='Geographic distance (km)', title='Nearest neighbor distance distributions')
    sns.move_legend(kdeplot, "upper left", bbox_to_anchor=(1, 1))
    figure = kdeplot.get_figure()
    plt.figure(figsize=(8,4))
    figure.tight_layout()
    figure.savefig('figures/NNDD_all.png', dpi=400)
    figure.show()