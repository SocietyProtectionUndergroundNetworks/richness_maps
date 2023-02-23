import ee
import pandas as pd
import numpy as np
import math
from sklearn.neighbors import BallTree
from sklearn.metrics import DistanceMetric
from pathlib import Path
from functools import partial
from contextlib import contextmanager
from ctypes import c_int
import multiprocessing
from multiprocessing import Value, Lock, Process
import re
from scipy.stats import ks_2samp
import seaborn as sns


ee.Initialize()

# Define a function to naturally sort
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

if __name__ == '__main__':

    # Read the fold assignments
    foldAssignments = pd.read_csv('../data/sitesToSample_wFoldIDs.csv').sample(10000)
    foldAssignments['latitude'] = np.radians(foldAssignments['latitude'])
    foldAssignments['longitude'] = np.radians(foldAssignments['longitude'])
    listOfFoldAssignments = [c for c in foldAssignments.columns if "foldID" in c]
    for foldAssigment in listOfFoldAssignments:
        hList = []
        for fold in sorted(foldAssignments[foldAssigment].drop_duplicates()):
            inFold = foldAssignments.loc[foldAssignments[foldAssigment] == fold][['latitude', 'longitude']]
            outFold = foldAssignments.loc[foldAssignments[foldAssigment] != fold][['latitude', 'longitude']]
            # Calculate the distance using k-d tree with haversine distance metric
            dist = DistanceMetric.get_metric('haversine')
            tree = BallTree(outFold, metric=dist)
            dists, ilocs = tree.query(inFold)
            distsInKm = dists.flatten() * 6367
            inFold['distance'] = distsInKm
            hList.append(inFold['distance'])
        foldAssignments['distance_' + foldAssigment] = pd.concat(hList)

    # Compute sample2sample
    dist = DistanceMetric.get_metric('haversine')
    distMatrix = dist.pairwise(foldAssignments[['latitude', 'longitude']].to_numpy()) * 6373
    np.fill_diagonal(distMatrix, np.nan)
    sample2sample = np.nanmin(distMatrix, axis=1)

    # Compute sample2pred
    randomPoints = pd.read_csv('../data/random_points.csv').sample(10000)
    randomPoints['latitude'] = np.radians(randomPoints['latitude'])
    randomPoints['longitude'] = np.radians(randomPoints['longitude'])
    dist = DistanceMetric.get_metric('haversine')
    tree = BallTree(randomPoints[['latitude', 'longitude']], metric=dist)
    dists, ilocs = tree.query(foldAssignments[['latitude', 'longitude']])
    sample2pred = dists.flatten() * 6367

    # Concatenate the data
    distanceDF = pd.DataFrame({'sample-to-sample': sample2sample, 'sample-to-prediction': sample2pred},
                              columns=['sample-to-sample', 'sample-to-prediction'])
    cvDistances = ['distance_' + cv for cv in listOfFoldAssignments]
    distanceDF = pd.concat([distanceDF.reset_index(drop=True), foldAssignments[cvDistances].reset_index(drop=True)], axis=1)

    # Compare distributions using Kolmogorov-Smirnov-Test
    listOfKolSmResults = []
    for distribution in natural_sort([d for d in distanceDF.columns if d != 'sample-to-prediction']):
        dist, stat = distribution, ks_2samp(distanceDF['sample-to-prediction'], distanceDF[distribution]).statistic
        listOfKolSmResults.append((dist, stat))
    bestDistribution = min(listOfKolSmResults, key=lambda t: t[1])[0]
    print('Best fitting CV assignment block size: ', bestDistribution)

    # Plot the best fitting distribution
    dataToShow = distanceDF[['sample-to-sample','sample-to-prediction',bestDistribution]].rename(columns={bestDistribution:'blockCV'})
    color = [sns.color_palette('colorblind')[4], sns.color_palette('colorblind')[1]] + list(sns.color_palette("mako"))
    customPalette = sns.set_palette(sns.color_palette(color))
    kdeplot = sns.kdeplot(data=dataToShow, bw_adjust=1.5, bw_method='silverman', clip=(0, distanceDF.quantile(0.99).max()),
                log_scale=True, shade=True, palette=customPalette)
    kdeplot.set(xlabel='Geographic distance (km)', title='Nearest neighbor distance distributions')
    figure = kdeplot.get_figure()
    figure.savefig('../figures/NNDD_bestFoldAssignment.png', dpi=400)
    figure.show()
    kdeplot.get_figure().clf()

    # Plot all distributions
    kdeplot = sns.kdeplot(data=distanceDF, bw_adjust=1.5, bw_method='silverman', clip=(0, distanceDF.quantile(0.99).max()),
                log_scale=True, shade=True, palette=customPalette)
    kdeplot.set(xlabel='Geographic distance (km)', title='Nearest neighbor distance distributions')
    figure = kdeplot.get_figure()
    figure.savefig('../figures/NNDD_all.png', dpi=400)
    figure.show()