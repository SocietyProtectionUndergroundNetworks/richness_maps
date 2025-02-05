#!/bin/bash

#SBATCH --job-name=sloocv_ECM
#SBATCH --output="out.txt"
#SBATCH --error="error.txt"
#SBATCH --ntasks=256
#SBATCH --nodes=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL

module load stack/2024-06 python/3.11.6

source $HOME/geopandas/bin/activate

# Run script
python 6_bloocv_sklearn_EM.py