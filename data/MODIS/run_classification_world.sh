#!/bin/bash
#SBATCH --account=mh0010
#SBATCH --job-name=classify_world.run
#SBATCH --partition=gpu
#SBATCH --chdir=/work/mh0010/from_Mistral/mh0010/m300408/CloudMorphology/data/MODIS
#SBATCH --nodes=1
#SBATCH --output=/work/mh0010/from_Mistral/mh0010/m300408/CloudMorphology/data/MODIS/logs/LOG.run_classification_world.%j.o
#SBATCH --error=/work/mh0010/from_Mistral/mh0010/m300408/CloudMorphology/data/MODIS/logs/LOG.run_classification_world.%j.o
#SBATCH --exclusive
#SBATCH --mem=248GB
#SBATCH --time=02:00:00
#SBATCH --mail-user=hauke.schulz@mpimet.mpg.de
#SBATCH --mail-type=ALL

source activate /home/m/m300408/.conda/envs/sgff

python classification_world.py $1

