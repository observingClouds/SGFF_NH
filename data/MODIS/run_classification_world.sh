#!/bin/bash
#SBATCH --account=mh0010
#SBATCH --job-name=classify_world.run
#SBATCH --partition=gpu
#SBATCH --chdir=/work/mh0010/m300408/CharacterizationOfMesoscalePatterns/Classification/NN-application
#SBATCH --nodes=1
#SBATCH --output=/work/mh0010/m300408/CharacterizationOfMesoscalePatterns/Classification/NN-application/logs/LOG.run_classification_world.%j.o
#SBATCH --error=/work/mh0010/m300408/CharacterizationOfMesoscalePatterns/Classification/NN-application/logs/LOG.run_classification_world.%j.o
#SBATCH --exclusive
#SBATCH --mem-per-cpu=248000
#SBATCH --time=12:00:00
#SBATCH --mail-user=hauke.schulz@mpimet.mpg.de
#SBATCH --mail-type=ALL

module load anaconda3
source activate /work/mh0010/m300408/anaconda3/envs/github_sugar

python classification_world.py