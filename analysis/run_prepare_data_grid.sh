#! /bin/bash
#-----------------------------------------------------------------------------
#SBATCH --account=mh0010
#SBATCH --job-name=prepare_data_grid
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=244000     # Specify real memory required per CPU in MegaBytes
#SBATCH --output=/work/mh0010/from_Mistral/mh0010/m300408/CloudMorphology/analysis/logs/LOG.prepare_data_grid.%j.o
#SBATCH --error=/work/mh0010/from_Mistral/mh0010/m300408/CloudMorphology/analysis/logs/LOG.prepare_data_grid.%j.o
#SBATCH --exclusive
#SBATCH --time=05:00:00
#=============================================================================
module load git
source activate sgff_post
git --describe --always
python prepare_data_grid.py

#python convert_data_grid.py
#mv ../data/Daily_1x1_MODIS-IR_NorthAtlantic_SGFF.nc2 ../data/Daily_1x1_MODIS-IR_NorthAtlantic_SGFF.nc
