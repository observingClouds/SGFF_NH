#! /bin/bash
#-----------------------------------------------------------------------------
#SBATCH --account=mh0010
#SBATCH --job-name=prepare_data_grid
#SBATCH --partition=prepost
#SBATCH --mem-per-cpu=244000     # Specify real memory required per CPU in MegaBytes
#SBATCH --output=/work/mh0010/m300408/CloudMorphology/analysis/logs/LOG.prepare_data_grid.%j.o
#SBATCH --error=/work/mh0010/m300408/CloudMorphology/analysis/logs/LOG.prepare_data_grid.%j.o
#SBATCH --exclusive
#
#SBATCH --time=12:00:00
#=============================================================================
module load anaconda3
source activate /work/mh0010/m300408/anaconda3/envs/radiation_project
git --describe --always
cd /work/mh0010/m300408/CloudMorphology/analysis/
python prepare_data_grid.py

source activate /work/mh0010/m300408/anaconda3/envs/BCOenv
python convert_data_grid.py
mv ../data/Daily_1x1_MODIS-IR_NorthAtlantic_SGFF.nc2 ../data/Daily_1x1_MODIS-IR_NorthAtlantic_SGFF.nc
 
