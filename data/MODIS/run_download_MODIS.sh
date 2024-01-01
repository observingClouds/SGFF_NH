#!/bin/bash
#SBATCH -J worldview          # Specify job name
#SBATCH -p shared             # Use partition prepost
#SBATCH -t 12:00:00             # Set a limit on the total run time
#SBATCH --chdir /work/mh0010/from_Mistral/mh0010/m300408/CloudMorphology/data/MODIS
#SBATCH -A mh0010              # Charge resources on this project account
#SBATCH -o worldview.o%j          # File name for standard output
#SBATCH -e worldview.e%j          # File name for standard error output

mamba activate covariability

python data/MODIS/image_download_MODIS.py --experiment $1
