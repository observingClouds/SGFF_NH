#!/bin/bash
#SBATCH -J worldview          # Specify job name
#SBATCH -p shared             # Use partition prepost
#SBATCH -t 12:00:00             # Set a limit on the total run time
#SBATCH --chdir /work/mh0010/from_Mistral/mh0010/m300408/CloudMorphology/data/MODIS
#SBATCH -A mh0010              # Charge resources on this project account
#SBATCH -o worldview.o%j          # File name for standard output
#SBATCH -e worldview.e%j          # File name for standard error output

mamba activate covariability

# # INFRARED IMAGES
output_dir=Aqua_MODIS_IR
mkdir -p $output_dir

python image_download_MODIS.py --year_range [2003,2010] --months [1,2,3,4,5,6,7,8,9,10,11,12] --lon_range [-100,10] --lat_range [-10,55] --save_path $output_dir --satellite Aqua --var Brightness_Temp_Band31_Day --exist_skip True

