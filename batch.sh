#!/bin/sh
#PBS -N DVC_repro
#PBS -l nodes=1:ppn=1
#PBS -l walltime=80:00:00
#PBS -o output.txt
#PBS -e error.txt

source activate /home/disk/olympus/schulz/miniforge3/envs/schulz_et_al_2023

# Change to the directory where your script is located
cd /home/disk/olympus/schulz/SGFF_NH

# Run your script or command here
dvc repro -s -f mask_classifications
dvc repro -s -f grid_mask_TB

# Send an email notification when the job is done
echo "Your job is done. Please check the output." | mail -s "Job Completed" haschulz@uw.edu
