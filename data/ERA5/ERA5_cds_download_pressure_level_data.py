#!/work/mh0010/m300408/anaconda3/envs/era5_mistral/bin/python
# -*- coding: utf-8 -*-

# Retrieve ERA5 data on pressure levels

#SBATCH --account=mh0010
#SBATCH --job-name=era5.run
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --threads-per-core=2
#SBATCH --output=logs/LOG.era5.run.%j.o
#SBATCH --error=logs/LOG.era5.run.%j.o
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --mail-user=hauke.schulz@mpimet.mpg.de
#SBATCH --mail-type=ALL

import cdsapi
import tqdm
import os
import numpy as np

output_filefmt = "./data/1degx1deg/3D/ERA5__{}_{}_{}_{}__{:04g}{:02g}__3D_3h.nc"
extent_north = 54
extent_west = -100
extent_south = -10
extent_east = 9

if not os.path.exists(os.path.dirname(output_filefmt)):
    os.makedirs(os.path.dirname(output_filefmt))
 
c = cdsapi.Client()

for year in tqdm.tqdm(range(2018,2021)):
    for month in np.arange(1,13):
        month_str = '{:02g}'.format(month)
        c.retrieve('reanalysis-era5-pressure-levels', {
           'variable'      : ['relative_humidity', 'temperature', 'vertical_velocity'],
           'product_type'  : 'reanalysis',
           'pressure_level': [
            '1', '2', '3',
            '5', '7', '10',
            '20', '30', '50',
            '70', '100', '125',
            '150', '175', '200',
            '225', '250', '300',
            '350', '400', '450',
            '500', '550', '600',
            '650', '700', '750',
            '775', '800', '825',
            '850', '875', '900',
            '925', '950', '975',
            '1000',
        ],
           'year'          : year,
           'month'         : month_str,
           'day'           : [
                              '01','02','03',
                              '04','05','06',
                              '07','08','09',
                              '10','11','12',
                              '13','14','15',
                              '16','17','18',
                              '19','20','21',
                              '22','23','24',
                              '25','26','27',
                              '28','29','30',
                              '31'
                             ],
           'area'          : [extent_north, extent_west, extent_south, extent_east], # North, West, South, East. Default: global
           'grid'          : [1, 1], # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
           'time'          : ['16:00'],
           'format'        : 'netcdf' # Supported format: grib and netcdf. Default: grib
                                               }, output_filefmt.format(extent_south,extent_north,extent_west,extent_east,year,month))
