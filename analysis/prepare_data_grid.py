#!/usr/bin/env python
# coding: utf-8

print("Loading modules")
import numpy as np
import datetime as dt
import sys, os
import zarr
import dask
import xarray as xr
import tqdm
print("Finished loading modules")

fn_zarr = '../data/IR_worldview_everyotherline_NH_daily_pattern_distribution.zarr'
output_file = "../data/Daily_1x1_MODIS-IR_NorthAtlantic_SGFF.zarr"

lat_bins = np.arange(-10,55,1)
lat_center = np.arange(-9.5,54.5,1)
lon_bins = np.arange(-100,10,1)
lon_center = np.arange(-99.5,9.5,1)

label_map= {'Sugar':0, 'Fish': 3, 'Flowers': 2, 'Flower': 2, 'Gravel': 1}
label_map_rv = {0:'Sugar', 1:'Gravel', 2: 'Flowers', 3: 'Fish'}


def calculate_mean(da):

    lat_binned = da.groupby_bins('latitude', lat_bins, labels=lat_center).sum(dim='latitude')
    lat_lon_binned = lat_binned.groupby_bins('longitude', lon_bins, labels=lon_center).sum(dim='longitude')
    
    return lat_lon_binned


ds_classifications_input = xr.open_zarr(fn_zarr)


# Create file and calculate common boxes
print("Create output file")
nb_times = len(ds_classifications_input.dates)
nb_lons = len(lon_center)
nb_lats = len(lat_center)
nb_patterns = len(ds_classifications_input.pattern)

if os.path.exists(output_file):
    print("File already exists.")
    root_grp = zarr.open(output_file)
    count = root_grp["counts"]
else:
    print("File will be newly created.")
    store = zarr.DirectoryStore(output_file)
    root_grp = zarr.group(store, overwrite=True)
    count = root_grp.create_dataset('counts', shape=(nb_times, nb_lons, nb_lats, nb_patterns),
                                   chunks=(1, nb_lons, nb_lats, nb_patterns),
                                   dtype=bool, compressor=zarr.Zlib(level=1))
    times = root_grp.create_dataset('time', shape=(nb_times), chunks=(nb_times),
                                    dtype='<M8[ns]',compressor=zarr.Zlib(level=1))
    lats = root_grp.create_dataset('latitude', shape=(nb_lats), chunks=(nb_lats),
                                   dtype=float, compressor=zarr.Zlib(level=1))
    lons = root_grp.create_dataset('longitude', shape=(nb_lons), chunks=(nb_lons),
                                   dtype=float, compressor=zarr.Zlib(level=1))
    patterns = root_grp.create_dataset('class', shape=(nb_patterns), chunks=(nb_patterns),
                                       dtype=str, compressor=zarr.Zlib(level=1))

    lons[:] = lon_center
    lats[:] = lat_center
    patterns[:] = [label_map_rv[i] for i in range(4)]

    # Add attributes to file
    # Variable attributes
    count.attrs['_ARRAY_DIMENSIONS'] = ('time', 'longitude', 'latitude', 'class')
    count.attrs['description'] = 'number of classifications of particular pattern within 1 deg x 1 deg'
    lons.attrs['_ARRAY_DIMENSIONS'] = ('longitude')
    lons.attrs['standard_name'] = 'longitude'
    lons.attrs['units'] = 'degree_east'
    lats.attrs['_ARRAY_DIMENSIONS'] = ('latitude')
    lats.attrs['standard_name'] = 'latitude'
    lats.attrs['units'] = 'degree_north'
    times.attrs['_ARRAY_DIMENSIONS'] = ('time')
    times.attrs['description'] = 'classification id (basically each sighting of an image has a unique id)'
    patterns.attrs['_ARRAY_DIMENSIONS'] = ('class')

    # Global attributes
    root_grp.attrs['title'] = 'IR neural network meso-scale cloud pattern classifications (North Atlantic)'
    root_grp.attrs['description'] = 'NN detections of meso-scale cloud patterns'
    root_grp.attrs['author'] = 'Hauke Schulz (hauke.schulz@mpimet.mpg.de)'
    root_grp.attrs['institute'] = 'Max Planck Institut für Meteorologie, Germany'
    root_grp.attrs['created_on'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    root_grp.attrs['history'] = "grid averaging by prepare_data_grid of original file ({})".format(fn_zarr)
    root_grp.attrs['python_version'] = "{}".format(sys.version)


print("Start actual calculation")
for d, date in enumerate(tqdm.tqdm(ds_classifications_input.dates)):
    if d <= 1985: continue 
    day_sel = ds_classifications_input.sel(dates=date)

    count[d,:,:,:] = calculate_mean(day_sel.mask).compute().astype(float)

