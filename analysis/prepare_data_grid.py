#!/usr/bin/env python
import datetime as dt
import os
import sys

import dask
import numpy as np
import tqdm
import xarray as xr
import zarr

print("Finished loading modules")

overwrite = False
fn_zarr = "./data/SGFF/level1/TB/IR_TropicalBelt.zarr"
output_file = "./data/SGFF/level2/TB/Daily_2.5x2.5_MODIS-IR_TropicalBelt_SGFF_2001-2015.zarr"

lat_bins = np.arange(-16.25, 16.25, 2.5)
lat_center = np.arange(-15, 15, 2.5)
lon_bins = np.arange(-178.75, 178.75, 2.5)
lon_center = np.arange(-177.5, 177.5, 2.5)

label_map = {"Sugar": 0, "Fish": 1, "Flowers": 2, "Flower": 2, "Gravel": 3}
label_map_rv = {0: "Sugar", 3: "Gravel", 2: "Flowers", 1: "Fish"}


def calculate_mean(da):

    lat_binned = da.groupby_bins("latitude", lat_bins, labels=lat_center).sum(
        dim="latitude"
    )
    lat_lon_binned = lat_binned.groupby_bins(
        "longitude", lon_bins, labels=lon_center
    ).sum(dim="longitude")

    return lat_lon_binned


ds_classifications_input = xr.open_zarr(fn_zarr)


# Create file and calculate common boxes
print("Create output file")
nb_times = len(ds_classifications_input.time)
nb_lons = len(lon_center)
nb_lats = len(lat_center)
nb_patterns = len(ds_classifications_input.pattern)

if os.path.exists(output_file) and not overwrite:
    print("File already exists.")
    root_grp = zarr.open(output_file)
    count = root_grp["counts"]
    times = root_grp["time"]
else:
    print("File will be newly created.")
    store = zarr.DirectoryStore(output_file)
    root_grp = zarr.group(store, overwrite=True)
    count = root_grp.create_dataset(
        "counts",
        shape=(nb_times, nb_lons, nb_lats, nb_patterns),
        chunks=(1, nb_lons, nb_lats, nb_patterns),
        dtype=float,
        compressor=zarr.Zlib(level=1),
    )
    times = root_grp.create_dataset(
        "time",
        shape=(nb_times),
        chunks=(nb_times),
        dtype="<M8[ns]",
        compressor=zarr.Zlib(level=1),
    )
    lats = root_grp.create_dataset(
        "latitude",
        shape=(nb_lats),
        chunks=(nb_lats),
        dtype=float,
        compressor=zarr.Zlib(level=1),
    )
    lons = root_grp.create_dataset(
        "longitude",
        shape=(nb_lons),
        chunks=(nb_lons),
        dtype=float,
        compressor=zarr.Zlib(level=1),
    )
    patterns = root_grp.create_dataset(
        "class",
        shape=(nb_patterns),
        chunks=(nb_patterns),
        dtype=str,
        compressor=zarr.Zlib(level=1),
    )

    lons[:] = lon_center
    lats[:] = lat_center
    patterns[:] = [label_map_rv[i] for i in range(4)]

    # Add attributes to file
    # Variable attributes
    count.attrs["_ARRAY_DIMENSIONS"] = ("time", "longitude", "latitude", "class")
    count.attrs[
        "description"
    ] = "number of classifications of particular pattern within 2.5 deg x 2.5 deg"
    lons.attrs["_ARRAY_DIMENSIONS"] = "longitude"
    lons.attrs["standard_name"] = "longitude"
    lons.attrs["units"] = "degree_east"
    lats.attrs["_ARRAY_DIMENSIONS"] = "latitude"
    lats.attrs["standard_name"] = "latitude"
    lats.attrs["units"] = "degree_north"
    times.attrs["_ARRAY_DIMENSIONS"] = "time"
    times.attrs[
        "description"
    ] = "classification id (basically each sighting of an image has a unique id)"
    patterns.attrs["_ARRAY_DIMENSIONS"] = "class"

    # Global attributes
    root_grp.attrs[
        "title"
    ] = "IR neural network meso-scale cloud pattern classifications (TropicalBelt)"
    root_grp.attrs["description"] = "NN detections of meso-scale cloud patterns"
    root_grp.attrs["author"] = "Hauke Schulz (haschulz@uw.edu)"
    root_grp.attrs["institute"] = "University of Washington, USA"
    root_grp.attrs["created_on"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    root_grp.attrs[
        "history"
    ] = f"grid averaging by prepare_data_grid of original file ({fn_zarr})"
    root_grp.attrs["python_version"] = f"{sys.version}"


print("Start actual calculation")
for d, date in enumerate(tqdm.tqdm(ds_classifications_input.time)):
    day_sel = ds_classifications_input.sel(time=date)

    count[d, :, :, :] = (
        calculate_mean(day_sel.mask.fillna(0).astype(int)).compute().astype(float)
    )
    times[d] = date.values.astype("<M8[ns]")
