#!/usr/bin/env python
import datetime as dt
import os
import sys

import dask
import dask.array
import numpy as np
import tqdm
import xarray as xr
import zarr

print("Finished loading modules")

overwrite = False
fn_zarr = "./data/SGFF/level1/TB/IR_TropicalBelt.zarr"
output_file = (
    "./data/SGFF/level2/TB/Daily_2.5x2.5_MODIS-IR_TropicalBelt_SGFF_2001-2015.zarr"
)
filter = True
BT_THRESHOLD = 240

grid_spacing = 2.5
lat_bins = np.arange(-16.25, 16.25 + grid_spacing, grid_spacing)
lat_center = np.arange(-15, 15 + grid_spacing, grid_spacing)
lon_bins = np.arange(-181.25, 181.25 + grid_spacing, grid_spacing)
lon_center = np.arange(-180, 180 + grid_spacing, grid_spacing)

label_map = {"Sugar": 0, "Fish": 3, "Flowers": 2, "Flower": 2, "Gravel": 1}
label_map_rv = {0: "Sugar", 1: "Gravel", 2: "Flowers", 3: "Fish"}


def calculate_mean(da):

    lat_binned = da.groupby_bins("latitude", lat_bins, labels=lat_center).sum(
        dim="latitude"
    )
    lat_lon_binned = lat_binned.groupby_bins(
        "longitude", lon_bins, labels=lon_center
    ).sum(dim="longitude")

    return lat_lon_binned


def count_cells(da):
    """
    Count number of data points in binned grid cells.
    """
    lat_cells = da.groupby_bins("latitude", lat_bins, labels=lat_center).count(
        dim="latitude"
    )
    lon_cells = lat_cells.groupby_bins("longitude", lon_bins, labels=lon_center).sum(
        dim="longitude"
    )
    return lon_cells


if __name__ == "__main__":
    ds_classifications_input = xr.open_zarr(fn_zarr)

    # Create file and calculate common boxes
    print("Create output file")
    nb_times = len(ds_classifications_input.time)
    nb_lons = len(lon_center)
    nb_lats = len(lat_center)
    nb_patterns = len(ds_classifications_input.pattern)

    pixel_per_gridcell = count_cells(
        ds_classifications_input.mask.isel(time=0, pattern=0)
    )

    if os.path.exists(output_file) and not overwrite:
        print("File already exists.")
        root_grp = zarr.open(output_file)
        count = root_grp["counts"]
        times = root_grp["time"]
        store = root_grp.store
    else:
        print("File will be newly created.")
        store = zarr.DirectoryStore(output_file)
        root_grp = zarr.group(store, overwrite=True)
        if filter:
            domain_covered = root_grp.create_dataset(
                "domain_covered",
                shape=(nb_times, nb_lons, nb_lats),
                chunks=(1, nb_lons, nb_lats),
                dtype=float,
                compressor=zarr.Zlib(level=1),
                fill_value=-1,
            )
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
            fill_value=None,
            compressor=zarr.Zlib(level=1),
        )
        lats = root_grp.create_dataset(
            "latitude",
            shape=(nb_lats),
            chunks=(nb_lats),
            dtype=float,
            fill_value=None,
            compressor=zarr.Zlib(level=1),
        )
        lons = root_grp.create_dataset(
            "longitude",
            shape=(nb_lons),
            chunks=(nb_lons),
            dtype=float,
            fill_value=None,
            compressor=zarr.Zlib(level=1),
        )
        patterns = root_grp.create_dataset(
            "class",
            shape=(nb_patterns),
            chunks=(nb_patterns),
            dtype=str,
            fill_value=None,
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
        ] = "percentage of classifications of particular pattern within 2.5 deg x 2.5 deg"
        if filter:
            domain_covered.attrs["_ARRAY_DIMENSIONS"] = (
                "time",
                "longitude",
                "latitude",
            )
            domain_covered.attrs[
                "description"
            ] = f"percentage of 2.5 deg x 2.5 deg domain covered by satellite footprint and not covered by high clouds (BT < {BT_THRESHOLD}K)"
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
    slice_size = 1
    if filter:
        # Pixel outside footprint or covered by high clouds
        retrieval_mask = ds_classifications_input.sat_mask & (
            ds_classifications_input.sat_bt < BT_THRESHOLD
        )
        # Pattern detected and covered by satellite footprint
        mask = ds_classifications_input.mask & retrieval_mask
    else:
        mask = ds_classifications_input.mask
    mean = calculate_mean(mask)

    if filter:
        retrieval_sum = calculate_mean(retrieval_mask)
        mean = (
            dask.array.true_divide(
                mean,
                retrieval_sum,
                out=dask.array.zeros_like(mean),
                where=(retrieval_sum > 0).expand_dims("pattern", axis=-1),
            )
            * 100
        )
        retrieval_perc = retrieval_sum / pixel_per_gridcell * 100
    else:
        mean = mean / pixel_per_gridcell
    for d in tqdm.tqdm(range(len(mean.time))):
        count[d : d + slice_size, :, :, :] = mean.isel(
            time=slice(d, d + slice_size)
        ).values
        if filter:
            domain_covered[d : d + slice_size, :, :] = retrieval_perc.isel(
                time=slice(d, d + slice_size)
            ).values
        times[d : d + slice_size] = mean.isel(time=slice(d, d + slice_size)).time.values
        d += slice_size

    zarr.consolidate_metadata(store)
