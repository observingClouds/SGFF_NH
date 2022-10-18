import dask
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import xarray as xr
import zarr

input_file = "../data/Daily_1x1_MODIS-IR_NorthAtlantic_SGFF.zarr"  # zarr input
output_file = "../data/Daily_1x1_MODIS-IR_NorthAtlantic_SGFF.nc"  # netCDF output
ds_in = xr.open_zarr(input_file)


ds_in_dropped_NaTs = ds_in.sel(time=~np.isnat(ds_in.time))

ds_in_dropped_NaTs.to_netcdf(
    output_file,
    encoding={
        "counts": {"zlib": True, "dtype": "int", "_FillValue": 0},
        "time": {"dtype": "object"},
        "class": {"_FillValue": None},
    },
)

ds_in2 = xr.open_dataset(output_file)
ds_in2["counts"] = ds_in2.counts[:, :, :, [0, 3, 2, 1]]
ds_in2["counts"].data = ds_in2.counts.values[:, :, :, [0, 3, 2, 1]]

ds_in2.counts.attrs["units"] = ""
ds_in2.counts.attrs["cell_methods"] = "area: count"
ds_in2.counts.attrs[
    "long_name"
] = "Number of cells with a particular pattern within each 1deg x 1deg box"
ds_in2["latitude_bnds"] = xr.DataArray(
    np.array([np.arange(-10, 54, 1), np.arange(-9, 55, 1)]).T, dims=["latitude", "bnds"]
)
ds_in2["longitude_bnds"] = xr.DataArray(
    np.array([np.arange(-100, 9, 1), np.arange(-99, 10, 1)]).T,
    dims=["longitude", "bnds"],
)
ds_in2["longitude"].attrs["bounds"] = "longitude_bnds"
ds_in2["longitude"].attrs["standard_name"] = "longitude"
ds_in2["longitude"].attrs["units"] = "degree_east"
ds_in2["latitude"].attrs["bounds"] = "latitude_bnds"
ds_in2["latitude"].attrs["standard_name"] = "latitude"
ds_in2["latitude"].attrs["units"] = "degree_north"
ds_in2.attrs["history"] += "Switching Gravel/Fish labels"
ds_in2.to_netcdf(
    output_file + "2",
    encoding={
        "counts": {"zlib": True, "dtype": "int", "_FillValue": 0},
        "time": {"dtype": "object"},
        "class": {"_FillValue": None},
    },
)
