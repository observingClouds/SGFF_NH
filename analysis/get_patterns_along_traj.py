import os
import sys

import numpy as np
import pandas as pd
import tqdm
import xarray as xr

sys.path.append("../src/helpers")
import mesoscale_dataset_helpers as mdh  # noqa: E402
import traj_helpers as th  # noqa: E402

# Input
traj_fn = "../data/ERA5/trajectories/Trajectories_back_ClassificationCenters_IR.nc"
SGFF_fn = "../data/SGFF/Daily_1x1_MODIS-IR_NorthAtlantic_SGFF.nc"
MCC_fn = "../data/MCC/Daily_1x1_MODIS_C6_MCC_2018.nc"
MEASURES_fn = "../data/MEASURES/Daily_1x1_MEASURES_CLASS_2018.nc"
# Output
fn_pattern_at_traj = "../data/result/patterns_along_trajectories_{ds}.pq"

testing = False  # run reduced datasample
overwrite = False
pattern_start_persistance = slice(
    0, 1
)  # how many timesteps does a pattern need to persist

ds_traj = xr.open_dataset(traj_fn)
ds_MCC = xr.open_dataset(MCC_fn)
ds_SGFF = xr.open_dataset(SGFF_fn)
ds_MEASURES = xr.open_dataset(MEASURES_fn)

# Postprocessing to fit SGFF format
ds_MCC = mdh.postprocess_classification_ds(ds_MCC)
ds_MEASURES = mdh.postprocess_classification_ds(ds_MEASURES)
## Give the classes some names
class_names = {
    "MCC": ["Open MCC", "Closed MCC", "Cellular but Disorganized"],
    "MEASURES": ["Closed-cellular", "Clustered Cu", "Disorganized MCC", "Open-cellular MCC", "Solid Stratus", "Suppressed Cu"],
    "SGFF": ["Sugar", "Gravel", "Flowers", "Fish"],
}
ds_MCC["class"] = class_names["MCC"]
ds_MEASURES["class"] = class_names["MEASURES"]

classification_datasets = {"MCC": ds_MCC, "MEASURES": ds_MEASURES, "SGFF": ds_SGFF}

for classification_ds, ds in classification_datasets.items():
    print(classification_ds)
    fn_out = fn_pattern_at_traj.format(ds=classification_ds)
    if os.path.exists(fn_out) and overwrite is False:
        pattern_sequence = pd.read_parquet(fn_out)
    else:
        # Get classifications along trajectory
        dfs = {}
        for i, t in enumerate(tqdm.tqdm(ds_traj.trajectory_idx.values)):
            traj = ds_traj.sel(trajectory_idx=t)
            patterns = th.get_patterns_along_traj(
                traj.time_at_traj, traj.lat_at_traj, traj.lon_at_traj, ds
            )
            dfs[t] = patterns
            if testing and i == 3:
                break
        pattern_sequence = pd.concat(dfs).rename_axis(index=["trajectory", "timestep"])
        if not os.path.exists(os.path.dirname(fn_out)):
            os.mkdir(os.path.dirname(fn_out))
        pattern_sequence.to_parquet(fn_out)
