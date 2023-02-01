# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: covariability
#     language: python
#     name: covariability
# ---

# # Pattern classifications along trajectories

import os

## Load packages
import sys

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import xarray as xr

sys.path.append("../src/helpers")
import traj_helpers as th  # noqa: E402

# Input
traj_fn = "../data/ERA5/trajectories/Trajectories_back_ClassificationCenters_IR.nc"
class_fn = "../data/SGFF/Daily_1x1_MODIS-IR_NorthAtlantic_SGFF.nc"
# Output
fn_pattern_at_traj = "../data/result/patterns_along_trajectories.pq"
fn_figure = "../figures/patterns_along_trajectory.pdf"

threshold_count = 10
testing = False  # run reduced datasample
overwrite = False
pattern_start_persistance = slice(
    0, 1
)  # how many timesteps does a pattern need to persist

if os.path.exists(fn_pattern_at_traj) and overwrite is False:
    pattern_sequence = pd.read_parquet(fn_pattern_at_traj)
else:
    # Reading trajectories
    ds_traj = xr.open_dataset(traj_fn)
    # Reading classifications
    ds_class = xr.open_dataset(class_fn)
    # Get classifications along trajectory
    dfs = {}
    for i, t in enumerate(tqdm.tqdm(ds_traj.trajectory_idx.values)):
        traj = ds_traj.sel(trajectory_idx=t)
        patterns = th.get_patterns_along_traj(
            traj.time_at_traj, traj.lat_at_traj, traj.lon_at_traj, ds_class
        )
        dfs[t] = patterns
        if testing and i == 3:
            break
    pattern_sequence = pd.concat(dfs).rename_axis(index=["trajectory", "timestep"])
    if not os.path.exists(os.path.dirname(fn_pattern_at_traj)):
        os.mkdir(os.path.dirname(fn_pattern_at_traj))
    pattern_sequence.to_parquet(fn_pattern_at_traj)

## Filter trajectories by starting/ending pattern
starting_patterns = pattern_sequence.loc[
    (slice(None), 0), :
]  # start is def. as init of backtrajectory
ending_patterns = pattern_sequence.loc[(slice(None), 7), :]

# Create colormap specific to patterns
color_dict = {
    "Sugar": "#A1D791",
    "Gravel": "#3EAE47",
    "Flowers": "#93D2E2",
    "Fish": "#2281BB",
}
cmap_patterns = mc.ListedColormap(color_dict.values(), name="cmap_patterns")

# Plot how frequently specific patterns are detected along a trajectory
# when the patterns at time of initialization is given
fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
for p, pattern in enumerate(["Sugar", "Gravel", "Flowers", "Fish"]):
    mask = (starting_patterns[pattern] > threshold_count).groupby("trajectory").all()
    trajs_with_spec_pattern = mask.index.get_level_values(0)[mask]
    pattern_sequence.loc[trajs_with_spec_pattern].groupby("timestep").count().plot(
        ax=axs.flatten()[p], cmap=cmap_patterns
    )
    axs.flatten()[p].set_title(pattern)
    if p % 2 == 0:
        axs.flatten()[p].set_ylabel("N")
sns.despine()
plt.savefig(fn_figure, bbox_inches="tight")
