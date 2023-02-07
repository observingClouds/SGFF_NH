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

# # Meso-scale classifications along trajectories from all available datasets

# For 2018 several classification datasets exists, which should be compared in this section


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
import mesoscale_dataset_helpers as mdh  # noqa: E402
import traj_helpers as th  # noqa: E402

# Input
fn_pattern_at_traj = "../data/result/patterns_along_trajectories_{ds}.pq"
traj_fn = "../data/ERA5/trajectories/Trajectories_back_ClassificationCenters_IR.nc"
# Output
fn_figure = "../figures/patterns_along_trajectory_{ds}.pdf"

threshold_count = 0
pattern_start_persistance = slice(
    0, 1
)  # how many timesteps does a pattern need to persist
class_names = {
    "MCC": ["Open MCC", "Closed MCC", "Cellular but Disorganized"],
    "MEASURES": [
        "Closed-cellular",
        "Clustered Cu",
        "Disorganized MCC",
        "Open-cellular MCC",
        "Solid Stratus",
        "Suppressed Cu",
    ],
    "SGFF": ["Sugar", "Gravel", "Flowers", "Fish"],
}
classification_datasets = {"SGFF": {}, "MCC": {}, "MEASURES": {}}

for classification_ds in classification_datasets.keys():
    ds_class = pd.read_parquet(fn_pattern_at_traj.format(ds=classification_ds))
    classification_datasets[classification_ds] = ds_class

# Select trajectoriess in year of interest
ds_traj = xr.open_dataset(traj_fn)
traj_mask = ds_traj.time_at_traj.sel(index=0).dt.year == 2018
ds_traj_sel = ds_traj.sel(trajectory_idx=traj_mask.values)

# Combine sequences
sequences = {}
for classification_ds, ds in classification_datasets.items():
    sequences[classification_ds] = ds
pattern_sequence = pd.concat(sequences, axis=1, levels=[])

# +
## Filter trajectories by starting/ending pattern

starting_patterns = pattern_sequence.loc[
    (slice(None), pattern_start_persistance), :
]  # start is def. as init of backtrajectory
# -

pattern_sequence[pattern_sequence == 0] = np.nan

# Plot how frequently specific patterns are detected along a trajectory
# when the patterns at time of initialization is given
for classification_ds, pattern_grp in class_names.items():
    rows = int(np.ceil(len(pattern_grp) / 2))
    fig, axs = plt.subplots(rows, 2, figsize=(16, 4 * rows), sharex=True)
    for p, pattern in enumerate(pattern_grp):
        mask = (
            (starting_patterns[pattern] > threshold_count).groupby("trajectory").all()
        )
        trajs_with_spec_pattern = mask.index.get_level_values(0)[
            mask.values.reshape(-1)
        ]
        pattern_sequence.loc[trajs_with_spec_pattern].groupby("timestep").count().plot(
            ax=axs.flatten()[p], cmap="tab20"
        )
        axs.flatten()[p].set_title(pattern)
        if p % 2 == 0:
            axs.flatten()[p].set_ylabel("N")
        if p == 1:
            lgd = axs.flatten()[p].legend(bbox_to_anchor=(1.1, 0), loc="lower left")
        else:
            lgd = axs.flatten()[p].legend()
            lgd.remove()

    sns.despine()
    plt.savefig(fn_figure.format(ds=classification_ds), bbox_inches="tight")
