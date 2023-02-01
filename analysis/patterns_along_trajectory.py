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
fn_pattern_at_traj = "../data/result/patterns_along_trajectories_SGFF.pq"
traj_fn = "../data/ERA5/trajectories/Trajectories_back_ClassificationCenters_IR.nc"
# Output
fn_figure = "../figures/patterns_along_trajectory_SGFF_allTimes.pdf"

threshold_count = 0
include_no_class = True
# which timesteps a pattern need to persist
pattern_start_persistance = slice(0, 1)

pattern_sequence = pd.read_parquet(fn_pattern_at_traj)
pattern_sequence[pattern_sequence == 0] = np.nan

## Filter trajectories by starting/ending pattern
starting_patterns = pattern_sequence.loc[
    (slice(None), pattern_start_persistance), :
]  # start is def. as init of backtrajectory

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
    if include_no_class:
        np.isnan(pattern_sequence.loc[trajs_with_spec_pattern]).all(axis=1).groupby(
            "timestep"
        ).sum().plot(ax=axs.flatten()[p], color="grey")
    axs.flatten()[p].set_title(pattern)
    if p % 2 == 0:
        axs.flatten()[p].set_ylabel("N")
    if p == 1:
        lgd = axs.flatten()[p].legend(bbox_to_anchor=(1.1, 0), loc="lower left")
    else:
        lgd = axs.flatten()[p].legend()
        lgd.remove()
sns.despine()
plt.savefig(fn_figure, bbox_inches="tight")
