import numpy as np
import pandas as pd


def get_patterns_along_traj(time_at_traj, lat_at_traj, lon_at_traj, pattern_ds):
    patterns_along_traj = {}
    for t, time in enumerate(time_at_traj):
        try:
            ds_time_sel = pattern_ds.sel(time=time, method="nearest", tolerance="12H")
            patterns_along_traj[t] = ds_time_sel.sel(
                latitude=lat_at_traj[t],
                longitude=lon_at_traj[t],
                method="nearest",
                tolerance=1,
            )["counts"].values
        except KeyError:
            patterns_along_traj[t] = np.array([np.nan, np.nan, np.nan, np.nan])
    df = pd.DataFrame.from_dict(
        patterns_along_traj,
        orient="index",
        columns=["Sugar", "Gravel", "Flowers", "Fish"],
    )
    return df
