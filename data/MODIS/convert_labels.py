"""Merge pkl files of classification_world.py to something useful
"""
import datetime as dt
import sys
from glob import glob

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm, trange

label_files = np.array(sorted(glob("world_pattern_*.pkl")))


def wh2xy(x, y, w, h):
    """Converts [x, y, w, h] to [x1, y1, x2, y2], i.e. bottom left and top right coords."""
    return x, y, x + w, y + h


def create_mask(
    boxes, labels, out, label_map={"Sugar": 0, "Fish": 1, "Flower": 2, "Gravel": 3}
):
    """
    Create or add mask to array
    """
    xy_boxes = [wh2xy(*b) for b in boxes]

    for i, lab in enumerate(labels):
        mask_layer = label_map[lab]
        x1, y1, x2, y2 = np.array(xy_boxes[i]).astype(int)
        out[x1:x2, y1:y2, mask_layer] += 1

    return out


# Correct the index
# Because the retrieval of the labels had to be splitted due to computing time limitations, the index starts for every file at 0. We make it monotonical increasing again
dataframes = [None] * len(label_files)
for f, file in enumerate(label_files):
    dataframes[f] = pd.read_pickle(file)

df_all = pd.concat(dataframes)
df_all["month"] = df_all.index.month
df_all["date"] = df_all.index
print(len(df_all.index))
df_all = df_all.drop_duplicates(subset="date")
print(len(df_all.index))


# Daily pattern mask
# Create file and calculate common boxes
nb_days = len(df_all.index)
nb_lons = 11000 // 2
nb_lats = 6500 // 2
nb_patterns = 4

store = zarr.DirectoryStore(
    "IR_worldview_everyotherline_NH_daily_pattern_distribution_2003-2009.zarr"
)
root_grp = zarr.group(store, overwrite=True)
mask = root_grp.create_dataset(
    "mask",
    shape=(nb_days, nb_lons, nb_lats, nb_patterns),
    chunks=(1, nb_lons, nb_lats, nb_patterns),
    dtype=bool,
    compressor=zarr.Zlib(level=1),
)
dates = root_grp.create_dataset(
    "dates", shape=(nb_days), chunks=(1), dtype="<M8[ns]", compressor=zarr.Zlib(level=1)
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
    "pattern",
    shape=(nb_patterns),
    chunks=(nb_patterns),
    dtype=str,
    compressor=zarr.Zlib(level=1),
)

lons[:] = np.linspace(-100, 10, nb_lons)
lats[:] = np.linspace(55, -10, nb_lats)
patterns[:] = ["Sugar", "Fish", "Flowers", "Gravel"]

# Add attributes to file
# Variable attributes
mask.attrs["_ARRAY_DIMENSIONS"] = ("dates", "longitude", "latitude", "pattern")
mask.attrs[
    "description"
] = "classification mask for every single pattern and classification_id"
lons.attrs["_ARRAY_DIMENSIONS"] = "longitude"
lons.attrs["standard_name"] = "longitude"
lons.attrs["units"] = "degree_east"
lats.attrs["_ARRAY_DIMENSIONS"] = "latitude"
lats.attrs["standard_name"] = "latitude"
lats.attrs["units"] = "degree_north"
dates.attrs["_ARRAY_DIMENSIONS"] = "dates"
dates.attrs[
    "description"
] = "classification id (basically each sighting of an image has a unique id)"
patterns.attrs["_ARRAY_DIMENSIONS"] = "pattern"

# Global attributes
root_grp.attrs[
    "title"
] = "EUREC4A: IR neural network meso-scale cloud pattern classifications"
root_grp.attrs[
    "description"
] = "North Atlantic NN detections of meso-scale cloud patterns (windowmethod)"
root_grp.attrs["author"] = "Hauke Schulz (hauke.schulz@mpimet.mpg.de)"
root_grp.attrs["institute"] = "Max Planck Institut für Meteorologie, Germany"
root_grp.attrs["created_on"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
# root_grp.attrs['created_with'] = os.path.basename(__file__) + " with its last modification on " + time.ctime(
#             os.path.getmtime(os.path.realpath(__file__)))
# root_grp.attrs['version'] = git_module_version
root_grp.attrs["python_version"] = f"{sys.version}"

for i, index in enumerate(tqdm(df_all.index)):
    o = np.empty((nb_lons, nb_lats, nb_patterns), dtype="bool")
    create_mask(df_all.loc[index]["boxes"], df_all.loc[index]["labels"], out=o)
    mask[i, :, :, :] = o
    dates[i] = index.to_datetime64()
