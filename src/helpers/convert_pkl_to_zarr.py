import datetime as dt
import os
import sys
from glob import glob

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from omegaconf import OmegaConf
from tqdm import tqdm, trange


def get_setup(conf_file):
    conf = OmegaConf.load(conf_file)

    lon1 = conf["domain"]["lon_min"]
    lon2 = conf["domain"]["lon_max"]
    lat1 = conf["domain"]["lat_min"]
    lat2 = conf["domain"]["lat_max"]

    domain = [lat1, lat2, lon1, lon2]

    merged_pickle_out = conf["classification"]["output_pkl_joint"]
    label_file_fmt = conf["classification"]["output_pkl_fmt"].replace(
        "{s1:04g}-{s2:04g}", "*"
    )
    output_file = conf["classification"]["output_zarr"]

    return domain, merged_pickle_out, label_file_fmt, output_file


if __name__ == "__main__":
    domain, merged_pickle_out, label_file_fmt, output_file = fire.Fire(get_setup)

    label_map = {"Sugar": 0, "Fish": 3, "Flowers": 2, "Flower": 2, "Gravel": 1}
    label_map_rv = {0: "Sugar", 1: "Gravel", 2: "Flowers", 3: "Fish"}

    # Load labels
    label_files = np.array(sorted(glob(label_file_fmt)))
    print(label_files)
    dataframes = [None] * len(label_files)
    for f, file in enumerate(label_files):
        dataframes[f] = pd.read_pickle(file)

    df_all = pd.concat(dataframes)
    df_all.head()

    df_all.reset_index(inplace=True)

    # Export concatenated dataframe as pickle
    outdir = os.path.dirname(merged_pickle_out)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    df_all.to_pickle(merged_pickle_out)

    # Create file and calculate common boxes
    nb_times = len(df_all.index.unique())

    a = np.array([0, 0, 0, 0])
    c = np.array([0, 0])
    for box in df_all.boxes:
        if len(box) > 0:
            b = np.array(box).max(axis=0)
            a = np.array([a, b]).max(axis=0)
            c_ = np.array([box[:, 0] + box[:, 2], box[:, 1] + box[:, 3]]).max(axis=1)
            c = np.array([c, c_]).max(axis=0)

    nb_lats = int(np.ceil(c[1]))  # np.int(np.floor((df_all.y+df_all.h).max()))
    nb_lons = int(np.ceil(c[0]))  # np.int(np.floor((df_all.x+df_all.w).max()))
    nb_patterns = 4

    store = zarr.DirectoryStore(output_file)
    root_grp = zarr.group(store, overwrite=True)
    mask = root_grp.create_dataset(
        "mask",
        shape=(nb_times, nb_lons, nb_lats, nb_patterns),
        chunks=(1, nb_lons, nb_lats, nb_patterns),
        dtype=bool,
        fill_value=None,
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
        "pattern",
        shape=(nb_patterns),
        chunks=(nb_patterns),
        dtype=str,
        compressor=zarr.Zlib(level=1),
    )

    lons[:] = np.linspace(domain[2], domain[3], nb_lons)
    lats[:] = np.linspace(domain[1], domain[0], nb_lats)
    patterns[:] = [label_map_rv[i] for i in range(4)]

    # Add attributes to file
    # Variable attributes
    mask.attrs["_ARRAY_DIMENSIONS"] = ("time", "longitude", "latitude", "pattern")
    mask.attrs[
        "description"
    ] = "classification mask for every single pattern and classification_id"
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
    patterns.attrs["_ARRAY_DIMENSIONS"] = "pattern"

    # Global attributes
    root_grp.attrs[
        "title"
    ] = "IR neural network meso-scale cloud pattern classifications"
    root_grp.attrs["description"] = "NN detections of meso-scale cloud patterns"
    root_grp.attrs["author"] = "Hauke Schulz (haschulz@uw.edu)"
    root_grp.attrs["institute"] = "University of Washington"
    root_grp.attrs["created_on"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    # root_grp.attrs['created_with'] = os.path.basename(__file__) + " with its last modification on " + time.ctime(
    #             os.path.getmtime(os.path.realpath(__file__)))
    # root_grp.attrs['version'] = git_module_version
    root_grp.attrs["python_version"] = f"{sys.version}"

    def wh2xy(x, y, w, h):
        """
        Converts [x, y, w, h] to [x1, y1, x2, y2], i.e. bottom left and top right coords.
        >>> helpers.wh2xy(10, 20, 30, 1480)
        (10, 20, 39, 1499)
        """
        return x, y, x + w - 1, y + h - 1

    def create_mask(
        boxes, labels, out, label_map={"Sugar": 0, "Fish": 3, "Flower": 2, "Gravel": 1}
    ):
        """
        Create or add mask to array
        """
        xy_boxes = [wh2xy(*b) for b in boxes]

        for nlab, lab in enumerate(labels):
            mask_layer = label_map[lab]
            x1, y1, x2, y2 = np.array(xy_boxes[nlab]).astype(int)
            out[x1:x2, y1:y2, mask_layer] = True

        return out

    for i, (index, idx_grp) in enumerate(tqdm(df_all.groupby(df_all.index.unique()))):
        o = np.zeros((nb_lons, nb_lats, nb_patterns), dtype=bool)
        assert np.shape(idx_grp["boxes"].values)[0] == 1, "More than one date entry!"
        if len(idx_grp["boxes"].index) > 0:
            create_mask(
                idx_grp["boxes"].iloc[0],
                idx_grp["labels"].iloc[0],
                out=o,
                label_map=label_map,
            )
        else:
            print(f"No labels found on {index}")
        mask[i, :, :, :] = o
        times[i] = idx_grp["index"].item()

    zarr.consolidate_metadata(store)
    print("Zarr file written")
