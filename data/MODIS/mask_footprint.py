"""Mask classifications that are outside the satellite's footprint."""
import glob
import xml.etree.ElementTree as ET

import cv2
import fire
import matplotlib.colors
import numpy as np
import scipy.cluster.vq as scv
import tqdm
import xarray as xr
import zarr
from omegaconf import OmegaConf


def pixel_mask(image, color=[0, 0, 0]):
    """Create a mask for certain colored pixels."""
    return (image == color).all(axis=2)


def load_image(date, path_fmt):
    """Load an image from a path."""
    pattern = path_fmt.replace("*", f"*{date.strftime('%Y%m%d')}*")
    try:
        file = glob.glob(pattern)[0]
        image = cv2.imread(file)
        return image
    except IndexError:
        print(f"Could not find image matching pattern {pattern}.")
        return None


def read_input(experiment):
    conf = OmegaConf.load(experiment)
    return conf


def create_colormap_from_xml(filename, cmap_name="MODIS_BT"):
    tree = ET.parse(filename)
    root = tree.getroot()

    rgb = []
    cmap_label = []

    for elem in root:
        rgb.append(tuple(np.array((elem.get("rgb") + ",1").split(","), dtype=int)))
        cmap_label.append(float(elem.get("label").split(" ")[0]))
    rgb = np.vstack(rgb) / 255
    rgb[:, 3] = 1

    # Creating colormap
    cmap = matplotlib.colors.ListedColormap(rgb, cmap_name)
    return cmap, cmap_label


def shifting(arr, gradient):
    code, dist = scv.vq(arr, gradient, check_finite=True)
    return code, dist


def imagecolor2data(img_arr, data_range, cmap):
    """
    Converts RGB color values to data values

    Image data contains only RGB values,
    but if the colormap and the original
    data range is know, the data information
    can be retrieved again.

    Source
    ------
    adapted from http://stackoverflow.com/questions/3720840/how-to-reverse-color-map-image-to-scalar-values/3722674#3722674
    """

    if isinstance(cmap, str):
        cmap = eval("cm." + cmap)
        gradient = cmap(np.linspace(0.0, 1.0, len(data_range)))
    elif isinstance(cmap, np.ndarray):
        gradient = cmap
    elif isinstance(cmap, matplotlib.colors.ListedColormap):
        gradient = cmap(np.linspace(0.0, 1.0, len(data_range)))

    # Reshape arr to something like (240*240, 4), all the 4-tuples in a long list...
    arr2 = img_arr.reshape((img_arr.shape[0] * img_arr.shape[1], img_arr.shape[2]))

    # Use vector quantization to shift the values in arr2 to the nearest point in
    # the code book (gradient).
    code, dist = shifting(arr2, gradient)

    # code is an array of length arr2 (240*240), holding the code book index for
    # each observation. (arr2 are the "observations".)
    # Scale the values so they are from 0 to 1.
    values = code.astype("float") / gradient.shape[0]

    # Reshape values back to (240,240)
    values = values.reshape(img_arr.shape[0], img_arr.shape[1])
    values = values[::-1]

    # Transform the values from 0..1 to actual data
    lookup_table = dict(
        zip(np.linspace(0, 1, len(data_range) + 1).round(3), data_range)
    )
    data_values = np.vectorize(lookup_table.get)(values.round(3))

    return data_values


def color2data(image, colormap, cmap_label):
    """Convert colored image to data based on given colormap."""
    img = image[::-1, :, :]
    x = np.zeros(np.shape(img)[0:2]) + 1
    x = x[:, :, None]
    im2 = np.concatenate([np.array(img), x], axis=2)

    # Create mask where no data has been gathered (in principle threshold should be ==1),
    # but between data and no data are spurious colors
    no_data_mask = np.where(np.sum(im2, axis=-1) <= 20, 1, 0)
    data = imagecolor2data(im2 / 255, np.array(cmap_label), colormap)
    data[no_data_mask.astype(bool)[::-1]] = np.nan
    return data


if __name__ == "__main__":
    conf = fire.Fire(read_input)
    sat_image_path_fmt = conf["classification"]["input_images_fmt"]
    classification_path = conf["classification"]["output_zarr"]
    output_path = conf["classification"]["output_zarr_masked"]

    cmap, cmap_label = create_colormap_from_xml("data/MODIS/worldview_BT_colormap.xml")

    ds = xr.open_zarr(classification_path)

    store = zarr.DirectoryStore(classification_path)
    root_grp = zarr.group(store, overwrite=False)
    sat_mask = root_grp.create_dataset(
        "sat_mask",
        shape=ds.mask.shape[:-1],
        chunks=ds.mask.encoding["chunks"][:-1],
        dtype=bool,
        fill_value=None,
        compressor=zarr.Zlib(level=1),
    )
    sat_mask.attrs["_ARRAY_DIMENSIONS"] = ("time", "longitude", "latitude")
    sat_mask.attrs[
        "description"
    ] = "mask of valid satellite retrievals, e.g. masks gaps due to footprint limitations"

    sat_bt = root_grp.create_dataset(
        "sat_bt",
        shape=ds.mask.shape[:-1],
        chunks=ds.mask.encoding["chunks"][:-1],
        dtype="f2",
        compressor=zarr.Zlib(level=1),
    )
    sat_bt.attrs["_ARRAY_DIMENSIONS"] = ("time", "longitude", "latitude")
    sat_bt.attrs[
        "description"
    ] = "approximate brightness temperature of satellite retrievals"
    sat_bt.attrs["units"] = "K"

    for t, time in enumerate(tqdm.tqdm(ds.time)):
        daily_ds = ds.sel(time=time)
        sat_image = load_image(time.dt.date.values.item(0), sat_image_path_fmt)[
            ::2, ::2
        ]
        if sat_image is None:
            print("Failure to load image.")
            continue
        else:
            sat_mask[t, :, :] = ~pixel_mask(sat_image).T
            bt = color2data(sat_image, cmap, cmap_label)
            sat_bt[t, :, :] = bt.T

    zarr.consolidate_metadata(store)
    store.close()
