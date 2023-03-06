"""
Script to download images from NASA Worldview
"""
import datetime
import os
import urllib.request
from calendar import monthrange

import fire
from omegaconf import OmegaConf


def download_MODIS_imgs(
    experiment=None,
    year_range=[],
    months=[],
    save_path="./",
    lon_range=[None, None],
    lat_range=[None, None],
    deg2pix=100,
    satellite="Aqua",
    exist_skip=False,
    var="CorrectedReflectance_TrueColor",
    filetype="jpeg",
):
    assert (experiment is not None) or all(
        [v is not None for v in [year_range, months, save_path, lon_range, lat_range]]
    )
    if experiment is not None:
        print("Loading experiment")
        conf = OmegaConf.load(experiment)
    else:
        conf = conf = OmegaConf.create({})

    lon1 = OmegaConf.select(conf, "domain.lon_min", default=lon_range[0])
    lon2 = OmegaConf.select(conf, "domain.lon_max", default=lon_range[1])
    lat1 = OmegaConf.select(conf, "domain.lat_min", default=lat_range[0])
    lat2 = OmegaConf.select(conf, "domain.lat_max", default=lat_range[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    year_range = OmegaConf.select(conf, "year_range", default=year_range)
    months = OmegaConf.select(conf, "months", default=months)
    save_path = OmegaConf.select(conf, "output_dir_images", default=save_path)
    var = OmegaConf.select(conf, "variable", default=var)
    satellite = OmegaConf.select(conf, "satellite", default=satellite)
    loc = f"&BBOX={lat1},{lon1},{lat2},{lon2}"
    loc_str = f"_{lon1}-{lon2}_{lat1}-{lat2}"
    size = f"&WIDTH={int(dlon * deg2pix)}&HEIGHT={int(dlat * deg2pix)}"
    layer = f"&LAYERS=MODIS_{satellite}_{var}"  # ,Coastlines"

    os.makedirs(save_path, exist_ok=True)

    for yr in range(year_range[0], year_range[1]):
        for m in months:
            nday = monthrange(yr, m)[1]
            for nd in range(1, nday + 1):
                date = datetime.datetime(yr, m, nd)
                print(date.strftime("%y %m %d"))
                url = (
                    "https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME="
                    + date.strftime("%Y-%m-%d")
                    + loc
                    + "&CRS=EPSG:4326"
                    + layer
                    + "&FORMAT=image/"
                    + filetype
                    + size
                )
                save_str = (
                    save_path
                    + f"/{satellite}_"
                    + var
                    + str(yr)
                    + date.strftime("%m")
                    + f"{date.day:02d}"
                    + loc_str
                    + "."
                    + filetype
                )
                if exist_skip and os.path.exists(save_str):
                    print("Skip")
                else:
                    try:
                        urllib.request.urlretrieve(url, save_str)
                    except:
                        print(f"Download failed for {save_str} ({url})")


if __name__ == "__main__":
    fire.Fire(download_MODIS_imgs)
