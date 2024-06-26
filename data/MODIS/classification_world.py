import datetime as dt
import os
from glob import glob

import fire
import keras
import keras_retinanet
import numpy as np
import pandas as pd
import skimage
import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import preprocess_image, read_image_bgr, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm as tqdm


def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


def xy2wh(x1, y1, x2, y2):
    return [x1, y1, x2 - x1, y2 - y1]


def get_retinanet_preds(
    model,
    fn,
    thresh=0.3,
    min_side=800,
    max_side=1050,
    subset=False,
    block_lat_extend=500,
    block_lon_extend=1000,
    method="windows",
    window_lat_extend=400,
    window_lon_extend=800,
):
    """
    Input
    -----
    method : str
        method describing how to subset images e.g. blocks or windows
    """
    image = read_image_bgr(fn)
    image = preprocess_image(image[::2, ::2])
    boxes_all = []
    scores_all = []
    labels_all = []
    window = np.array([window_lat_extend, window_lon_extend, 3])
    block = np.array([block_lat_extend, block_lon_extend, 3])
    assert np.all(
        (np.shape(image) - block) % window == 0
    ), f"Window size, block size and image size do not fit (image: {np.shape(image)} block: {block}, window: {window})"
    if subset:
        # Split image in smaller portions
        try:
            if method == "blocks":
                image_blocks = skimage.util.view_as_blocks(
                    image,
                    block,
                )
            elif method == "windows":
                image_blocks = skimage.util.view_as_windows(
                    image,
                    block,
                    window,
                )
        except ValueError:
            print("Shape of image is ", np.shape(image))
        for lon_box in range(image_blocks.shape[1]):
            for lat_box in range(image_blocks.shape[0]):
                image = image_blocks[lat_box, lon_box, :, :, :].squeeze()
                image, scale = resize_image(image, min_side, max_side)
                boxes, scores, labels = (
                    o[0] for o in model.predict_on_batch(np.expand_dims(image, axis=0))
                )
                boxes /= scale
                boxes = boxes[scores > thresh]
                if len(boxes) >= 1 and method == "blocks":
                    boxes[:, 0] += lon_box * block_lon_extend
                    boxes[:, 1] += lat_box * block_lat_extend
                    boxes[:, 2] += lon_box * block_lon_extend
                    boxes[:, 3] += lat_box * block_lat_extend
                elif len(boxes) >= 1 and method == "windows":
                    boxes[:, 0] += lon_box * window_lon_extend
                    boxes[:, 1] += lat_box * window_lat_extend
                    boxes[:, 2] += lon_box * window_lon_extend
                    boxes[:, 3] += lat_box * window_lat_extend
                boxes = [xy2wh(*b) for b in boxes]
                labels = labels[scores > thresh]
                labels = [labels_to_names[i] for i in labels]
                scores = scores[scores > thresh]
                boxes_all.extend(boxes)
                scores_all.extend(scores)
                labels_all.extend(labels)
    return np.array(boxes_all), labels_all, scores_all


def read_input(experiment):
    conf = OmegaConf.load(experiment)
    return conf


if __name__ == "__main__":
    conf = fire.Fire(read_input)
    overwrite = True
    # use this environment flag to change which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # set the modified tf session as backend in keras
    tf.compat.v1.keras.backend.set_session(get_session())

    # !keras_retinanet/bin/convert_model.py /path/to/training/model.h5 /path/to/save/inference/model.h5
    model = models.load_model(conf["classification"]["model"], backbone_name="resnet50")

    labels_to_names = {
        i: l for i, l in enumerate(["Flower", "Fish", "Gravel", "Sugar"])
    }

    block_lat_extend = conf["classification"]["classification_window_size"][0]
    block_lon_extend = conf["classification"]["classification_window_size"][1]
    window_lat_extend = conf["classification"]["classification_window_increment"][0]
    window_lon_extend = conf["classification"]["classification_window_increment"][1]

    files = sorted(glob(conf["classification"]["input_images_fmt"]))

    print("Files to process: ", len(files))

    # Remove files that are to small ( error during download )
    file_sizes = [os.path.getsize(f) for f in files]
    file_mask = np.array(file_sizes) > 2000000
    files = np.array(files)[file_mask]

    times = np.empty(len(files), dtype=dt.datetime)

    sub_ind = [0, 10]
    sub_ind.extend(np.arange(11, (len(times) - 10) + 500, 500))
    for s in range(1, len(sub_ind)):
        result_dict = {}
        s1 = sub_ind[s - 1]
        s2 = sub_ind[s]
        output_pkl = conf["classification"]["output_pkl_fmt"].format(s1=s1, s2=s2)
        if os.path.exists(output_pkl) and overwrite is False:
            continue
        for f, file in enumerate(tqdm(files[s1:s2])):
            time_str = file.split("_")[-4][-8:]
            times[f] = dt.datetime.strptime(time_str, "%Y%m%d")
            boxes, labels, scores = get_retinanet_preds(
                model,
                file,
                0.5,
                subset=True,
                block_lat_extend=block_lat_extend,
                block_lon_extend=block_lon_extend,
                window_lat_extend=window_lat_extend,
                window_lon_extend=window_lon_extend,
            )

            result_dict[times[f]] = {"boxes": boxes, "labels": labels, "scores": scores}

        df = pd.DataFrame.from_dict(result_dict, orient="index")
        df.head()
        if not os.path.exists(os.path.dirname(output_pkl)):
            os.makedirs(os.path.dirname(output_pkl))
        df.to_pickle(output_pkl)
