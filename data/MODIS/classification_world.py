import datetime as dt
import os
from glob import glob

import keras
import keras_retinanet
import numpy as np
import pandas as pd
import skimage
import tensorflow as tf
import tensorflow.keras.backend as K
from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import preprocess_image, read_image_bgr, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from PIL import Image
from tqdm import tqdm as tqdm

model_path_IR = "/work/mh0010/from_Mistral/mh0010/m300408/CharacterizationOfMesoscalePatterns/Classification/NN-training/inference_models/resnet50_csv_11.h5"
model_path_VIS = "/work/mh0010/from_Mistral/mh0010/m300408/CharacterizationOfMesoscalePatterns/Classification/NN-training/inference_models/exp5_resnet50_csv_20_inference.h5"
output_labels = "labels.csv"

model_type = "IR"  # Choose between IR and VIS
input_files_fmt = "/work/mh0010/from_Mistral/mh0010/m300408/CloudMorphology/data/MODIS/Aqua_MODIS_IR/Aqua_Brightness_Temp_Band31_Day*.jpeg"


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def xy2wh(x1, y1, x2, y2):
    return [x1, y1, x2 - x1, y2 - y1]


# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set the modified tf session as backend in keras
K.tensorflow_backend.set_session(get_session())

# !keras_retinanet/bin/convert_model.py /path/to/training/model.h5 /path/to/save/inference/model.h5
if model_type == "VIS":
    model = models.load_model(model_path_VIS, backbone_name="resnet50")
elif model_type == "IR":
    model = models.load_model(model_path_IR, backbone_name="resnet50")

labels_to_names = {i: l for i, l in enumerate(["Flower", "Fish", "Gravel", "Sugar"])}


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
    image = preprocess_image(image[::2, ::2, :])
    boxes_all = []
    scores_all = []
    labels_all = []
    if subset:
        # Split image in smaller portions
        try:
            if method == "blocks":
                image_blocks = skimage.util.view_as_blocks(
                    image, (block_lat_extend, block_lon_extend, 3)
                )
            elif method == "windows":
                image_blocks = skimage.util.view_as_windows(
                    image,
                    (block_lat_extend, block_lon_extend, 3),
                    (window_lat_extend, window_lon_extend, 3),
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


files = sorted(glob(input_files_fmt))

print("Files to process: ", len(files))

# Remove files that are to small ( error during download )
file_sizes = [os.path.getsize(f) for f in files]
file_mask = np.array(file_sizes) > 2000000
files = np.array(files)[file_mask]


# out = np.zeros((len(files), 18000, 7000, 4), dtype='bool')
times = np.empty(len(files), dtype=dt.datetime)
result_dict = {}

sub_ind = [0, 500, 1000, 1500, 2000, 2500, 3000]
for s in range(1, len(sub_ind)):
    s1 = sub_ind[s - 1]
    s2 = sub_ind[s]
    for f, file in enumerate(tqdm(files[s1:s2])):
        time_str = file[-28:-20]
        times[f] = dt.datetime.strptime(time_str, "%Y%m%d")
        boxes, labels, scores = get_retinanet_preds(model, file, 0.5, subset=True)

        result_dict[times[f]] = {"boxes": boxes, "labels": labels, "scores": scores}
    #     out[f,:,:,:] = create_mask(boxes, labels, out[f,:,:,:])

    df = pd.DataFrame.from_dict(result_dict, orient="index")
    df.head()
    df.to_pickle(
        f"world_pattern_windowmethod_everyotherline_{model_type}_{str(s1)}-{str(s2)}_202103.pkl"
    )