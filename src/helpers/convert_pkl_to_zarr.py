import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import xarray as xr
import zarr
import sys

import skimage
from tqdm import tqdm, trange

label_file_fmt = f'../../data/SGFF/level0/IR_????-????_IndianOcean.pkl'
merged_pickle_out = f'../../data/SGFF/level0/IR_IndianOcean.pkl'
output_file = f'../../data/SGFF/level1/IR_IndianOcean.zarr'
domain = [-15, 15, 60, 170]
label_map= {'Sugar':0, 'Fish': 3, 'Flowers': 2, 'Flower': 2, 'Gravel': 1}
label_map_rv = {0:'Sugar', 1:'Gravel', 2: 'Flowers', 3: 'Fish'}

# Load labels
label_files = np.array(sorted(glob(label_file_fmt)))
print(label_files)
dataframes = [None]*len(label_files)
for f,file in enumerate(label_files):
    dataframes[f] = pd.read_pickle(file)
    
import pdb;pdb.set_trace()
df_all = pd.concat(dataframes);
df_all.head()

if len(dataframes) > 1:
    dfs_to_merge = np.array(np.arange(len(dataframes)), dtype=int)
    df_all = pd.concat(np.array(dataframes)[dfs_to_merge]);
    df_all.head()
else:
    df_all = dataframes[0]

df_all.reset_index(inplace=True)

# Export concatenated dataframe as pickle
df_all.to_pickle(merged_pickle_out)

# Create file and calculate common boxes
nb_times = len(df_all.index.unique())

a = np.array([0,0,0,0])
c = np.array([0,0])
for box in df_all.boxes:
    if len(box) > 0:
        b=np.array(box).max(axis=0)
        a=np.array([a,b]).max(axis=0)
        c_=np.array([box[:,0] + box[:,2], box[:,1] + box[:,3]]).max(axis=1)
        c=np.array([c,c_]).max(axis=0)

nb_lats = np.int(np.ceil(c[1]))  # np.int(np.floor((df_all.y+df_all.h).max()))
nb_lons = np.int(np.ceil(c[0]))  # np.int(np.floor((df_all.x+df_all.w).max()))
nb_patterns = 4

store = zarr.DirectoryStore(output_file)
root_grp = zarr.group(store, overwrite=True)
mask = root_grp.create_dataset('mask', shape=(nb_times, nb_lons, nb_lats, nb_patterns),
                               chunks=(1, nb_lons, nb_lats, nb_patterns),
                               dtype=bool, compressor=zarr.Zlib(level=1))
times = root_grp.create_dataset('time', shape=(nb_times), chunks=(nb_times),
                                dtype='<M8[ns]',compressor=zarr.Zlib(level=1))
lats = root_grp.create_dataset('latitude', shape=(nb_lats), chunks=(nb_lats),
                               dtype=float, compressor=zarr.Zlib(level=1))
lons = root_grp.create_dataset('longitude', shape=(nb_lons), chunks=(nb_lons),
                               dtype=float, compressor=zarr.Zlib(level=1))
patterns = root_grp.create_dataset('pattern', shape=(nb_patterns), chunks=(nb_patterns),
                                   dtype=str, compressor=zarr.Zlib(level=1))

lons[:] = np.linspace(domain[2], domain[3], nb_lons)
lats[:] = np.linspace(domain[1], domain[0], nb_lats)
patterns[:] = [label_map_rv[i] for i in range(4)]

# Add attributes to file
# Variable attributes
mask.attrs['_ARRAY_DIMENSIONS'] = ('time', 'longitude', 'latitude', 'pattern')
mask.attrs['description'] = 'classification mask for every single pattern and classification_id'
lons.attrs['_ARRAY_DIMENSIONS'] = ('longitude')
lons.attrs['standard_name'] = 'longitude'
lons.attrs['units'] = 'degree_east'
lats.attrs['_ARRAY_DIMENSIONS'] = ('latitude')
lats.attrs['standard_name'] = 'latitude'
lats.attrs['units'] = 'degree_north'
times.attrs['_ARRAY_DIMENSIONS'] = ('time')
times.attrs['description'] = 'classification id (basically each sighting of an image has a unique id)'
patterns.attrs['_ARRAY_DIMENSIONS'] = ('pattern')

# Global attributes
root_grp.attrs['title'] = 'IR neural network meso-scale cloud pattern classifications'
root_grp.attrs['description'] = 'NN detections of meso-scale cloud patterns'
root_grp.attrs['author'] = 'Hauke Schulz (haschulz@uw.edu)'
root_grp.attrs['institute'] = 'University of Washington'
root_grp.attrs['created_on'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M UTC')
# root_grp.attrs['created_with'] = os.path.basename(__file__) + " with its last modification on " + time.ctime(
#             os.path.getmtime(os.path.realpath(__file__)))
# root_grp.attrs['version'] = git_module_version
root_grp.attrs['python_version'] = "{}".format(sys.version)


def wh2xy(x, y, w, h):
    """
    Converts [x, y, w, h] to [x1, y1, x2, y2], i.e. bottom left and top right coords.
    >>> helpers.wh2xy(10, 20, 30, 1480)
    (10, 20, 39, 1499)
    """
    return x, y, x+w-1, y+h-1

def create_mask(boxes, labels, out, label_map={'Sugar':0, 'Fish': 3, 'Flower': 2, 'Gravel': 1}):
    """
    Create or add mask to array
    """
    xy_boxes = [wh2xy(*b) for b in boxes]
    
    for l, lab in enumerate(labels):
        mask_layer = label_map[lab]
        x1,y1,x2,y2 = np.array(xy_boxes[l]).astype(int)
        out[x1:x2,y1:y2,mask_layer] = True
    
    return out

for i, (index, idx_grp) in enumerate(tqdm(df_all.groupby('time'))):
    o=np.zeros((nb_lons,nb_lats,nb_patterns), dtype=np.bool)
    create_mask(np.array([idx_grp['x'].values,
                           idx_grp['y'].values,
                           idx_grp['w'].values,
                           idx_grp['h'].values
                          ]).T
                ,
                idx_grp['labels'], out=o,
                label_map=label_map)
    mask[i,:,:,:] = o
    times[i] = index

    
    
    
    
    
df = pd.read_pickle(resnet_classifications)  

for label, grp_labels in df.groupby('labels'):
    print('label: %s (%i cases)' % (label, len(grp_labels)))


result_dicts = {}
for l in tqdm.trange(len(df.index)):
    classification = df.iloc[l]
    box = [df.x[l], df.y[l], df.w[l], df.h[l]]
    x0, y0, x1, y1 = hc.get_image_coords(box)
    
#     print('Filename: {}'.format(df.file[l]))
    center = hc.calc_center_latlon(box)
    file_info_df = decode_filepath(df.file[l].replace('TrueColor','').replace('Brightness_Temp_Band31_Day','BrightnessTempBand31Day_'))
    lon0, lat0, lon1, lat1 = convert_pixelCoords2latlonCoords(np.array([[x0, y0, x1, y1]]).astype('int'), regions=[file_info_df.region.iloc[0]])
    overpass_time = hc.retrieve_overpass_time(file_info_df.date.iloc[0], center, path_tle=path_tle)
    if isinstance(overpass_time, dt.datetime):
        overpass_time = date2num(overpass_time, "seconds since 1970-01-01 00:00:00 UTC")
    result_dict = {'region': file_info_df.region.iloc[0],
                   'overpass_time': overpass_time,
                   'overpass_sat_name': file_info_df.satellite.iloc[0],
                   'label_type': df.labels[l],
                   'label_NN_score': df.score[l],
                   'label_imgx': df.x[l],
                   'label_imgy': df.y[l],
                   'label_imgw': df.w[l],
                   'label_imgh': df.h[l],
                   'label_imgx0': x0,
                   'label_imgx1': x1,
                   'label_imgy0': y0,
                   'label_imgy1': y1,
                   'label_lat0': float(lat0),
                   'label_lon0': float(lon0),
                   'label_lat1': float(lat1),
                   'label_lon1': float(lon1),
                   'label_center_lat': center[0],
                   'label_center_lon': center[1],
                   'filename': df.file[l]
                  }
    result_dicts[l] = result_dict

result_df = pd.DataFrame.from_dict(result_dicts, orient='index'); result_df.head()

# Add metadata
result_ds = xr.Dataset.from_dataframe(result_df)

#Global attributes
result_ds.attrs['title'] = 'Meso-scale cloud classifications of a neural network'
result_ds.attrs['author'] = 'Hauke Schulz (hauke.schulz@mpimet.mpg.de)'
result_ds.attrs['source'] = resnet_classifications
result_ds.attrs['created_date'] = dt.datetime.now().strftime('%Y/%m/%d %H:%M UTC')
result_ds.attrs['created_with'] = __file__
result_ds.attrs['python_version'] = 'python: {}; numpy: {}, xarray: {}, pandas: {}'.format(sys.version,
                                                np.__version__, xr.__version__, pd.__version__)

#Variable attributes
result_ds.region.attrs['description'] = 'number of original region that were used during training'
result_ds.region.attrs['valid_range'] = (1, 3)

result_ds.overpass_time.attrs['description'] = 'time the satellite has passed over the center of the classification'
result_ds.overpass_time.attrs['units'] = 'seconds since 1970-01-01 00:00:00 UTC'
result_ds.overpass_time.attrs['calendar'] = 'standard'


result_ds.overpass_sat_name.attrs['description'] = 'name of satellite'

result_ds.label_type.attrs['description'] = 'type of detected mesoscale organization'
result_ds.label_type.attrs['valid_values'] = 'sugar gravel flower fish'

result_ds.label_NN_score.attrs['description'] = 'confidence score of the NN that the label_type is correct'

result_ds.label_imgx.attrs['description'] = 'position of label-box on original image in the format (x,y,w,h)(x)'
result_ds.label_imgy.attrs['description'] = 'position of label-box on original image in the format (x,y,w,h) (y)'
result_ds.label_imgw.attrs['description'] = 'position of label-box on original image in the format (x,y,w,h) (width)'
result_ds.label_imgh.attrs['description'] = 'position of label-box on original image in the format (x,y,w,h) (height)'
result_ds.label_imgx0.attrs['description'] = 'position of label-box on original image in the format (x0,y0,x1,y1) (x0)'
result_ds.label_imgy0.attrs['description'] = 'position of label-box on original image in the format (x0,y0,x1,y1) (y0)'
result_ds.label_imgx1.attrs['description'] = 'position of label-box on original image in the format (x0,y0,x1,y1) (x1)'
result_ds.label_imgy1.attrs['description'] = 'position of label-box on original image in the format (x0,y0,x1,y1) (y1)'
result_ds.label_lat0.attrs['description'] = 'geographic position of label-box (lat0)'
result_ds.label_lon0.attrs['description'] = 'geographic position of label-box (lon0)'
result_ds.label_lat1.attrs['description'] = 'geographic position of label-box (lat1)'
result_ds.label_lon1.attrs['description'] = 'geographic position of label-box (lon1)'
result_ds.label_center_lat.attrs['description'] = 'geographical center of label (latitude)'
result_ds.label_center_lon.attrs['description'] = 'geographical center of label (longitude)'

result_ds.filename.attrs['description'] = 'image filename'

for var in result_ds.data_vars:
    result_ds[var].encoding['zlib'] = True

result_ds.to_netcdf(file_classification_netCDF)