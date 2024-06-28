# python extras/load_hdf5.py
import h5py
import pandas as pd
import numpy as np
import logging
import torch
from preprocessing import *
from tqdm import tqdm
import json

def preprocess_keypoints(data):
    # Spoter format
    df_keypoints = pd.read_csv("Mapeo landmarks librerias.csv", skiprows=1)
    df_keypoints = df_keypoints[(df_keypoints['Selected 54']=='x')]
    body_part = (df_keypoints['Section']+'_'+df_keypoints['Key']).values
    body_section = df_keypoints['Section']

    data = np.moveaxis(data, 1, 2)

    data = normalize_pose_hands_function(data, body_section, body_part)

    #data = np.moveaxis(data, 1, 2)

    depth_map = torch.from_numpy(np.copy(data))
    depth_map = depth_map - 0.5

    return depth_map

def filter_data(data):
    mp_pos_values = [
        0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 501, 502, 503, 504, 505, 506, 507, 508, 509,
        510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526,
        527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543,
    ]

    output_list = []

    for frame in data:
        selected_joints = np.array(mp_pos_values) 
        frame = frame[:,selected_joints]
        output_list.append(frame)

    result = np.asarray(output_list)

    return result

def preproccess(data):
    #logging.warning("FILTER_KEYPOINTS")
    data = filter_data(data)
    #logging.warning("PREPROCESS_KEYPOINT")
    data = preprocess_keypoints(data)

    return data

# Specify the path to the .hdf5 file
file_path = "hdf5/DGI305_mediapipe.hdf5"

# Open the .hdf5 file
with h5py.File(file_path, "r") as file:
    # Print the keys of the groups and datasets in the file
    print("Content of the .hdf5 file:")
    keys = list(file.keys())
    labels_dict = {}
    label_counter = 0
    for key in tqdm(keys, position=0):
        for group_key, group_value in file[key].items():
            data = file[key][group_key][()]
            if group_key == 'data':
                clean_data = preproccess(data)
            elif group_key == 'label':
                label = data.decode('utf-8')
                if label not in labels_dict:
                    labels_dict[label] = 0
                else:
                    labels_dict[label] += 1

    # Filter labels that are repeated less than or equal to 10 times
    labels_dict = {label: count for label, count in labels_dict.items() if count >= 10}
    dict_save = {}
    counter = 0
    for key in labels_dict.keys():
        dict_save[key] = counter
        counter += 1
    print('Labels Dictionary:', dict_save)
    
    with open('hdf5/DGI305_labels_dict.json', 'w') as f:
        json.dump(dict_save, f)
