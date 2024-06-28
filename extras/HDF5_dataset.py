# python extras/HDF5_dataset.py
import h5py
import pandas as pd
import numpy as np
import logging
import torch
import torch.utils.data as torch_data
from preprocessing import *
from random import randrange
from augmentations import *
from tqdm import tqdm
import json

def preprocess_keypoints(data):
    # Spoter format
    df_keypoints = pd.read_csv("Mapeo landmarks librerias.csv", skiprows=1)
    df_keypoints = df_keypoints[df_keypoints['Selected 54'] == 'x']
    body_part = (df_keypoints['Section'] + '_' + df_keypoints['Key']).values
    body_section = df_keypoints['Section']

    data = np.moveaxis(data, 1, 2)
    data = normalize_pose_hands_function(data, body_section, body_part)
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
        frame = frame[:, selected_joints]
        output_list.append(frame)

    result = np.asarray(output_list)

    return result

def preprocess(data):
    #logging.warning("FILTER_KEYPOINTS")
    data = filter_data(data)
    #logging.warning("PREPROCESS_KEYPOINT")
    data = preprocess_keypoints(data)
    return data

class HDF5Dataset(torch_data.Dataset):
    def __init__(self, file_path, transform=None, augmentations=False, augmentations_prob=0.5, label_dictionary=None):
        self.data = []
        self.labels = []
        with h5py.File(file_path, "r") as file:
            for key in tqdm(list(file.keys()),position=0):
                word = file[key]['label'][()]
                word = word.decode('utf-8')
                if label_dictionary and (word in label_dictionary):
                    label = label_dictionary[word]
                    self.labels.append(label)
                    read_data = file[key]['data'][()]
                    clean_data = preprocess(read_data)
                    self.data.append(clean_data)

        #self.data = np.concatenate(self.data, axis=0)
        #self.labels = np.concatenate(self.labels, axis=0)
        self.transform = transform
        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob

    def __getitem__(self, idx):
        depth_map = self.data[idx]
        label = torch.Tensor([self.labels[idx]])

        if self.augmentations and random.random() < self.augmentations_prob:
            selected_aug = randrange(3)
            if selected_aug == 0:
                depth_map = augment_rotate(depth_map, (-13, 13))
            if selected_aug == 1:
                depth_map = augment_shear(depth_map, "perspective", (0, 0.1))
            if selected_aug == 2:
                depth_map = augment_shear(depth_map, "squeeze", (0, 0.15))

        if self.transform:
            depth_map = self.transform(depth_map)

        return depth_map, label

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    with open("hdf5/DGI305_labels_dict.json", 'r') as file:
        labels_dict = json.load(file)

    dataset = HDF5Dataset("hdf5/DGI305_mediapipe.hdf5",label_dictionary=labels_dict)
    
    print(f"Dataset size: {len(dataset)}")
    image1, label1 = dataset[1]
    print(f"Sample data shape: {image1.shape}")
    print(f"Sample label: {label1}")

    # Save the processed dataset
    data_save_path = "hdf5/DGI305_data.pt"
    labels_save_path = "hdf5/DGI305_labels.pt"
    torch.save(dataset.data, data_save_path)
    torch.save(dataset.labels, labels_save_path)

    print(f"Processed data saved to {data_save_path} and {labels_save_path}")