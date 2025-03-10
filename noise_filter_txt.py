import os
import os.path as osp

from tqdm import tqdm
import cv2
import json
import numpy as np

import scipy
import matplotlib.pyplot as plt


def median_filter_fit(x_data, window_size=5):
    x_data_fitted = scipy.signal.medfilt(x_data, window_size)
    return x_data_fitted


def ema_correction(res_data, abnormal_indices, interval=1, alpha=0.0):
    corrected_data = res_data.copy()
    
    for i in abnormal_indices:
        prev_idx = i - 1
        
        next_idx = i + 1
        
        while next_idx in abnormal_indices and next_idx < len(res_data)-1:
            next_idx += 1
        
        prev_idx_start = max(prev_idx-interval, 0)
        next_idx_end = min(next_idx+interval, len(res_data)-1)
        
        if prev_idx_start == prev_idx or next_idx_end == next_idx:
            continue
        
        if next_idx_end == len(res_data)-1:
            corrected_data[i] = np.mean(corrected_data[prev_idx_start:prev_idx], axis=0)
        else:
            corrected_data[i] = alpha * np.mean(corrected_data[prev_idx_start:prev_idx], axis=0) \
                + (1-alpha) * np.mean(corrected_data[next_idx:next_idx_end], axis=0)

    return corrected_data


def find_abnormal_indices(data, tolerance=10):
    data_filtered = median_filter_fit(data)
    delta = np.abs(data - data_filtered)
    delta_mean = np.mean(delta)
    abnormal_indices = np.where(delta > delta_mean * tolerance)[0]
    
    return abnormal_indices


def write_txt(data, output_file):
    bbox_list = []
    for bbox in data:
        bbox = [float(x) for x in bbox]
        bbox_list.append(bbox)
    
    with open(output_file, 'w') as f:
        json.dump({'res': bbox_list}, f)


measurement_folder_path = './data/swin_exp3_latest'
samples = os.listdir(measurement_folder_path)
samples.sort()
# samples = ['20190925_133630_1_6']

output_folder_path = f'{measurement_folder_path}_filtered'
os.makedirs(output_folder_path, exist_ok=True)


for sample in tqdm(samples):
    measurement_file = osp.join(measurement_folder_path, sample)
    json_data = json.load(open(measurement_file, 'r'))
    res_data = json_data['res']
    res_data = np.array(res_data) # [x, y, w, h]
    
    # split results into x and y
    x_data, y_data = res_data[:, 0], res_data[:, 1]
    abnormal_indices_x = find_abnormal_indices(x_data)
    abnormal_indices_y = find_abnormal_indices(y_data)
    # abnormal_indices = np.union1d(abnormal_indices_x, abnormal_indices_y)
    abnormal_indices = np.intersect1d(abnormal_indices_x, abnormal_indices_y)
    
    if len(abnormal_indices) > 0:
        corrected_res_data = ema_correction(res_data, abnormal_indices)
    else:
        corrected_res_data = res_data
    
    output_file = osp.join(output_folder_path, sample)
    write_txt(corrected_res_data, output_file)
