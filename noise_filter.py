import os
import os.path as osp

from tqdm import tqdm
import cv2
import json
import numpy as np

import scipy
import matplotlib.pyplot as plt


color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] # red, green, blue


def load_frames(video_dir):
    frames = sorted([osp.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    frames = [cv2.imread(f) for f in frames]
    h, w = frames[0].shape[:2]
    
    return frames, h, w


def float_to_int(bbox):
    return np.round(bbox).astype(np.int32)


def median_filter_fit(x_data, window_size=5):
    x_data_fitted = scipy.signal.medfilt(x_data, window_size)
    return x_data_fitted


def ema_correction(res_data, abnormal_indices, interval=1, alpha=0.2):
    corrected_data = res_data.copy()
    
    for i in abnormal_indices:
        prev_idx = i - 1
        
        next_idx = i + 1
        while next_idx in abnormal_indices and next_idx < len(res_data)-1:
            next_idx += 1
        
        prev_idx_start = max(prev_idx-interval, 0)
        next_idx_end = min(next_idx+interval, len(res_data)-1)
        
        if next_idx_end == len(res_data)-1:
            corrected_data[i] = np.mean(corrected_data[prev_idx_start:prev_idx], axis=0)
        else:
            corrected_data[i] = alpha * np.mean(corrected_data[prev_idx_start:prev_idx], axis=0) \
                + (1-alpha) * np.mean(corrected_data[next_idx:next_idx_end], axis=0)

    return corrected_data


video_folder_path = 'E:/Downloads/AntiUAV/track1_test'
measurement_folder_path = './data/codetr'
samples = os.listdir(video_folder_path)
samples.sort()
samples = ['20190925_133630_1_6']

tolerance = 3

for sample in samples:
    measurement_file = osp.join(measurement_folder_path, sample + '.txt')
    video_dir = osp.join(video_folder_path, sample)
    frames, img_h, img_w = load_frames(video_dir)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"tmp/{sample}.mp4", fourcc, 30, (img_w, img_h))
    
    json_data = json.load(open(measurement_file, 'r'))
    res_data = json_data['res']
    res_data = np.array(res_data) # [x, y, w, h]
    
    # split results into x and y
    x_data, y_data = res_data[:, 0], res_data[:, 1]
    
    x_data_filtered = median_filter_fit(x_data)
    y_data_filtered = median_filter_fit(y_data)
    
    plt.plot(x_data, label='x_data')
    plt.plot(x_data_filtered, label='x_data_filtered')
    plt.legend()
    plt.show()
    
    delta_x = np.abs(x_data - x_data_filtered)
    delta_y = np.abs(y_data - y_data_filtered)
    
    delta_x_mean = np.mean(delta_x)
    delta_y_mean = np.mean(delta_y)
    
    abnormal_indices_x = np.where(delta_x > delta_x_mean * tolerance)[0]
    abnormal_indices_y = np.where(delta_y > delta_y_mean * tolerance)[0]
    
    # abnormal_indices = np.union1d(abnormal_indices_x, abnormal_indices_y)
    abnormal_indices = np.intersect1d(abnormal_indices_x, abnormal_indices_y)
    
    corrected_res_data = ema_correction(res_data, abnormal_indices)
    
    for i, frame in tqdm(enumerate(frames), total=len(frames)):
        img = frame.copy()
        cv2.putText(img, f'{i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        x, y, w, h = float_to_int(res_data[i])
        
        if i in abnormal_indices:
            x_c, y_c, w_c, h_c = float_to_int(corrected_res_data[i])
            cv2.rectangle(img, (x, y), (x + w, y + h), color[0], 2)
            cv2.rectangle(img, (x_c, y_c), (x_c + w_c, y_c + h_c), color[2], 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), color[1], 2)
            
        out.write(img)
        
    out.release()
    
    pass
