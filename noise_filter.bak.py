import os
import os.path as osp

from tqdm import tqdm
import cv2
import json
import numpy as np


color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] # red, green, blue


def load_frames(video_dir):
    frames = sorted([osp.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    frames = [cv2.imread(f) for f in frames]
    h, w = frames[0].shape[:2]
    
    return frames, h, w


def find_abnormal_zero_indices(res_data, w, h, dist_threshold=10):
    zero_indices = np.where(np.all(res_data[:, :2] == 0, axis=1))[0]
    
    first_zero_index = []
    for i in range(1, len(zero_indices)-1):
        if zero_indices[i+1] - zero_indices[i] == 1:
            if zero_indices[i] - zero_indices[i-1] > 1:
                first_zero_index.append(zero_indices[i])
            
    last_zero_index = []
    for i in range(1, len(zero_indices)-1):
        if zero_indices[i] - zero_indices[i-1] == 1:
            if zero_indices[i+1] - zero_indices[i] > 1:
                last_zero_index.append(zero_indices[i])
    
    # detect image boundary
    for i in range(len(first_zero_index)):
        prev_start = max(first_zero_index[i]-5, 0)
        prev_x = np.mean(res_data[prev_start:first_zero_index[i], 0])
        prev_y = np.mean(res_data[prev_start:first_zero_index[i], 1])
        
        dist = (prev_x, prev_y, w-prev_x, h-prev_y)
        if np.min(dist) < dist_threshold:
            mask = (zero_indices >= first_zero_index[i]) & (zero_indices <= last_zero_index[i])
            zero_indices = zero_indices[~mask]
    
    return zero_indices


# def find_abnormal_vary_indices(res_data, zero_indices, threshold_factor=0.5):
#     none_zero_data = np.delete(res_data, zero_indices, axis=0)
    
#     # normal box size
#     normal_w = get_mode_mean(none_zero_data[:, 2])
#     normal_h = get_mode_mean(none_zero_data[:, 3])
    
#     abnormal_indices = []
#     for i in range(len(res_data)):
#         if i in zero_indices:
#             continue
        
#         w, h = res_data[i, 2], res_data[i, 3]
#         if np.abs(w - normal_w) > threshold_factor * normal_w \
#             or np.abs(h - normal_h) > threshold_factor * normal_h:
#             abnormal_indices.append(i)
    
#     return abnormal_indices, normal_w, normal_h


def get_mode_mean(data, tolerance=1):
    value_counts = {}
    for val in data:
        count = sum(1 for x in data if abs(x - val) <= tolerance)
        value_counts[val] = count
    
    max_count = max(value_counts.values())
    mode_values = [val for val, count in value_counts.items() if count == max_count]
    
    return np.mean(mode_values)


def find_abnormal_vary_indices(res_data, alpha=0.1, beta=0.5):
    delta_bbox = np.diff(res_data, axis=0)
    delta_bbox = np.abs(delta_bbox)
    
    delta_bbox_norm = np.mean(delta_bbox[:, :2], axis=1) + alpha * np.mean(delta_bbox[:, 2:], axis=1)
    delta_bbox_tol = get_mode_mean(delta_bbox_norm, tolerance=1) * (1+beta)
    
    abnormal_indices = np.where(delta_bbox_norm > delta_bbox_tol)[0]
    abnormal_indices = abnormal_indices + 1
    

    return abnormal_indices


def float_to_int(bbox):
    return np.round(bbox).astype(np.int32)


def ema_correction(res_data, abnormal_indices, alpha=0.8):
    corrected_data = res_data.copy()
    
    for i in abnormal_indices:
        prev_idx = i - 1
        
        next_idx = i + 1
        while next_idx in abnormal_indices and next_idx < len(res_data)-1:
            next_idx += 1
        
        prev_idx_start = max(prev_idx-3, 0)
        next_idx_end = min(next_idx+3, len(res_data)-1)
        
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

for sample in samples:
    measurement_file = osp.join(measurement_folder_path, sample + '.txt')
    video_dir = osp.join(video_folder_path, sample)
    frames, img_h, img_w = load_frames(video_dir)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"tmp/{sample}.mp4", fourcc, 30, (img_w, img_h))
    
    json_data = json.load(open(measurement_file, 'r'))
    res_data = json_data['res']
    res_data = np.array(res_data) # [x, y, w, h]
    
    zero_indices = np.where(np.all(res_data[:, :2] == 0, axis=1))[0]
    abnormal_zero_indices = find_abnormal_zero_indices(res_data, img_w, img_h)
    abnormal_vary_indices = find_abnormal_vary_indices(res_data)
    
    abnormal_indices = np.union1d(abnormal_vary_indices, abnormal_zero_indices)
    corrected_res_data = ema_correction(res_data, abnormal_indices)
    
    for i, frame in tqdm(enumerate(frames), total=len(frames)):
        img = frame.copy()
        cv2.putText(img, f'{i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        x, y, w, h = float_to_int(res_data[i])
        
        if i in abnormal_indices:
            x_c, y_c, w_c, h_c = float_to_int(corrected_res_data[i])
            # cv2.rectangle(img, (x, y), (x + w, y + h), color[0], 2)
            cv2.rectangle(img, (x_c, y_c), (x_c + w_c, y_c + h_c), color[1], 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), color[1], 2)
            
        out.write(img)
        
    out.release()
    
    pass
