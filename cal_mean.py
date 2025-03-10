import os
import os.path as osp
from tqdm import tqdm

import cv2
import numpy as np

import random


base_dir = 'E:/Downloads/AntiUAV/track1_test'
image_dirs = [d for d in os.listdir(base_dir) if osp.isdir(osp.join(base_dir, d))]
image_dirs.sort()

# random sample 20 folders
image_dirs = random.sample(image_dirs, 20)

means = []
stds = []
for image_dir in tqdm(image_dirs):
    image_dir_path = osp.join(base_dir, image_dir)
    
    image_files = sorted([osp.join(image_dir_path, f) for f in os.listdir(image_dir_path) \
        if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    for image_file in image_files:
        image = cv2.imread(image_file)
        image = image.astype(np.float32) / 255.0
        
        mean, std = cv2.meanStdDev(image)
        means.append(mean)
        stds.append(std)

mean = np.array(means).mean(axis=0) * 255.0
std = np.array(stds).mean(axis=0) * 255.0

print(mean, std)
