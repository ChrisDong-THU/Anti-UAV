import os
import os.path as osp
import json
import cv2

from tqdm import tqdm

# transform the json file to coco format
coco_format = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "drone", "supercategory": ""}]
}

# global variables
image_id = 0
ann_id = 0
frame_stride = 4

# image folders
base_dir = '/mnt/disk_2/yuji/UAV_data/cvprworkshop/train'
image_dirs = [d for d in os.listdir(base_dir) if osp.isdir(osp.join(base_dir, d))]
image_dirs.sort()

for image_dir in tqdm(image_dirs[200:]):
    image_dir_path = osp.join(base_dir, image_dir)
    
    image_files = sorted([osp.join(image_dir_path, f) for f in os.listdir(image_dir_path) \
        if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # get the image size
    image_file = image_files[0]
    image = cv2.imread(image_file)
    height, width, _ = image.shape
    
    raw_ann_path = osp.join(image_dir_path, 'IR_label.json')
    with open(raw_ann_path, 'r') as f:
        raw_ann = json.load(f)
    
    exist_label = raw_ann['exist'][::frame_stride]
    bbox = raw_ann['gt_rect'][::frame_stride]

    for i, image_file in enumerate(image_files[::frame_stride]): # skip continuous frames
        image_id += 1
        
        image_entry = {
            'id': image_id,
            'file_name': image_file,
            'width': width,
            'height': height
        }
        coco_format['images'].append(image_entry)
        
        if exist_label[i]==1 :
            ann_id += 1
            ann_entry = {
                'id': ann_id,
                'image_id': image_id, # corr. annotated image
                'category_id': 1, # only drone
                'bbox': bbox[i],
                'area': bbox[i][2] * bbox[i][3],
                'segmentation': [[]],
                'iscrowd': 0
            }
            coco_format['annotations'].append(ann_entry)
            
        
with open('drone_data/annotations/drone_val.json', 'w') as f:
    json.dump(coco_format, f)
    
print(f'Total images: {len(coco_format["images"])}')
print(f'Total annotations: {len(coco_format["annotations"])}')
            
        