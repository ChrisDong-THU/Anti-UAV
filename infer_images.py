import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import json
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

color = [(0, 255, 0)]


def not_exist(pred):
    return (len(pred) == 1 and pred[0] == 0) or len(pred) == 0


def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                        (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T


def laod_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    prompts = {}
    points = {}
    
    x, y, w, h = data['gt_rect'][0]
    x_center, y_center = int(x + w / 2), int(y + h / 2)
    
    # obj_id = 0, (bbox, label)
    prompts[0] = ((x, y, x + w, y + h), 1)  # for labels, `1` means positive click and `0` means negative click
    points[0] = [[x_center, y_center]]
    
    return prompts, points


def result_to_bbox(result, score_thr=0.3):
    bboxes = np.vstack(result)
    scores = bboxes[:, -1]
    inds = scores > score_thr
    
    # no bbox found
    if len(inds) == 0:
        return [0, 0, 0, 0]
    
    # get the bbox with the highest score
    ind = scores.argmax()
    bbox = bboxes[ind, :4]
    bbox = bbox.tolist()
    
    return bbox


def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")


def main(args):
    config_file = 'projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_drone.py'
    checkpoint_file = 'logs/exp1/latest.pth'
    detector = init_detector(config=config_file, checkpoint=checkpoint_file)

    out_folder = f"tmp/track1_test/{args.output_suffix}"
    video_folder = f"{out_folder}/videos"
    measurement_folder = f"{out_folder}/measurements"
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(measurement_folder, exist_ok=True)
    
    test_video_paths = os.listdir(args.test_path)
    test_video_paths.sort()
    
    test_video_paths = ['20190925_133630_1_6']
    
    for test_video_path in test_video_paths:
        # video path
        frames_or_path = prepare_frames_or_path(osp.join(args.test_path, test_video_path))
        visible_path = frames_or_path
        
        # save results as video
        if args.save_to_video:
            if osp.isdir(visible_path):
                frames = sorted([osp.join(visible_path, f) for f in os.listdir(visible_path) if f.endswith((".jpg", ".jpeg", ".png"))]) # compatible with more formats
                height, width = cv2.imread(frames[0]).shape[:2]
                
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f"{video_folder}/{test_video_path}.mp4", fourcc, 30, (width, height))

        # inference
        bbox_results = []
        for frame in tqdm(frames, desc=f"Processing {test_video_path}"):
            result = inference_detector(detector, frame)
            bbox = result_to_bbox(result)

            if args.save_to_video:
                img = cv2.imread(frame)
                bbox_draw = [int(x) for x in bbox]

                cv2.rectangle(img, (bbox_draw[0], bbox_draw[1]), (bbox_draw[2], bbox_draw[3]), color[0], 2)
                
                out.write(img)
                
            bbox_results.append(bbox)

        if args.save_to_video:
            out.release()
        
        if args.save_to_txt:
            with open(f"{measurement_folder}/{test_video_path}.txt", "w") as f:
                json.dump({'res': bbox_results}, f)
    
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="/mnt/disk_2/yuji/UAV_data/track1_test", help="path to test video path")
    parser.add_argument("--output_suffix", default="exp1", help="Suffix to output folder")
    
    parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    parser.add_argument("--save_to_txt", default=True, help="Save results to a txt.")
    
    args = parser.parse_args()
    main(args)
