import os
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

config_file = 'projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_drone.py'
checkpoint_file = 'logs/exp1/latest.pth'
image = 'test_samples/000250.jpg'

model_init = init_detector(config=config_file, checkpoint=checkpoint_file)

print("--->>> inference start <<<---")
image_result = inference_detector(model_init, image)
print("--->>> inference complete <<<---")

show_result_pyplot(
    model_init,
    image,
    image_result,
    score_thr=0.4,
    out_file=f'tmp/{os.path.basename(image).split(".")[0]}.jpg')