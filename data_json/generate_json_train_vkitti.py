import os
from glob import glob

root_folder = '/home/wodon326/datasets/vkitti'
depth_folder = os.path.join(root_folder, "depth")
segment_folder = os.path.join(root_folder, "segment")
rgb_folder = os.path.join(root_folder, "rgb")
scene_arr = ['0001', '0002', '0006', '0018', '0020']
situation = ['15-deg-left','15-deg-right','30-deg-left','30-deg-right','clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']

vkitti_depth = []
vkitti_segment = []
vkitti_rgb = []

train_file_list = []

K = [725, 725, 620.5, 187]
for scene in scene_arr:
    for sit in situation:
        vkitti_1_depth_folder = sorted(glob(os.path.join(depth_folder, "vkitti_1.3.1_depthgt", scene, sit,'*.png')))
        # vkitti_1_segment_folder = sorted(glob(os.path.join(segment_folder, "vkitti_1.3.1_scenegt", scene, sit,'*.png')))
        vkitti_1_rgb_folder = sorted(glob(os.path.join(rgb_folder, "vkitti_1.3.1_rgb", scene, sit,'*.png')))
        for depth_path, rgb_path in zip(vkitti_1_depth_folder, vkitti_1_rgb_folder):
            curr_file = [{'rgb':rgb_path, 'depth':depth_path}]
            train_file_list = train_file_list + curr_file

print(len(train_file_list))
K = [725.0087, 725.0087, 620.5, 187]
scene_arr = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
for scene in scene_arr:
    for sit in situation:
        vkitti_2_depth_folder = sorted(glob(os.path.join(depth_folder, "vkitti_2.0.3_depthgt", scene, sit,'frames','depth','*','*.png')))
        # vkitti_2_segment_folder = sorted(glob(os.path.join(segment_folder, "vkitti_2.0.3_scenegt", scene, sit,'frames','classSegmentation','*','*.png')))
        vkitti_2_rgb_folder = sorted(glob(os.path.join(rgb_folder, "vkitti_2.0.3_rgb", scene, sit,'frames','rgb','*','*.jpg')))
        # print(vkitti_2_depth_folder)
        # print(vkitti_2_segment_folder)
        # print(vkitti_2_rgb_folder)
        for depth_path, rgb_path in zip(vkitti_2_depth_folder, vkitti_2_rgb_folder):
            curr_file = [{'rgb':rgb_path, 'depth':depth_path}]
            train_file_list = train_file_list + curr_file

# # print(train_file_list)
# print(len(train_file_list))
# val_sample_size = 1200
# train_sample_size = len(train_file_list) - val_sample_size

# import random

# random.seed(42)  # For reproducibility
# random.shuffle(train_file_list)
# train = train_file_list[:train_sample_size]
# val = train_file_list[train_sample_size:]

# print(len(val))


print(len(train_file_list))

import json

train_file_dict = {}
train_file_dict['files'] = train_file_list
with open('train_vkitti.json', 'w') as fj:
    json.dump(train_file_dict, fj)


# val_file_dict = {}
# val_file_dict['files'] = val
# with open('val_vkitti.json', 'w') as fj:
#     json.dump(val_file_dict, fj)