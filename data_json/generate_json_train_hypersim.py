import h5py
import numpy as np
import pandas as pd
import os


import numpy as np


# 파일 경로
root_path = "/home/wodon326/datasets/Hypersim"
# Extrinsic 파라미터를 사용하여 Intrinsic 파라미터를 정확히 추정하는 코드


with open('hypersim_train.txt') as f:
    lines_train = f.readlines()
# Extrinsic 기반 Intrinsic 데이터 저장할 리스트

#ai_001_002/rgb_cam_01_fr0072.png
#ai_001_002/images/scene_cam_01_final_preview/frame.0072.png
#ai_001_002/depth_plane_cam_01_fr0065.png
#ai_001_002/images/scene_cam_01_fr0065.png
#ai_023_009/images/scene_cam_00_geometry_hdf5/frame.0000.depth_meters.hdf5
#ai_023_009/images/scene_cam_00_geometry_hdf5/frame.0000.normal_cam.hdf5

train_file_list = []
train_file_dict = {}
for line in lines_train:
    l = line.split(' ')
    rgb = {}
    rgb_tonemap = l[0].replace('rgb_cam','images/scene_cam').replace('fr','final_preview/frame.').replace('.png','.tonemap.jpg')
    depth = l[1].replace('depth_plane','images/scene').replace('fr','geometry_hdf5/frame.').replace('.png','.depth_meters.hdf5').replace('\n','')
    folder_name = l[0].split('/')[0]
    curr_file = [{'rgb':rgb_tonemap, 'depth':depth}]
    print(curr_file)
    train_file_list = train_file_list + curr_file

print(len(train_file_list))
import json

train_file_dict['files'] = train_file_list
with open('train_hypersim.json', 'w') as fj:
    json.dump(train_file_dict, fj)