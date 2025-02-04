# Last modified: 2024-03-11
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import argparse
import logging
import os
import sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CODE_SPACE)
print(CODE_SPACE)

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
)
from dataset.util import metric
from dataset.util.alignment_gpu import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)
from dataset.util.metric import MetricTracker

import torch.nn.functional as F
# from depth_anything_for_evaluate.dpt import DepthAnything
from segment_anything import sam_model_registry, SamPredictor
# from AsymKD_student import AsymKD_Student_Infer
from BriGeS.dpt import Compress_DepthAnythingV2, Compress_DepthAnythingV2_Infer
# from AsymKD_channel_compress import AsymKD_compress
from torch.multiprocessing import Manager
import torch.distributed as dist
import torch.multiprocessing as mp
# try:
#     from mmcv.utils import Config, DictAction
# except:
#     from mmengine import Config, DictAction
    
# from mono.model.monodepth_model import get_configured_monodepth_model
# from mono.model.criterion import build_criterions
import torchvision.transforms as T
from PIL import Image

def save_tensor_as_png(tensor, file_path):
    """
    PyTorch 텐서를 PNG 이미지로 저장하는 함수.
    
    Args:
        tensor (torch.Tensor): 저장할 텐서. (C, H, W) 또는 (H, W) 형태여야 함.
        file_path (str): 저장할 파일 경로. (예: "output.png")
    """
    # 텐서가 (C, H, W)인지 확인. (H, W)면 채널 추가.
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
    
    # Normalize: 텐서가 [0, 1] 범위가 아닐 경우 정규화.
    if tensor.max() > 1 or tensor.min() < 0:
        tensor = tensor - tensor.min()
        tensor = tensor / tensor.max()

    # To PIL Image (채널 순서 변경 필요: PyTorch는 (C, H, W), PIL은 (H, W, C))
    transform = T.ToPILImage()
    image = transform(tensor.cpu())

    # Save as PNG
    image.save(file_path)
    print(f"Image saved to {file_path}")

@torch.no_grad()
def infer(model, image, seg_image=None, model_type=None, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    if(model_type == None):
        if seg_image is None:
            pred1 = model(image, **kwargs)
            pred2 = model(torch.flip(image, [3]), **kwargs)
        else:
            pred1 = model(image, seg_image, **kwargs)
            pred2 = model(torch.flip(image, [3]), torch.flip(seg_image, [3]), **kwargs)
    elif(model_type == 'metric3d'):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            output  = model.inference({'input': image})
            pred1 = output['prediction']
            output  = model.inference({'input': torch.flip(image, [3])})
            pred2 = output['prediction']

    pred1 = get_depth_from_prediction(pred1)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred

eval_metrics = [
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "i_rmse",
    "silog_rmse",
]


# def parse_args_metric3d():
#     parser = argparse.ArgumentParser(description='Train a segmentor')
#     # parser.add_argument('config', help='train config file path')
#     parser.add_argument('--work-dir', help='the dir to save logs and models')
#     parser.add_argument('--tensorboard-dir', help='the dir to save tensorboard logs')
#     parser.add_argument(
#         '--load-from', help='the checkpoint file to load weights from')
#     parser.add_argument(
#         '--resume-from', help='the checkpoint file to resume from')
#     parser.add_argument(
#         '--no-validate',
#         action='store_true',
#         help='whether not to evaluate the checkpoint during training')
#     parser.add_argument(
#         '--gpu-ids',
#         type=int,
#         nargs='+',
#         help='ids of gpus to use '
#         '(only applicable to non-distributed training)')
#     parser.add_argument('--seed', type=int, default=88, help='random seed')
#     parser.add_argument(
#         '--deterministic',
#         action='store_true',
#         help='whether to set deterministic options for CUDNN backend.')
#     parser.add_argument(
#         '--use-tensorboard',
#         action='store_true',
#         help='whether to set deterministic options for CUDNN backend.')
#     parser.add_argument(
#         '--options', nargs='+', action=DictAction, help='custom options')
#     parser.add_argument('--node_rank', type=int, default=0)
#     parser.add_argument('--nnodes', 
#                         type=int, 
#                         default=1, 
#                         help='number of nodes')
#     parser.add_argument(
#         '--launcher', choices=['None', 'pytorch', 'slurm', 'mpi', 'ror'], default='slurm',
#         help='job launcher') 
#     parser.add_argument('--local_rank', 
#                         type=int, 
#                         default=0, 
#                         help='rank')  
#     parser.add_argument('--experiment_name', default='debug', help='the experiment name for mlflow')
#     args = parser.parse_args()
#     return args

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

import queue


def eval(rank, world_size, queue, args):
    try:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()

        output_dir = args.output_dir

        model_type = args.model

        dataset_config = args.dataset_config
        base_data_dir = args.base_data_dir

        alignment = args.alignment
        alignment_max_res = args.alignment_max_res

        no_cuda = args.no_cuda
        pred_suffix = ".npy"

        os.makedirs(output_dir, exist_ok=True)
        cfg_data = OmegaConf.load(dataset_config)

        depth_any_a1 = None

        if(cfg_data.name == 'kitti'):
            depth_any_a1 = 0.944
            depth_any_abs = 0.080
        elif(cfg_data.name == 'nyu_v2'):
            depth_any_a1 = 0.979
            depth_any_abs = 0.043
        elif(cfg_data.name == 'eth3d'):
            depth_any_a1 = 0.983
            depth_any_abs = 0.053
        elif(cfg_data.name == 'scannet'):
            depth_any_a1 = 0.978
            depth_any_abs = 0.042
        elif(cfg_data.name == 'diode'):
            depth_any_a1 = 0.754
            depth_any_abs = 0.265

        cfg_data.filenames =  CODE_SPACE + '/evaluation/' + cfg_data.filenames
        dataset: BaseDepthDataset = get_dataset(
            cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.EVAL
        )

        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
        # -------------------- Model --------------------
        
        while not queue.empty():
            restore_ckpt = str(queue.get())
            print(restore_ckpt)
            torch.cuda.empty_cache()
            if "depth_anything" in model_type:
                if "small" in model_type:
                    encoder = "vits"
                elif "base" in model_type:
                    encoder = "vitb"
                elif "large" in model_type:
                    encoder = "vitl"
                model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(rank)
            elif model_type == "bfm":
                model = Compress_DepthAnythingV2_Infer().to(rank)

                if restore_ckpt is not None:
                    assert restore_ckpt.endswith(".pth")
                    checkpoint = torch.load(restore_ckpt, map_location='cpu')
                    model__state_dict = model.state_dict()
                    new_state_dict = {}
                    for k, v in checkpoint.items():
                        # 키 매핑 규칙을 정의
                        new_key = k.replace('module.','')
                        if new_key in model__state_dict:
                            new_state_dict[new_key] = v

                    model__state_dict.update(new_state_dict)
                    model.load_state_dict(model__state_dict)
                model = model.to(rank).eval()
            elif model_type == "KD_bfm":
                model = AsymKD_Student_Infer().to(rank)
                '''AsymKD_Student pretrain model loading'''
                if restore_ckpt is not None:
                    checkpoint = torch.load(restore_ckpt, map_location=torch.device('cuda', rank))
                    model__state_dict = model.state_dict()
                    new_state_dict = {}
                    for k, v in checkpoint.items():
                        new_key = k.replace('module.', '')
                        if new_key in model__state_dict:
                            new_state_dict[new_key] = v

                    model__state_dict.update(new_state_dict)
                    model.load_state_dict(model__state_dict)
            elif model_type == "bfm_compress":
                model = Compress_DepthAnythingV2().to(rank)

                if restore_ckpt is not None:
                    assert restore_ckpt.endswith(".pth")
                    checkpoint = torch.load(restore_ckpt, map_location='cpu')
                    model__state_dict = model.state_dict()
                    new_state_dict = {}
                    for k, v in checkpoint.items():
                        # 키 매핑 규칙을 정의
                        new_key = k.replace('module.','')
                        if new_key in model__state_dict:
                            new_state_dict[new_key] = v

                    model__state_dict.update(new_state_dict)
                    model.load_state_dict(model__state_dict)
                model = model.to(rank).eval()
            elif model_type == "briges_metric3d":
                # args_metric3d = parse_args_metric3d()
                # args_metric3d.config = args.config
                cfg = Config.fromfile(args.config)
                criterions = build_criterions(cfg)
                # build model
                model = get_configured_monodepth_model(cfg,
                                           criterions,
                                           ).to(rank)
                if restore_ckpt is not None:
                    logging.info("Loading checkpoint...")
                    checkpoint = torch.load(cfg.restore_ckpt, map_location=torch.device('cuda', rank))
                    model_state_dict = model.state_dict()
                    new_state_dict = {}
                    for k, v in checkpoint['model_state_dict'].items():
                        new_key = k
                        if new_key in model_state_dict:
                            new_state_dict[new_key] = v

                    model_state_dict.update(new_state_dict)
                    model.load_state_dict(model_state_dict)
                print(new_state_dict.keys())
            # -------------------- Eval metrics --------------------
            metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

            metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
            metric_tracker.reset()

            # -------------------- Per-sample metric file head --------------------
            # per_sample_filename = os.path.join(output_dir, "per_sample_metrics.csv")
            # # write title
            # with open(per_sample_filename, "w+") as f:
            #     f.write("filename,")
            #     f.write(",".join([m.__name__ for m in metric_funcs]))
            #     f.write("\n")

            # -------------------- Evaluate --------------------
            model.eval()
            for data in dataloader:
                # GT data
                depth_raw_ts = data["depth_raw_linear"].squeeze().to(rank)
                valid_mask_ts = data["valid_mask_raw"].squeeze().to(rank)
                rgb_float = data["rgb_float"].to(rank)
                rgb = data["rgb_norm"].to(rank)

                depth_raw = depth_raw_ts
                valid_mask = valid_mask_ts

                depth_raw_ts = depth_raw_ts.to(rank)
                valid_mask_ts = valid_mask_ts.to(rank)

                # Get prediction
                # (352, 1216)
                if "metric3d" in model_type:
                    if "kitti" in dataset_config:
                        pred_size = (616, 2128)
                    # (480, 640), (768, 1024), (480, 640)
                    elif "nyu" in dataset_config or "diode" in dataset_config or "scannet" in dataset_config:
                        pred_size = (616, 812)
                    # (4032, 6048)
                    elif "eth3d" in dataset_config:
                        pred_size = (616, 924)
                else:
                    if "kitti" in dataset_config:
                        pred_size = (518, 1792)
                    # (480, 640), (768, 1024), (480, 640)
                    elif "nyu" in dataset_config or "diode" in dataset_config or "scannet" in dataset_config:
                        pred_size = (518, 686)
                    # (4032, 6048)
                    elif "eth3d" in dataset_config:
                        pred_size = (518, 770)

                rgb_resized = F.interpolate(rgb, size=pred_size, mode='bilinear', align_corners=False)
                rgb_float_resized = F.interpolate(rgb_float, size=pred_size, mode='bilinear', align_corners=False)
                if "depth_anything" in model_type:
                    pred = infer(model, rgb_resized)
                elif model_type == "bfm":
                    pred = infer(model, rgb_resized)
                elif model_type == "KD_bfm":
                    pred = infer(model, rgb_resized)
                elif model_type == "bfm_compress":
                    rgb_resized_seg = segment_anything_predictor.set_image(rgb_float_resized.squeeze())
                    pred = infer(model, rgb_resized, rgb_resized_seg)
                elif "metric3d" in model_type:
                    rgb_float = rgb_float.squeeze().permute(1,2,0)
                    # print(f'rgb_float {rgb_float.shape}')
                    target_h, target_w, _ = rgb_float.shape
                    resize_h, resize_w = 616, 1064
                    resize_ratio_h = resize_h / target_h
                    resize_ratio_w = resize_w / target_w
                    resize_ratio = min(resize_ratio_h, resize_ratio_w)
                    reshape_h = int(resize_ratio * target_h)
                    reshape_w = int(resize_ratio * target_w)
                    pad_h = max(resize_h - reshape_h, 0)
                    pad_w = max(resize_w - reshape_w, 0)
                    pad_h_half = int(pad_h / 2)
                    pad_w_half = int(pad_w / 2)
                    
                    pad_info = [pad_h, pad_w, pad_h_half, pad_w_half]
                    h, w, _ = rgb_float.shape
                    reshape_h = int(resize_ratio * h)
                    reshape_w = int(resize_ratio * w)

                    pad_h, pad_w, pad_h_half, pad_w_half = pad_info

                    image = rgb_float.permute(2,0,1).unsqueeze(0)
                    # save_tensor_as_png(image.squeeze(), "input_image.png")
                    # Resize
                    image = F.interpolate(
                        image, 
                        size=(reshape_h, reshape_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)

                    # print(f'image interpolate {image.shape}')

                    # Padding
                    pad_h_half = pad_h // 2
                    pad_w_half = pad_w // 2

                    image = F.pad(
                        image, 
                        (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half), 
                        mode='constant', 
                        value=0
                    )
                    # save_tensor_as_png(image.squeeze(), "input_image_pad.png")
                    

                    # print(f'image pad {image.shape}')
                    mean=[123.675, 116.28, 103.53]
                    std=[58.395, 57.12, 57.375]
                    mean = torch.tensor(mean).float()[:, None, None].to(rank)
                    std = torch.tensor(std).float()[:, None, None].to(rank)
                    image = torch.div((image - mean), std).to(rank)

                    pred = infer(model, image.unsqueeze(0), model_type="metric3d")

                    # save_tensor_as_png(pred.squeeze(), "pred_pad.png")
                    top_pad = pad_h_half
                    bottom_pad = pad_h - pad_h_half
                    left_pad = pad_w_half
                    right_pad = pad_w - pad_w_half

                    # 패딩 제거
                    pred = pred.squeeze()
                    pred = pred[
                        top_pad: pred.shape[0] - bottom_pad,  # 높이에서 상단과 하단 패딩 제거
                        left_pad: pred.shape[1] - right_pad  # 너비에서 좌측과 우측 패딩 제거
                    ]
                    pred = pred.unsqueeze(0).unsqueeze(0)
                    # save_tensor_as_png(pred.squeeze(), "pred_pad_remove.png")
                    # print(f'pred shape{pred.shape}')
                    # quit()

                depth_pred_ts = F.interpolate(pred, size=depth_raw_ts.shape, mode='bilinear', align_corners=False)
                depth_pred = depth_pred_ts.squeeze()
                # Align with GT using least square
                if "least_square" == alignment:
                    depth_pred, scale, shift = align_depth_least_square(
                        gt_arr=depth_raw,
                        pred_arr=depth_pred,
                        valid_mask_arr=valid_mask,
                        return_scale_shift=True,
                        max_resolution=alignment_max_res,
                    )
                elif "least_square_disparity" == alignment:
                    # convert GT depth -> GT disparity
                    gt_disparity, gt_non_neg_mask = depth2disparity(
                        depth=depth_raw, return_mask=True
                    )
                    # LS alignment in disparity space
                    pred_non_neg_mask = depth_pred > 0
                    valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

                    disparity_pred, scale, shift = align_depth_least_square(
                        gt_arr=gt_disparity,
                        pred_arr=depth_pred,
                        valid_mask_arr=valid_nonnegative_mask,
                        return_scale_shift=True,
                        max_resolution=alignment_max_res,
                    )
                    # convert to depth
                    disparity_pred = torch.clamp(
                        disparity_pred, min=1e-3
                    )  # avoid 0 disparity
                    depth_pred = disparity2depth(disparity_pred)  # 이 함수는 이미 GPU에서 동작하도록 수정됨

                # Clip to dataset min max
                depth_pred = torch.clamp(
                    depth_pred, min=dataset.min_depth, max=dataset.max_depth
                )

                # clip to d > 0 for evaluation
                depth_pred = torch.clamp(depth_pred, min=1e-6)

                # Evaluate (using CUDA if available)
                sample_metric = []
                depth_pred_ts = depth_pred

                for met_func in metric_funcs:
                    _metric_name = met_func.__name__
                    _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
                    sample_metric.append(_metric.__str__())
                    metric_tracker.update(_metric_name, _metric)
                # print(tabulate(
                #     [metric_tracker.result().keys(), metric_tracker.result().values()]
                #     ))
            # -------------------- Save metrics to file --------------------
            

            print_str = f'#######AsymKD {restore_ckpt} evaluate result#############\n'

            for key in metric_tracker.result().keys():
                if(depth_any_a1-metric_tracker.result()['delta1_acc']<0 and depth_any_abs - metric_tracker.result()['abs_relative_difference']>0):
                    print_str += f'@@@@{key} : {round(metric_tracker.result()[key], 3)}\n'
                else:
                    print_str += f'{key} : {round(metric_tracker.result()[key], 3)}\n'
            
            print(print_str)


            metrics_filename = f"eval_metrics-{model_type}-ddp-compress_0101.txt"
            _save_to = os.path.join(output_dir, metrics_filename)
            with open(_save_to, "a") as f:
                f.write(f'{print_str}\n')

    finally:
        cleanup()


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    # input
    # parser.add_argument(
    #     "--config", type=str, required=True, help="config directory.",
    # )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # model
    parser.add_argument(
        "--model", type=str, required=True, help="Model to evaluate."
    )

    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Path to base data directory.",
    )

    # LS depth alignment
    parser.add_argument(
        "--alignment",
        choices=[None, "least_square", "least_square_disparity"],
        default=None,
        help="Method to estimate scale and shift between predictions and ground truth.",
    )
    parser.add_argument(
        "--alignment_max_res",
        type=int,
        default=None,
        help="Max operating resolution used for LS alignment",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to base checkpoint directory.",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda")

    args = parser.parse_args()




    # -------------------- Data --------------------


    world_size = torch.cuda.device_count()
    manager = Manager()
    queue = manager.Queue()    
    # start_num = 1
    # end_num = 325

    # for i in range(end_num,start_num-1,-1):
    #     queue.put(f'{args.checkpoint_dir}/{i}00_AsymKD_new_loss.pth')
    
    arr = ['26100', '22900', '18700', '9700', '8000', '8900', '15900', '13000', '14500', '6700', '14400', '6100', '4300', '2200', '15700', '25300', '19400', '23000', '30500', '28400', '24400', '400']
    for i in arr:
        queue.put(f'{args.checkpoint_dir}/{i}_AsymKD_new_loss.pth')

    # for step in range(end_num,start_num-1,-500):
    #     queue.put(f'{args.checkpoint_dir}/step{step:08d}.pth')
    # arr = ['00225000', '00219000', '00218000', '00221000', '00207000', '00203000', '00201000', '00200000', '00199000', '00180000', '00169000', '00164000', '00157000', '00147000', '00149000', '00134000', '00135000', '00131000', '00130000', '00107000', '00099000', '00068000', '00071000', '00062000', '00065000', '00058000', '00056000', '00052000', '00030000', '00029000', '00027000', '00017000', '00006000']
    # for step in arr:
    #     queue.put(f'{args.checkpoint_dir}/step{step}.pth')

    os.chdir(CODE_SPACE)
    mp.spawn(eval, args=(world_size,queue, args,), nprocs=world_size, join=True)
