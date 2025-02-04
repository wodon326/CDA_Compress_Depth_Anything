import os
import sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CODE_SPACE)
os.chdir(CODE_SPACE)
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os.path as osp
import json
from depth_anything_v2.dpt import DepthAnythingV2
import argparse
import torch.nn.functional as F
from tqdm.auto import tqdm
from segment_anything import sam_model_registry, SamPredictor
# from BriGeS.dpt import BriGeS_DepthAnythingV2, BriGeS_DepthAnythingV2_Infer
import torch.multiprocessing as mp
# JSON 로더
def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def make_divisible_by_14(img):
    fixed_height=518
    w, h = img.size
    # height를 fixed_height(518)로 고정
    # 원본 비율 유지: new_w = (w * fixed_height) / h
    new_w = int(round((w * fixed_height) / h))
    
    # 너비 14로 나누어떨어지게
    new_w = ((new_w + 13) // 14) * 14
    
    # 새 사이즈로 리사이즈 (높이는 이미 14로 나누어떨어짐)
    img = img.resize((new_w, fixed_height), Image.BILINEAR)
    return img

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, segment_anything_predictor = None):
        self.root_dir = root_dir
        self.segment_anything_predictor = segment_anything_predictor

        self.data = load_json(osp.join(self.root_dir, 'annotations.json'))
        self.keys = list(self.data.keys())  # 이미지 경로 리스트
        # transform 정의
        transform = transforms.Compose([
            transforms.Lambda(make_divisible_by_14),  # 커스텀 transform 적용
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        ])
        self.transform_ToTensor = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_path = self.keys[idx]
        # 이미지 로드
        image = Image.open(osp.join(self.root_dir, img_path)).convert("RGB")
        seg_image = None
        if(self.segment_anything_predictor is not None):
            seg_image = self.transform_ToTensor(image)
            seg_image = self.segment_anything_predictor.set_image(seg_image)
            seg_image = seg_image.squeeze(0)

        img_shape = image.size
        # 해당 이미지의 label 정보
        # [{"point1": [x1, y1], "point2": [x2, y2], "closer_point": "point1"}, ...] 형태
        label_info = self.data[img_path]
        
        # 필요에 따라 레이블 가공 (예: point 좌표 텐서로 변환)
        # 여기서는 point1, point2 좌표와 closer_point 인덱스(0 또는 1)를 하나의 텐서로 만든다고 가정
        # closer_point가 point1이면 0, point2면 1이라고 하자.
        
        # 여러 개의 포인트가 있을 수 있으므로, 각 포인트에 대해 처리
        # 텐서 형태: [[x1,y1,x2,y2,closer_idx], [x1,y1,x2,y2,closer_idx], ...]
        labels = {}
        for item in label_info:
            labels["point1"] = item["point1"]
            labels["point2"] = item["point2"]
            labels["closer_point"] = 1 if item["closer_point"] == "point1" else 2
        
        
        if self.transform:
            image = self.transform(image)

        if seg_image is None:
            seg_image = image
        return image, seg_image, labels, img_shape


if "__main__" == __name__:
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument(
        "--model", type=str, required=True, help="Model to evaluate."
    )
    parser.add_argument(
        "--root_dir", type=str, required=True, help="root_dir"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = args.model
    segment_anything_predictor = None
    if "depth_anything" in model_type:
        if("v2" in model_type):
            if "small" in model_type:
                encoder = "vits"
            elif "base" in model_type:
                encoder = "vitb"
            elif "large" in model_type:
                encoder = "vitl"
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            model = DepthAnythingV2(**model_configs[encoder]).to(device)
            model.load_state_dict(torch.load(f'/home/wodon326/datasets/AsymKD_checkpoints/depth_anything_v2_{encoder}.pth', map_location=device))
            model = model.to(device).eval()
    elif model_type == "metric3d":
        model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True).to(device)
    elif model_type == "bfm":
        segment_anything = sam_model_registry["vit_l"](checkpoint="/home/wodon326/datasets/AsymKD_checkpoints/sam_vit_l_0b3195.pth").to(device)
        segment_anything_predictor = SamPredictor(segment_anything)

        for child in segment_anything.children():
            ImageEncoderViT = child
            break
        model = BriGeS_DepthAnythingV2_Infer(ImageEncoderViT=ImageEncoderViT).to(device)
        restore_ckpt = '/home/wodon326/project/BriGeS-Depth-Anything-V2/best_checkpoint/1212_20800_AsymKD_new_loss.pth'
        if restore_ckpt is not None:
            checkpoint = torch.load(restore_ckpt, map_location=device)
            model_state_dict = model.state_dict()
            new_state_dict = {}
            for k, v in checkpoint.items():
                new_key = k.replace('module.', '')
                if new_key in model_state_dict:
                    new_state_dict[new_key] = v
            model_state_dict.update(new_state_dict)
            model.load_state_dict(model_state_dict)
        model.eval()

    dataset = CustomImageDataset(
        root_dir=args.root_dir,
        segment_anything_predictor = segment_anything_predictor
    )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )



    answer_true = 0
    answer_false = 0
    answer_num = 0
    for images, seg_image, labels, img_shape in tqdm(data_loader):
        W,H = img_shape[0].item(),img_shape[1].item()
        with torch.no_grad():
            if model_type == "bfm":
                pred = model(images.cuda(), seg_image.cuda())
            elif model_type == "metric3d":
                pred, confidence, output_dict  = model.inference({'input': images.cuda()})
            else:
                pred = model(images.cuda())
            
        pred = pred.squeeze().unsqueeze(0).unsqueeze(0)
        pred = F.interpolate(pred, size=(H,W), mode='bilinear', align_corners=False)
        pred = pred.squeeze()
        
        point1 = pred[labels['point1'][0].item(),labels['point1'][1].item()]
        point2 = pred[labels['point2'][0].item(),labels['point2'][1].item()]
        
        if(model_type == 'metric3d'):
            closer_point = 1 if point1.item()<point2.item() else 2
        else:
            closer_point = 1 if point1.item()>point2.item() else 2
        if(closer_point ==labels['closer_point'].item()):
            # print('True')
            answer_true += 1
            answer_num += 1
        else:
            # print('False')
            answer_false += 1
            answer_num += 1
        
        print(f'accuracy : {answer_true/answer_num}')

        # print("Batch Index:", batch_idx)
        # print("Images Shape:", images.shape)  
        # print("Original Images Shape:", img_shape)  
        # print("Labels:", labels)
        # print("point1:", point1)
        # print("point2:", point2)

    print(f'final accuracy : {answer_true/answer_num}')