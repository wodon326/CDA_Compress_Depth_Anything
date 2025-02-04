import os
import sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CODE_SPACE)
# os.chdir(CODE_SPACE)
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
from BriGeS.dpt import Compress_DepthAnythingV2
from BriGeS.dpt_kd_conv_cbam_self import KD_DepthAnythingV2_Infer
from torch.multiprocessing import Manager
import torch.distributed as dist
import torch.multiprocessing as mp
import queue


# JSON 로더
def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


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
    def __init__(self, root_dir):
        self.root_dir = root_dir

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

        return image, labels, img_shape



def eval(rank, world_size, queue, args):
    try:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()

        output_dir = args.output_dir
        while not queue.empty():
            restore_ckpt = str(queue.get())
            print(restore_ckpt)
            torch.cuda.empty_cache()
            model_type = args.model
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
                    model = DepthAnythingV2(**model_configs[encoder]).to(rank)
                    model.load_state_dict(torch.load(f'/home/wodon326/datasets/AsymKD_checkpoints/depth_anything_v2_{encoder}.pth', map_location=rank))
                    model = model.to(rank).eval()
            elif model_type == "bfm":
        
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
                
            elif model_type == "kd_bfm":
        
                model = KD_DepthAnythingV2_Infer().to(rank)
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

            # model.cuda()
            model.eval()

            dataset = CustomImageDataset(
                root_dir=args.root_dir
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
            for images, labels, img_shape in data_loader:
                W,H = img_shape[0].item(),img_shape[1].item()
                with torch.no_grad():
                    pred = model(images.cuda())
                    
                pred = pred.squeeze().unsqueeze(0).unsqueeze(0)
                pred = F.interpolate(pred, size=(H,W), mode='bilinear', align_corners=False)
                pred = pred.squeeze()
                
                point1 = pred[labels['point1'][0].item(),labels['point1'][1].item()]
                point2 = pred[labels['point2'][0].item(),labels['point2'][1].item()]
                
                closer_point = 1 if point1.item()>point2.item() else 2
                if(closer_point ==labels['closer_point'].item()):
                    # print('True')
                    answer_true += 1
                    answer_num += 1
                else:
                    # print('False')
                    answer_false += 1
                    answer_num += 1
                
                # print(f'accuracy : {answer_true/answer_num}')

                # print("Batch Index:", batch_idx)
                # print("Images Shape:", images.shape)  
                # print("Original Images Shape:", img_shape)  
                # print("Labels:", labels)
                # print("point1:", point1)
                # print("point2:", point2)

            # print(f'final accuracy : {answer_true/answer_num}')

    
            print_str = f'#######AsymKD {restore_ckpt} evaluate result#############\n'

            print_str += f'accuracy : {round(answer_true/answer_num, 3)}\n'
            print(print_str)


            metrics_filename = f"eval_metrics-{model_type}-ddp-kd-combine-0106.txt"
            _save_to = os.path.join(output_dir, metrics_filename)
            with open(_save_to, "a") as f:
                f.write(f'{print_str}\n')

    finally:
        cleanup()



        
if "__main__" == __name__:
    # mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument(
        "--model", type=str, required=True, help="Model to evaluate."
    )
    parser.add_argument(
        "--root_dir", type=str, required=True, help="root_dir"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to base checkpoint directory.",
    )

    args = parser.parse_args()

    

    # -------------------- Data --------------------
    world_size = torch.cuda.device_count()
    manager = Manager()
    queue = manager.Queue()    
    start_num = 5
    end_num = 585

    for i in range(end_num,start_num-1,-5):
        queue.put(f'{args.checkpoint_dir}/{i}00_AsymKD.pth')
    
    
    # arr = ['57800', '56400', '55600', '57000', '54800', '54400', '54200', '53000', '52600', '54000', '52400', '51200', '45600', '42600', '43400', '41200', '40200', '40000', '39800', '39400', '39200', '39000', '38200', '38000', '37800', '37600', '37400', '38800', '36000', '35200', '35000', '34600', '35800', '34400', '34200', '33800', '33200', '32200', '32000', '28600', '27800', '27400', '26200', '26000', '25600', '25000', '24600', '24400', '23200', '22400', '22600', '20800', '20600', '20400', '20000', '19800', '19400', '19000', '18800', '18400', '18200', '17600', '17200', '17000', '16800', '16000', '15400', '15200', '14600', '14200', '14000', '14800', '12400', '11600', '10600', '10000', '9800', '9200', '9000', '9400', '7800', '7400', '6600', '6200', '7000', '4600', '4400', '3800', '3600', '3200', '3000', '2800', '4200', '2600', '2400', '2200', '1600', '1400', '1000', '800', '600', '400', '200', '1200']
    # arr = ['23000', '20500', '19500', '19000', '12500', '12000', '15000', '10500', '11000', '7500', '9000', '8500', '6500', '8000', '5500', '5000', '4500', '3500', '3000', '1000']
    # for i in arr:
    #     queue.put(f'{args.checkpoint_dir}/{i}_AsymKD_new_loss.pth')

    # for step in range(end_num,start_num-1,-500):
    #     queue.put(f'{args.checkpoint_dir}/step{step:08d}.pth')
    # arr = ['00225000', '00219000', '00218000', '00221000', '00207000', '00203000', '00201000', '00200000', '00199000', '00180000', '00169000', '00164000', '00157000', '00147000', '00149000', '00134000', '00135000', '00131000', '00130000', '00107000', '00099000', '00068000', '00071000', '00062000', '00065000', '00058000', '00056000', '00052000', '00030000', '00029000', '00027000', '00017000', '00006000']
    # for step in arr:
    #     queue.put(f'{args.checkpoint_dir}/step{step}.pth')

    os.chdir(CODE_SPACE)
    mp.spawn(eval, args=(world_size,queue, args,), nprocs=world_size, join=True)

