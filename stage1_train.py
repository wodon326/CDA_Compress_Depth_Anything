from __future__ import print_function, division

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os


from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from CDA.depth_latent1_avg_ver import depth_latent1_avg_ver
from CDA.depth_latent4_avg_ver import depth_latent4_avg_ver

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from core.loss import GradL1Loss, ScaleAndShiftInvariantLoss, GradientMatchingLoss
from metric_calc import sequence_loss

import core.AsymKD_datasets as datasets
import gc

import torch.nn.functional as F
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
    #         pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, 
            div_factor=1, final_div_factor=10000, 
            pct_start=0.7, three_phase=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler, model_type):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.model_type = model_type
        self.writer = SummaryWriter(log_dir=f'runs/{self.model_type}')

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=f'runs/{self.model_type}')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=f'runs/{self.model_type}')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


class State(object):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def capture(self):
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
    
    def apply_snapshot(self, obj):
        self.model.load_state_dict(obj['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(obj['optimizer_state_dict'])
        self.scheduler.load_state_dict(obj['scheduler_state_dict'])
    
    def save(self, path):
        torch.save(self.capture(), path)
    
    def load(self, path, device):
        obj = torch.load(path, map_location=device)
        self.apply_snapshot(obj)

def train(rank, world_size, args):
    try:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()

        if(args.model_type == 'depth_latent1_avg_ver'):
            CDA = depth_latent1_avg_ver().to(rank)
            student_ckpt = '/home/wodon326/datasets/AsymKD_checkpoints/depth_anything_v2_vits.pth'

            teacher_ckpt = '/home/wodon326/datasets/AsymKD_checkpoints/depth_anything_v2_vitl.pth'
            CDA.load_backbone_from_ckpt(student_ckpt, teacher_ckpt, device=torch.device('cuda', rank))
            CDA.freeze_depth_latent1_style()
        elif(args.model_type == 'depth_latent4_avg_ver'):
            CDA = depth_latent4_avg_ver().to(rank)
            student_ckpt = '/home/wodon326/datasets/AsymKD_checkpoints/depth_anything_v2_vits.pth'

            teacher_ckpt = '/home/wodon326/datasets/AsymKD_checkpoints/depth_anything_v2_vitl.pth'
            CDA.load_backbone_from_ckpt(student_ckpt, teacher_ckpt, device=torch.device('cuda', rank))
            CDA.freeze_depth_latent4_style()


        if rank == 0:
            for n, p in CDA.named_parameters():
                print(f'{n} : {p.requires_grad}')

        CDA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(CDA)
        model = DDP(CDA, device_ids=[rank])
        print("Parameter Count: %d" % count_parameters(model))
        train_loader = datasets.fetch_dataloader(args, rank, world_size)
        optimizer, scheduler = fetch_optimizer(args, model)
        total_steps = 0
        if rank == 0:
            logger = Logger(model, scheduler,args.model_type)
        state = State(model, optimizer, scheduler)

        model.train()
        #model.module.freeze_bn() # We keep BatchNorm frozen

        validation_frequency = 10000

        scaler = GradScaler(enabled=args.mixed_precision)

        should_keep_training = True
        global_batch_num = 0
        epoch = 0

        SSILoss = ScaleAndShiftInvariantLoss()
        grad_loss = GradientMatchingLoss()

        save_step = 200

        # load snapshot
        if args.restore_ckpt is not None:
            assert args.restore_ckpt.endswith(".pth")
            state.load(args.restore_ckpt, torch.device('cuda', rank))
        
        while should_keep_training:

            for i_batch, data_blob in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                depth_image, flow, valid = [x.cuda() for x in data_blob]
                assert model.training
                flow_predictions = model(depth_image)
                assert model.training

                try:
                    l_si, scaled_pred = SSILoss(
                        flow_predictions, flow, mask=valid.bool(), interpolate=True, return_interpolated=True)
                    loss = l_si
                    l_grad = grad_loss(scaled_pred, flow, mask=valid.bool().unsqueeze(1))
                    loss = loss + 2 * l_grad
                except Exception as e:
                    loss, _ = sequence_loss(flow_predictions, flow, valid)

                    filename = 'Exception_catch.txt'
                    a = open(filename, 'a')
                    a.write(str(e)+'\n')
                    a.close()



                if(rank==0):
                    logger.writer.add_scalar("live_loss", l_si.item(), global_batch_num)
                    
                    logger.writer.add_scalar("gradient_matching_loss", l_grad.item(), global_batch_num)
                    
                    logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)

                    if(total_steps % 10 == 10-1):
                        # inference visualization in tensorboard while training
                        rgb = depth_image[0].cpu().detach().numpy()
                        rgb = ((rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))) * 255
            
                        gt = flow[0].cpu().detach().numpy()
                        gt = ((gt - np.min(gt)) / (np.max(gt) - np.min(gt))) * 255
            
                        pred = flow_predictions[0].cpu().detach().numpy()
                        pred = ((pred - np.min(pred)) / (np.max(pred) - np.min(pred))) * 255
            
                        logger.writer.add_image('RGB', rgb.astype(np.uint8), global_batch_num)
                        logger.writer.add_image('GT', gt.astype(np.uint8), global_batch_num)
                        logger.writer.add_image('Prediction', pred.astype(np.uint8), global_batch_num)
                    _ , metrics = sequence_loss(flow_predictions, flow, valid)
                    logger.push(metrics)

                


                global_batch_num += 1
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()


                if epoch >= 0 and total_steps % save_step == save_step-1 and rank == 0:
                    save_path = Path('checkpoint_stage1_%s/%d_%s.pth' % (args.model_type,total_steps + 1, args.name))
                    logging.info(f"Saving file {save_path.absolute()}")
                    state.save(save_path)


                if total_steps%100==0:
                    torch.cuda.empty_cache()
                    gc.collect()

                total_steps += 1



                if total_steps > args.num_steps:
                    should_keep_training = False
                    break
            epoch += 1     
            if len(train_loader) >= 10000:
                save_path = Path('checkpoint_stage1_%s/%d_%s.pth' % (args.model_type,total_steps + 1, args.name))
                logging.info(f"Saving file {save_path}")
                state.save(save_path)
                

        print("FINISHED TRAINING")
        logger.close()
        PATH = 'checkpoint_stage1_%s/%s.pth' % (args.model_type,args.name)
        state.save(PATH)

        return PATH
    finally:
        cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='AsymKD_new_loss', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--epoch', type=int, default=3, help="length of training schedule.")
    parser.add_argument('--model_type', type=str, help="model_type")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['tartan_air'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.00005, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[518, 518], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['hf','h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    # print('args.epoch : ', args.epoch)
    Path(f"checkpoint_stage1_{args.model_type}").mkdir(exist_ok=True, parents=True)
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,args,), nprocs=world_size, join=True)
