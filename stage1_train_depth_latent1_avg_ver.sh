CUDA_VISIBLE_DEVICES=0,1,2,3 python stage1_train.py \
    --batch_size 16 \
    --num_steps 180000 \
    --lr 0.00005 \
    --train_datasets  VKITTI Hypersim tartan_air \
    --model_type depth_latent1_avg_ver \
    --restore_ckpt /home/wodon326/project/CDA_Compress_Depth_Anything/checkpoint_stage1_depth_latent1_avg_ver/14600_AsymKD_new_loss.pth