CUDA_VISIBLE_DEVICES=4,5,6,7 python stage1_train.py \
    --batch_size 16 \
    --num_steps 180000 \
    --lr 0.00005 \
    --train_datasets  VKITTI Hypersim tartan_air \
    --model_type depth_latent4_avg_ver