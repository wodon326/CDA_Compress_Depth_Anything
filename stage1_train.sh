CUDA_VISIBLE_DEVICES=0,1,2,3,4 python stage1_train.py \
    --batch_size 16 \
    --num_steps 180000 \
    --lr 0.00005 \
    --train_datasets  VKITTI Hypersim tartan_air