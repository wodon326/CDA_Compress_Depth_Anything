set -e
set -x

CUDA_VISIBLE_DEVICES=1 python AsymKD_evaluate_affine_inv_gpu.py \
    --model depth_anything_v2_small \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_eth3d.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/eth3d \
    --alignment_max_res 1024