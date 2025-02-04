set -e
set -x

CUDA_VISIBLE_DEVICES=6 python AsymKD_evaluate_affine_inv_gpu.py \
    --model depth_anything_v2_small \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_scannet_val.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/scannet