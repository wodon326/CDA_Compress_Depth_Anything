set -e
set -x

CUDA_VISIBLE_DEVICES=2 python AsymKD_evaluate_affine_inv_gpu.py \
    --model depth_anything_v2_small \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_diode_all.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/diode