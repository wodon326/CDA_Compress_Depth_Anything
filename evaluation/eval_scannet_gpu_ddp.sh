set -e
set -x

python AsymKD_evaluate_affine_inv_gpu_ddp.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_scannet_val.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/scannet \
    --checkpoint_dir /home/wodon326/project/BriGeS-Depth-Anything-V2/checkpoints_depthanything_v2_setup
