set -e
set -x

python AsymKD_evaluate_affine_inv_gpu_ddp.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_eth3d.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/eth3d \
    --alignment_max_res 1024 \
    --checkpoint_dir /home/wodon326/project/BriGeS-Depth-Anything-V2/checkpoints_depthanything_v2_setup