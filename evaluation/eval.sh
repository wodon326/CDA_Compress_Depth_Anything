set -e
set -x

# python AsymKD_evaluate_affine_inv_gpu_ddp.py \
#     --model bfm \
#     --base_data_dir ~/data/AsymKD \
#     --dataset_config evaluation/config/data_kitti_eigen_test.yaml \
#     --alignment least_square_disparity \
#     --output_dir evaluation/output/kitti_eigen_test \
#     --checkpoint_dir /home/wodon326/project/Compress-Depth-Anything-V2/checkpoints_new_loss_001_smooth

python AsymKD_evaluate_affine_inv_gpu_ddp.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_nyu_test.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/nyu_test \
    --checkpoint_dir /home/wodon326/project/Compress-Depth-Anything-V2/checkpoints_new_loss_001_smooth

python AsymKD_evaluate_affine_inv_gpu_ddp.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_eth3d.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/eth3d \
    --alignment_max_res 1024 \
    --checkpoint_dir /home/wodon326/project/Compress-Depth-Anything-V2/checkpoints_new_loss_001_smooth

python AsymKD_evaluate_affine_inv_gpu_ddp.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_diode_all.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/diode \
    --checkpoint_dir /home/wodon326/project/Compress-Depth-Anything-V2/checkpoints_new_loss_001_smooth

python AsymKD_evaluate_affine_inv_gpu_ddp.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_scannet_val.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/scannet \
    --checkpoint_dir /home/wodon326/project/Compress-Depth-Anything-V2/checkpoints_new_loss_001_smooth