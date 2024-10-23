python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
     --batch_size 256 \
     --accum_iter 1\
     --model mae_vit_base_patch16 \
     --norm_pix_loss \
     --mask_ratio 0.75 \
     --epochs 200 \
     --warmup_epochs 40 \
     --blr 1.5e-4 --weight_decay 0.05 \
     --data_path /home/hccs/lhp/dataset/imagenet \
     --output_dir  ./scm_target_dp1_bs1024 \
     --log_dir ./scm_target_dp1_bs1024 


OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --accum_iter 1 \
    --batch_size 256 \
    --model vit_base_patch16 \
    --finetune ./scm_target_dp1_bs1024/checkpoint-199.pth \
    --epochs 100 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /home/hccs/lhp/dataset/imagenet \
    --output_dir ./scm_target_dp1_bs1024_ep200_ft\
    --log_dir ./scm_target_dp1_bs1024_ep200_ft