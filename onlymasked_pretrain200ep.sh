cd $LPAI_CODE_DIR_0 ;

unset LD_LIBRARY_PATH ;

source /root/anaconda3/etc/profile.d/conda.sh ;

conda activate pami ;

git clone -b pami_sdae --single-branch https://github.com/showstarpro/mae.git pami_sdae ;

cd ./pami_sdae ;

OMP_NUM_THREADS=1 torchrun --nproc-per-node=8 --master-port=29501 main_pretrain.py \
    --batch_size 256 \
    --accum_iter 1\
    --epochs 200 \
    --model mae_vit_base_patch16 \
    --model_teacher vit_base_patch16 \
    --data_path $LPAI_INPUT_DATASET_0 \
    --warmup_epochs 40 \
    --mask_ratio 0.75 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --ema_op per_epoch \
    --ema_frequent 1 \
    --momentum_teacher 0.96 \
    --momentum_teacher_final 0.99 \
    --drop_path 0.25 \
    --shrink_num 147 \
    --output_dir /lpai/output/models/lyy/pami_sdae_onlymasked_200ep/pre \
    --log_dir /lpai/output/models/lyy/pami_sdae_onlymasked_200ep/pre ;

OMP_NUM_THREADS=1 torchrun --nproc-per-node=8 --master-port=29502 main_finetune.py \
    --accum_iter 1 \
    --batch_size 128 \
    --model vit_base_patch16 \
    --finetune /lpai/output/models/lyy/pami_sdae_onlymasked_200ep/pre/checkpoint-199.pth \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 1e-3 \
    --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval \
    --data_path $LPAI_INPUT_DATASET_0 \
    --output_dir /lpai/output/models/lyy/pami_sdae_onlymasked_200ep/fin \
    --log_dir /lpai/output/models/lyy/pami_sdae_onlymasked_200ep/fin ;

OMP_NUM_THREADS=1 torchrun --nproc-per-node=8 --master-port=29503 main_linprobe.py \
    --accum_iter 1 \
    --batch_size 2048 \
    --model vit_base_patch16 \
    --cls_token \
    --finetune /lpai/output/models/lyy/pami_sdae_onlymasked_200ep/pre/checkpoint-199.pth \
    --epochs 90 \
    --warmup_epochs 10 \
    --blr 0.1 \
    --weight_decay 0.00 \
    --dist_eval \
    --data_path $LPAI_INPUT_DATASET_0 \
    --output_dir /lpai/output/models/lyy/pami_sdae_onlymasked_200ep/ln \
    --log_dir /lpai/output/models/lyy/pami_sdae_onlymasked_200ep/ln ;

sleep 1d ;