MASTER_ADDR=localhost
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_dmd \
  --disable-wandb