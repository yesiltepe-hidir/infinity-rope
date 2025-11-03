python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpointss/ema_model.pt \
    --output_folder videos/interactive \
    --data_path prompts/5.txt \
    --use_ema \
    --num_output_frames 168