python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpointss/ema_model.pt \
    --output_folder videos/infinite_forcing \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema \
    --num_output_frames 129