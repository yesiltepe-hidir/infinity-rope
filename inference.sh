python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --output_folder videos/self_forcing_rolling_rope \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema \
    --num_output_frames 129