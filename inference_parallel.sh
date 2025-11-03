for file_idx in {0..6}
do
    (
        export CUDA_VISIBLE_DEVICES=$file_idx
        python inference.py \
        --config_path configs/self_forcing_dmd.yaml \
        --checkpoint_path checkpointss/ema_model.pt \
        --output_folder videos/infinite_forcing_longer \
        --data_path prompts/${file_idx}.txt \
        --use_ema \
        --num_output_frames 1536
        )  &
done
wait