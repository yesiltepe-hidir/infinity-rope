# torchrun --nproc_per_node 8 scripts/generate_ode_pairs.py --output_folder  data/ode_pairs --caption_path prompts/mixkit_prompts.txt --resume --num_samples 6480
python scripts/create_lmdb_iterative.py  --data_path data/ode_pairs --lmdb_path data/ode_pairs_lmdb 
