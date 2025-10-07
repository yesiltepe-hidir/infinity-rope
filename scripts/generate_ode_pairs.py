from utils.distributed import launch_distributed_job
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from utils.dataset import TextDataset
import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math
import os


def init_model(device):
    model = WanDiffusionWrapper().to(device).to(torch.float32)
    encoder = WanTextEncoder().to(device).to(torch.float32)
    model.model.requires_grad_(False)

    scheduler = FlowMatchScheduler(
        shift=8.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=48, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

    unconditional_dict = encoder(
        text_prompts=[sample_neg_prompt]
    )

    return model, encoder, scheduler, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_samples", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing multiple samples simultaneously")

    args = parser.parse_args()

    launch_distributed_job()

    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, encoder, scheduler, unconditional_dict = init_model(device=device)

    dataset = TextDataset(args.caption_path)

    # if global_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

    # Calculate total batches per process
    samples_per_process = int(math.ceil(args.num_samples / dist.get_world_size()))
    total_batches = int(math.ceil(samples_per_process / args.batch_size))
    is_main_process = dist.get_rank() == 0

    outer_pbar = tqdm(
        range(total_batches),
        disable=not is_main_process,
        desc="Generating ODE pairs",
        dynamic_ncols=True
    )

    for batch_idx in outer_pbar:
        # Calculate sample indices for this batch
        start_idx = batch_idx * args.batch_size
        batch_prompt_indices = []
        batch_prompts = []
        
        for i in range(args.batch_size):
            prompt_index = (start_idx + i) * dist.get_world_size() + dist.get_rank()
            if prompt_index >= args.num_samples:
                break
            batch_prompt_indices.append(prompt_index)
            
            prompt_entry = dataset[prompt_index]
            prompt = prompt_entry["prompts"]
            
            if isinstance(prompt, str):
                batch_prompts.extend([prompt])
            else:
                batch_prompts.extend(prompt)
        
        if not batch_prompt_indices:
            continue
            
        current_batch_size = len(batch_prompt_indices)
        
        # Encode all prompts in the batch
        conditional_dict = encoder(text_prompts=batch_prompts)
        
        # Create latents for the entire batch
        latents = torch.randn(
            [current_batch_size, 21, 16, 60, 104], dtype=torch.float32, device=device
        )

        noisy_input = []

        # Inner tqdm for timesteps, only on main process, with clear desc
        inner_pbar = tqdm(
            enumerate(scheduler.timesteps),
            total=len(scheduler.timesteps),
            disable=not is_main_process,
            leave=False,
            desc=f"Batch {batch_idx+1}/{total_batches} (samples {batch_prompt_indices[0]:05d}-{batch_prompt_indices[-1]:05d})",
            dynamic_ncols=True
        )

        for progress_id, t in inner_pbar:
            timestep = t * torch.ones([current_batch_size, 21], device=device, dtype=torch.float32)

            noisy_input.append(latents)

            _, x0_pred_cond = model(
                latents, conditional_dict, timestep
            )

            # Repeat unconditional_dict for batch processing
            unconditional_dict_batch = {
                k: v.repeat(current_batch_size, *[1] * (v.dim() - 1)) if v.dim() > 0 else v
                for k, v in unconditional_dict.items()
            }

            _, x0_pred_uncond = model(
                latents, unconditional_dict_batch, timestep
            )

            x0_pred = x0_pred_uncond + args.guidance_scale * (
                x0_pred_cond - x0_pred_uncond
            )

            flow_pred = model._convert_x0_to_flow_pred(
                scheduler=scheduler,
                x0_pred=x0_pred.flatten(0, 1),
                xt=latents.flatten(0, 1),
                timestep=timestep.flatten(0, 1)
            ).unflatten(0, x0_pred.shape[:2])

            latents = scheduler.step(
                flow_pred.flatten(0, 1),
                scheduler.timesteps[progress_id] * torch.ones(
                    [current_batch_size, 21], device=device, dtype=torch.long).flatten(0, 1),
                latents.flatten(0, 1)
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])

        noisy_input.append(latents)

        noisy_inputs = torch.stack(noisy_input, dim=2)  # Changed dim from 1 to 2 for batch processing

        noisy_inputs = noisy_inputs[:, :, [0, 12, 24, 36, -1]]

        # Save each sample in the batch individually
        for i, prompt_index in enumerate(batch_prompt_indices):
            stored_data = noisy_inputs[i:i+1]  # Keep batch dimension for consistency
            prompt = batch_prompts[i] if i < len(batch_prompts) else batch_prompts[0]
            
            torch.save(
                {prompt: stored_data.cpu().detach()},
                os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
            )

    dist.barrier()


if __name__ == "__main__":
    main()
