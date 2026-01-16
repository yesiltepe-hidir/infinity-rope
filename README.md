<p align="center">
<h1 align="center">Infinity-RoPE</h1>
<h3 align="center">Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout</h3>
</p>
<p align="center">
  <p align="center">
    <a href="https://yesiltepe-hidir.github.io/">Hidir Yesiltepe</a><sup>1</sup>
    ·
    <a href="https://tmeral.com/">Tuna Han Salih Meral</a><sup>1</sup>
    ·
    <a href="https://kaanakan.github.io/">Adil Kaan Akan</a><sup>2</sup>
    ·
    <a href="https://kaanoktay.github.io/">Kaan Oktay</a><sup>2</sup>
    ·
    <a href="https://pinguar.org/">Pinar Yanardag</a><sup>1</sup><br>
    <sup>1</sup>Virginia Tech <sup>2</sup>fal
  <p align="center">
  <a href="https://arxiv.org/abs/2511.20649">
    <img src="https://img.shields.io/badge/arXiv-2506.08009-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://infinity-rope.github.io/">
    <img src="https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white" alt="Website">
  </a>
  <a href="https://github.com/yesiltepe-hidir/infinity-rope">
    <img src="https://komarev.com/ghpvc/?username=yesiltepe-hidir&repo=infinity-rope&label=Visitors&color=0078d4&style=flat" alt="Visitors">
  </a>
</p>

---





https://github.com/user-attachments/assets/2f91630c-c536-4e25-995e-5167fd000c95




## Requirements
We tested this repo on the following setup:
* Nvidia GPU with at least 24 GB memory (RTX 4090, A100, and H100 are tested).
* Linux operating system.
* 64 GB RAM.

Other hardware setup could also work but hasn't been tested.

## Installation
Create a Python 3.10 environment, install dependencies, and download models:
```
bash setup_env.sh
```

## Inference
```
bash inference.sh
```

## 🎬 Prompting Structure

Infinity-RoPE utilizes a specific syntax to control temporal duration and scene transitions. The core format for a segment is:  
`"action_description[duration]"`

### 1. Syntax Overview

| Operator | Name | Function |
| :--- | :--- | :--- |
| **`[Ns]`** | **Duration** | Sets the segment length in seconds (e.g., `[10s]`). |
| **`\|`** | **Separator** | Chains multiple action prompts together. |
| **`#`** | **Scene Cut** | When placed inside brackets (e.g., `[10s#]`), it triggers a hard cut. |
| **`;`** | **Subtitle Toggle** | Separates action prompts (left) from subtitle text (right). |
---

### 2. Examples

#### **Continuous Single Action**
Generates one seamless video of the specified length.
> `"action_1_prompt[30s]"`
* **Total Length:** 30s
* **Result:** A single 30-second continuous shot.

#### **Multi-Action (Smooth Transition)**
Transitions between different behaviors within a single, continuous camera shot.
> `"action_1_prompt[5s] | action_2_prompt[10s] | action_3_prompt[15s]"`
* **Total Length:** 30s ($5s + 10s + 15s$)
* **Result:** The subject transitions naturally from action 1 to 2 to 3 without a camera break.

#### **Multi-Scene (Cinematic Cuts)**
Forces the model to perform a hard jump-cut at the beginning of specific segments.
> `"action_1_prompt[10s] | action_2_prompt[10s#] | action_3_prompt[10s#]"`
* **Total Length:** 30s
* **Result:** Three distinct 10-second scenes. The `#` at the start of action 2 and 3 initiates the scene cuts.

#### **Multi-Scene with Subtitles**
Combines scene cuts with synchronized text overlays.
> `"action_1[10s] | action_2[10s#] | action_3[10s#] ; subtitle_1 | subtitle_2 | subtitle_3"`
* **Total Length:** 30s
* **Result:** Three distinct 10-second scenes. Each segment displays its corresponding subtitle from the list provided after the `;`.

---

Note:
* **Our model works better with long, detailed prompts** since it's trained with such prompts. We will integrate prompt extension into the codebase (similar to [Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/main?tab=readme-ov-file#2-using-prompt-extention)) in the future. For now, it is recommended to use third-party LLMs (such as GPT-4o) to extend your prompt before providing to the model.
* You may want to adjust FPS so it plays smoothly on your device.
* The speed can be improved by enabling `torch.compile`, [TAEHV-VAE](https://github.com/madebyollin/taehv/), or using FP8 Linear layers, although the latter two options may sacrifice quality. It is recommended to use `torch.compile` if possible and enable TAEHV-VAE if further speedup is needed.

## Training
### Download text prompts and ODE initialized checkpoint
```
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
```
Note: Our training algorithm (except for the GAN version) is data-free (**no video data is needed**). For now, we directly provide the ODE initialization checkpoint and will add more instructions on how to perform ODE initialization in the future (which is identical to the process described in the [CausVid](https://github.com/tianweiy/CausVid) repo).

### Self Forcing Training with DMD
```
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_dmd \
  --disable-wandb
```
Our training run uses 600 iterations and completes in under 2 hours using 64 H100 GPUs. By implementing gradient accumulation, it should be possible to reproduce the results in less than 16 hours using 8 H100 GPUs.

## Acknowledgements
This codebase is built on top of the open-source implementation of [Self-Forcing](https://github.com/guandeh17/Self-Forcing). We also appreciate [Infinite-Forcing](https://github.com/SOTAMak1r/Infinite-Forcing) for providing an attention sink checkpoint.

## Citation
If you find this codebase useful for your research, please kindly cite our paper:
```
@article{yesiltepe2025infinity,
  title={Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout},
  author={Yesiltepe, Hidir and Meral, Tuna Han Salih and Akan, Adil Kaan and Oktay, Kaan and Yanardag, Pinar},
  journal={arXiv preprint arXiv:2511.20649},
  year={2025}
}
```
