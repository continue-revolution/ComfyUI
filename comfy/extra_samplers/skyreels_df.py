import math

from typing import List
from tqdm import tqdm

import torch

from comfy.samplers import CFGGuider

from .uni_pc_diffusers import FlowUniPCMultistepScheduler # TODO: reduce code duplicate


class DiffusionForcingPipeline:
    """
    A pipeline for diffusion-based video generation tasks.

    This pipeline supports two main tasks:
    - Image-to-Video (i2v): Generates a video sequence from a source image
    - Text-to-Video (t2v): Generates a video sequence from a text description

    The pipeline integrates multiple components including:
    - A transformer model for diffusion
    - A VAE for encoding/decoding
    - A text encoder for processing text prompts
    - An image encoder for processing image inputs (i2v mode only)
    """

    def generate_timestep_matrix(
        self,
        num_frames,
        step_template,
        base_num_frames,
        ar_step=5,
        num_pre_ready=0,
        casual_block_size=1,
        shrink_interval_with_mask=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple]]:
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template) + 1
        num_frames_block = num_frames // casual_block_size
        base_num_frames_block = base_num_frames // casual_block_size
        if base_num_frames_block < num_frames_block:
            infer_step_num = len(step_template)
            gen_block = base_num_frames_block
            min_ar_step = infer_step_num / gen_block
            assert ar_step >= min_ar_step, f"ar_step should be at least {math.ceil(min_ar_step)} in your setting"
        # print(num_frames, step_template, base_num_frames, ar_step, num_pre_ready, casual_block_size, num_frames_block, base_num_frames_block)
        step_template = torch.cat(
            [
                torch.tensor([999], dtype=torch.int64, device=step_template.device),
                step_template.long(),
                torch.tensor([0], dtype=torch.int64, device=step_template.device),
            ]
        )  # to handle the counter in row works starting from 1
        pre_row = torch.zeros(num_frames_block, dtype=torch.long)
        if num_pre_ready > 0:
            pre_row[: num_pre_ready // casual_block_size] = num_iterations

        while torch.all(pre_row >= (num_iterations - 1)) == False:
            new_row = torch.zeros(num_frames_block, dtype=torch.long)
            for i in range(num_frames_block):
                if i == 0 or pre_row[i - 1] >= (
                    num_iterations - 1
                ):  # the first frame or the last frame is completely denoised
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i - 1] - ar_step
            new_row = new_row.clamp(0, num_iterations)

            update_mask.append(
                (new_row != pre_row) & (new_row != num_iterations)
            )  # False: no need to update， True: need to update
            step_index.append(new_row)
            step_matrix.append(step_template[new_row])
            pre_row = new_row

        # for long video we split into several sequences, base_num_frames is set to the model max length (for training)
        terminal_flag = base_num_frames_block
        if shrink_interval_with_mask:
            idx_sequence = torch.arange(num_frames_block, dtype=torch.int64)
            update_mask = update_mask[0]
            update_mask_idx = idx_sequence[update_mask]
            last_update_idx = update_mask_idx[-1].item()
            terminal_flag = last_update_idx + 1
        # for i in range(0, len(update_mask)):
        for curr_mask in update_mask:
            if terminal_flag < num_frames_block and curr_mask[terminal_flag]:
                terminal_flag += 1
            valid_interval.append((max(terminal_flag - base_num_frames_block, 0), terminal_flag))

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)

        if casual_block_size > 1:
            step_update_mask = step_update_mask.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_index = step_index.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_matrix = step_matrix.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            valid_interval = [(s * casual_block_size, e * casual_block_size) for s, e in valid_interval]

        return step_matrix, step_index, step_update_mask, valid_interval

    @torch.no_grad()
    def __call__(
        self,
        dit: CFGGuider,
        num_inference_steps: int,
        latents_full: torch.Tensor,
        shift: float = 8.0,
        overlap_history: int = 17,
        addnoise_condition: int = 20,
        base_num_frames: int = 97,
        ar_step: int = 5,
        causal_block_size: int = 5,
    ):
        # 2. Basic parameters setup
        device = dit.model_patcher.load_device
        b, c, f, h, w = latents_full.shape
        num_frames = (f - 1) * 4 + 1
        prefix_video = None
        predix_video_latent_length = 0
        scheduler = FlowUniPCMultistepScheduler()
        scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        init_timesteps = scheduler.timesteps

        # 4. Short video generation. TODO: not yet modified properly
        if overlap_history is None or base_num_frames is None or num_frames <= base_num_frames:
            latents = latents_full
            base_num_frames = (base_num_frames - 1) // 4 + 1 if base_num_frames is not None else f
            step_matrix, _, step_update_mask, valid_interval = self.generate_timestep_matrix(
                f, init_timesteps, base_num_frames, ar_step, predix_video_latent_length, causal_block_size
            )
            sample_schedulers: List[FlowUniPCMultistepScheduler] = []
            sample_schedulers_counter = [0] * f
            for _ in range(f):
                sample_scheduler = FlowUniPCMultistepScheduler()
                sample_scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
                sample_schedulers.append(sample_scheduler)
            for i, timestep_i in enumerate(tqdm(step_matrix)):
                update_mask_i = step_update_mask[i]
                valid_interval_i = valid_interval[i]
                valid_interval_start, valid_interval_end = valid_interval_i
                timestep = timestep_i[None, valid_interval_start:valid_interval_end].clone()
                latent_model_input = latents[:, :, valid_interval_start:valid_interval_end, :, :].clone()
                if addnoise_condition > 0 and valid_interval_start < predix_video_latent_length:
                    noise_factor = 0.001 * addnoise_condition
                    timestep_for_noised_condition = addnoise_condition
                    latent_model_input[:, :, valid_interval_start:predix_video_latent_length] = (
                        latent_model_input[:, :, valid_interval_start:predix_video_latent_length] * (1.0 - noise_factor)
                        + torch.randn_like(latent_model_input[:, :, valid_interval_start:predix_video_latent_length])
                        * noise_factor
                    )
                    timestep[:, valid_interval_start:predix_video_latent_length] = timestep_for_noised_condition
                noise_pred = dit(latent_model_input, timestep * 0.001)
                for idx in range(valid_interval_start, valid_interval_end):
                    if update_mask_i[idx].item():
                        latents[:, :, idx] = sample_schedulers[idx].step(
                            noise_pred[:, :, idx - valid_interval_start],
                            timestep_i[idx],
                            latents[:, :, idx],
                            return_dict=False,
                        )[0]
                        sample_schedulers_counter[idx] += 1
            return [latents_full]
        # 4. Long video generation (sliding window)
        else:
            base_num_frames = (base_num_frames - 1) // 4 + 1 if base_num_frames is not None else f
            overlap_history_frames = (overlap_history - 1) // 4 + 1
            n_iter = 1 + (f - base_num_frames - 1) // (base_num_frames - overlap_history_frames) + 1
            print(f"# of large sliding windows: {n_iter}")
            # 4.1 Large sliding window: each sliding window goes through DiT as a short video, but only a few contribute to latent updates.
            latents_base = []
            for i in range(n_iter):
                if i > 0:  # i !=0
                    prefix_video = latents[:, :, -overlap_history_frames:]
                    if prefix_video.shape[2] % causal_block_size != 0:
                        truncate_len = prefix_video.shape[2] % causal_block_size
                        print("the length of prefix video is truncated for the casual block size alignment.")
                        prefix_video = prefix_video[:, :, : prefix_video.shape[2] - truncate_len]
                    predix_video_latent_length = prefix_video.shape[2]
                    finished_frame_num = i * (base_num_frames - overlap_history_frames) + overlap_history_frames
                    left_frame_num = f - finished_frame_num
                    base_num_frames_iter = min(left_frame_num + overlap_history_frames, base_num_frames)
                    latents = latents_full[:, :, finished_frame_num - predix_video_latent_length : finished_frame_num - predix_video_latent_length + base_num_frames_iter, :, :]
                else:  # i == 0
                    base_num_frames_iter = base_num_frames
                    latents = latents_full[:, :, :base_num_frames_iter, :, :]
                # 4.2 Decide the step of each frame in the sliding window
                step_matrix, _, step_update_mask, valid_interval = self.generate_timestep_matrix(
                    base_num_frames_iter,
                    init_timesteps,
                    base_num_frames_iter,
                    ar_step,
                    predix_video_latent_length,
                    causal_block_size,
                )
                # 4.3 Prepare sample schedulers for each frame
                sample_schedulers = []
                sample_schedulers_counter = [0] * base_num_frames_iter
                for _ in range(base_num_frames_iter):
                    sample_scheduler = FlowUniPCMultistepScheduler()
                    sample_scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                # 4.4 Denoise the short video in the sliding window
                for j, timestep_i in enumerate(tqdm(step_matrix)):
                    update_mask_i = step_update_mask[j]
                    valid_interval_i = valid_interval[j]
                    valid_interval_start, valid_interval_end = valid_interval_i
                    timestep = timestep_i[None, valid_interval_start:valid_interval_end].clone()
                    latent_model_input = latents[:, :, valid_interval_start:valid_interval_end, :, :].clone()
                    if addnoise_condition > 0 and valid_interval_start < predix_video_latent_length:
                        noise_factor = 0.001 * addnoise_condition
                        timestep_for_noised_condition = addnoise_condition
                        latent_model_input[:, :, valid_interval_start:predix_video_latent_length] = (
                            latent_model_input[:, :, valid_interval_start:predix_video_latent_length]
                            * (1.0 - noise_factor)
                            + torch.randn_like(
                                latent_model_input[:, :, valid_interval_start:predix_video_latent_length]
                            )
                            * noise_factor
                        )
                        timestep[:, valid_interval_start:predix_video_latent_length] = timestep_for_noised_condition
                    noise_pred = dit(latent_model_input, timestep * 0.001)
                    for idx in range(valid_interval_start, valid_interval_end):
                        if update_mask_i[idx].item():
                            latents[:, :, idx] = sample_schedulers[idx].step(
                                noise_pred[:, :, idx - valid_interval_start],
                                timestep_i[idx],
                                latents[:, :, idx],
                                return_dict=False,
                            )[0]
                            sample_schedulers_counter[idx] += 1
                latents_base.append(latents.clone())
            return latents_base
