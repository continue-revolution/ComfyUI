from types import MethodType

import torch

import comfy.sample
from comfy.samplers import CFGGuider, process_conds, KSampler
import comfy.sampler_helpers

from comfy.extra_samplers.skyreels_df import DiffusionForcingPipeline

class SkyReelsDFSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dit": ("MODEL", {"tooltip": "The DiT model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (["euler"], {"default": "uni_pc", "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (KSampler.SCHEDULERS, {"default": "simple", "tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                # TODO: It could be tricky to explain these parameters to users. For now let's use default.
                "overlap_history": ("INT", {"default": 17, "min": 0, "max": 50, "tooltip": "Number of frames to overlap for smooth transitions in long videos"}),
                "addnoise_condition": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Improves consistency in long video generation"}),
                "base_num_frames": ("INT", {"default": 97, "min": 97, "max": 121, "tooltip": "Base frame count (**97 for 540P**, **121 for 720P**)"}),
                "ar_step": ("INT", {"default": 5, "min": 0, "max": 10, "tooltip": "Controls asynchronous inference (0 for synchronous mode)"}),
                "causal_block_size": ("INT", {"default": 5, "min": 1, "max": 10, "tooltip": "Recommended when using asynchronous inference (--ar_step > 0)"}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    OUTPUT_TOOLTIPS = ("The decoded latent.", "Number of frames to overlap for smooth transitions in long videos. Connect this to overlap input at Node SkyReelsVAEDecode")
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, dit, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, overlap_history=17, addnoise_condition=20, base_num_frames=97, ar_step=5, causal_block_size=5):
        # copied from comfy.nodes.common_sampler
        latent = latent_image
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(dit, latent_image)

        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        # copied from comfy.sample.sample
        device = dit.load_device
        sampler = KSampler(dit, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=dit.model_options)

        # copied from comfy.samplers.sample
        cfg_guider = CFGGuider(dit)
        cfg_guider.set_conds(positive, negative)
        cfg_guider.set_cfg(cfg)
        cfg_guider.conds = {}
        for k in cfg_guider.original_conds:
            cfg_guider.conds[k] = list(map(lambda a: a.copy(), cfg_guider.original_conds[k]))
        cfg_guider.inner_model, cfg_guider.conds, cfg_guider.loaded_models = comfy.sampler_helpers.prepare_sampling(cfg_guider.model_patcher, noise.shape, cfg_guider.conds, cfg_guider.model_options)
        cfg_guider.conds = process_conds(cfg_guider.inner_model, noise, cfg_guider.conds, device, latent_image, seed=seed)
        noise = noise.to(device)

        try:
            cfg_guider.model_patcher.pre_run()
            original_calculte_denoised = cfg_guider.inner_model.model_sampling.calculate_denoised
            def identity_calculate_denoised(self, sigma, model_output, model_input):
                return model_output
            cfg_guider.inner_model.model_sampling.calculate_denoised = MethodType(identity_calculate_denoised, cfg_guider.inner_model.model_sampling)
            samples = DiffusionForcingPipeline()(
                dit=cfg_guider,
                latents_full=noise,
                overlap_history=overlap_history,
                addnoise_condition=addnoise_condition,
                base_num_frames=base_num_frames,
                ar_step=ar_step,
                causal_block_size=causal_block_size,
                sampler=sampler
            )
            cfg_guider.inner_model.model_sampling.calculate_denoised = original_calculte_denoised
        finally:
            cfg_guider.model_patcher.cleanup()
        comfy.sampler_helpers.cleanup_models(cfg_guider.conds, cfg_guider.loaded_models)

        out = latent.copy()
        out["samples"] = samples
        return (out, overlap_history)


class SkyReelsVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The latent to be decoded."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."}),
                "overlap": ("INT", {"tooltip": "Number of frames to overlap for smooth transitions in long videos"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The decoded image.",)
    FUNCTION = "decode"

    CATEGORY = "latent"
    DESCRIPTION = "Decodes latent images back into pixel space images."

    def decode(self, vae, samples, overlap):
        output_frames = None
        for sample in samples["samples"]:
            output_frames = vae.decode(sample) if output_frames is None else torch.cat((output_frames, vae.decode(sample)[:, overlap:]), dim=1)
        if len(output_frames.shape) == 5: #Combine batches
            output_frames = output_frames.reshape(-1, output_frames.shape[-3], output_frames.shape[-2], output_frames.shape[-1])
        return (output_frames, )


NODE_CLASS_MAPPINGS = {
    "SkyReelsDFSampler": SkyReelsDFSampler,
    "SkyReelsVAEDecode": SkyReelsVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SkyReelsDFSampler": "SkyReels DF Sampler",
    "SkyReelsVAEDecode": "SkyReels VAE Decode",
}