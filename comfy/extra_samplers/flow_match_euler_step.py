import torch


class FlowMatchEulerSampler:
    def __init__(self, sigmas: torch.Tensor):
        self.sigmas = sigmas

    def step(self, model_outputs: torch.Tensor, timestep_idx: int, latents: torch.Tensor):
        dt = self.sigmas[timestep_idx + 1] - self.sigmas[timestep_idx]
        latents = latents.to(dtype=torch.float32)
        latents = latents + model_outputs * dt
        latents = latents.to(dtype=model_outputs.dtype)
        return latents

    def add_noise(self, latents, noise, sigma):
        return (1 - sigma) * latents + noise * sigma