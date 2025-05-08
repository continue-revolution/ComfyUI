import math
import torch

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


class RecifitedFlowScheduler:
    def __init__(
        self,
        shift=1.0,
        sigma_min=0.001,
        sigma_max=1.0,
        num_train_timesteps=1000,
        use_dynamic_shifting=False,
    ):
        self.shift = shift
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_train_timesteps = num_train_timesteps
        self.use_dynamic_shifting = use_dynamic_shifting

    def _sigma_to_t(self, sigma):
        return sigma * self.num_train_timesteps

    def _t_to_sigma(self, t):
        return t / self.num_train_timesteps

    def _time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _shift_sigma(self, sigma: torch.Tensor, shift: float):
        return shift * sigma / (1 + (shift - 1) * sigma)

    def schedule(
        self,
        num_inference_steps: int,
        mu: float | None = None,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
    ):
        sigma_min = self.sigma_min if sigma_min is None else sigma_min
        sigma_max = self.sigma_max if sigma_max is None else sigma_max
        sigmas = torch.linspace(sigma_max, sigma_min, num_inference_steps)
        if self.use_dynamic_shifting:
            sigmas = self._time_shift(mu, 1.0, sigmas)  # FLUX
        else:
            sigmas = self._shift_sigma(sigmas, self.shift)
        timesteps = sigmas * self.num_train_timesteps
        sigmas = append_zero(sigmas)
        return sigmas, timesteps



class FlowMatchEulerSampler:
    def __init__(self, timesteps: torch.Tensor, sigmas: torch.Tensor):
        self.timesteps = timesteps
        self.sigmas = sigmas
        self.step_idx = 0

    def step(self, model_outputs: torch.Tensor, _, latents: torch.Tensor):
        i = self.step_idx
        dt = self.sigmas[i + 1] - self.sigmas[i]
        latents = latents.to(dtype=torch.float32)
        latents = latents + model_outputs * dt
        latents = latents.to(dtype=model_outputs.dtype)
        self.step_idx += 1
        return latents

    def add_noise(self, latents, noise, sigma):
        return (1 - sigma) * latents + noise * sigma