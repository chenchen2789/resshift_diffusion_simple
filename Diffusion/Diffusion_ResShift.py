import math
import numpy as np
import torch as th


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class DiffusionResShift:
    def __init__(self, configs):
        opt = configs['params']
        self.schedule_name = opt['schedule_name']
        if self.schedule_name == 'exponential':
            power = opt['schedule_kwargs']['power']
            min_noise_level = opt['min_noise_level']
            kappa = opt['kappa']
            num_diffusion_timesteps = opt['steps']
            etas_end = opt['etas_end']
            etas_start = min(min_noise_level / kappa, min_noise_level)
            increaser = math.exp(1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start))
            base = np.ones([num_diffusion_timesteps, ]) * increaser
            power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power
            power_timestep *= (num_diffusion_timesteps - 1)
            self.sqrt_etas = np.power(base, power_timestep) * etas_start
            self.etas = self.sqrt_etas ** 2
        else:
            raise KeyError(f'{self.schedule_name} is not a valid schedule')

        self.kappa = opt['kappa']
        self.normalize_input = opt['normalize_input']
        self.latent_flag = opt['latent_flag']
        self.num_diffusion_timesteps = num_diffusion_timesteps

        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas
        self.posterior_variance = self.kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
            self.posterior_variance[1], self.posterior_variance[1:]
        )
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)




    def forward_addnoise(self, x_start, y, t, noise):
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        )

    def inverse_denoise(self, x_start, x_t, t, noise):
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        ) #nos noise when t==0

        output = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
            + nonzero_mask * th.exp(0.5 * _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)) * noise
        )
        return output

    def prior_sample(self, y, noise):
        t = th.tensor([self.num_diffusion_timesteps - 1, ] * y.shape[0], device=y.device).long()
        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def scale_input_resshift(self, inputs, t):
        if self.normalize_input:
            if self.latent_flag:
                std = th.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * self.kappa**2 + 1)
                inputs_norm = inputs / std
            else:
                inputs_max = _extract_into_tensor(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs

        return inputs_norm



