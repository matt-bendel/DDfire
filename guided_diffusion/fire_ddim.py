import math
import numpy as np
import torch
from tqdm.auto import tqdm

from guided_diffusion.fire import FIRE
from .posterior_mean_variance import get_mean_processor, get_var_processor

__SAMPLER__ = {}


def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __SAMPLER__[name] = cls
        return cls

    return wrapper


def get_sampler(name: str):
    if __SAMPLER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name]


def create_sampler(sampler,
                   steps,
                   noise_schedule,
                   model_mean_type,
                   model_var_type,
                   dynamic_threshold,
                   clip_denoised,
                   rescale_timesteps,
                   timestep_respacing="",
                   eta=1.0,
                   delta_1=0.25,
                   nfe_budget=100,
                   clamp_denoise=True,
                   clamp_fire_ddim=True):
    sampler = get_sampler(name=sampler)

    betas = get_named_beta_schedule(noise_schedule, steps)
    new_betas = get_named_beta_schedule('linear', steps)
    if not timestep_respacing:
        timestep_respacing = [steps]

    alphas = 1.0 - new_betas
    alpha_bars = np.cumprod(alphas, axis=0)
    ddpm_vars = (1 - alpha_bars) / alpha_bars

    return sampler(use_timesteps=space_timesteps(steps, timestep_respacing, ddpm_vars),
                   betas=new_betas,
                   betas_model=betas,
                   model_mean_type=model_mean_type,
                   model_var_type=model_var_type,
                   dynamic_threshold=dynamic_threshold,
                   clip_denoised=clip_denoised,
                   rescale_timesteps=rescale_timesteps,
                   delta_1=delta_1,
                   nfe_budget=nfe_budget,
                   clamp_denoise=clamp_denoise,
                   clamp_fire_ddim=clamp_fire_ddim)


class GaussianDiffusion:
    def __init__(self,
                 betas,
                 betas_model,
                 model_mean_type,
                 model_var_type,
                 dynamic_threshold,
                 clip_denoised,
                 rescale_timesteps,
                 delta_1=0.1,
                 nfe_budget=100,
                 clamp_denoise=True,
                 clamp_fire_ddim=True
                 ):

        # use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        betas_model = np.array(betas_model, dtype=np.float64)
        self.betas = betas
        self.betas_model = betas_model
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (0 < self.betas).all() and (self.betas <= 1).all(), "betas must be in (0..1]"

        self.num_timesteps = int(self.betas.shape[0])
        self.rescale_timesteps = rescale_timesteps

        alphas = 1.0 - self.betas
        alphas_model = 1.0 - self.betas_model
        self.alphas_cumprod_model = np.cumprod(alphas_model, axis=0)
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )

        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

        self.mean_processor = get_mean_processor(model_mean_type,
                                                 betas=betas,
                                                 dynamic_threshold=dynamic_threshold,
                                                 clip_denoised=clip_denoised)

        self.var_processor = get_var_processor(model_var_type,
                                               betas=betas)

        self.clamp_denoise = clamp_denoise
        self.clamp_fire_ddim = clamp_fire_ddim

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """

        mean = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start) * x_start
        variance = extract_and_expand(1.0 - self.alphas_cumprod, t, x_start)
        log_variance = extract_and_expand(self.log_one_minus_alphas_cumprod, t, x_start)

        return mean, variance, log_variance

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start)
        coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)

        return coef1 * x_start + coef2 * noise, noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
        posterior_mean = coef1 * x_start + coef2 * x_t
        posterior_variance = extract_and_expand(self.posterior_variance, t, x_t)
        posterior_log_variance_clipped = extract_and_expand(self.posterior_log_variance_clipped, t, x_t)

        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_sample_loop(self,
                      model,
                      H,
                      x_start,
                      measurement,
                      noise_sig=0.001,
                      eta=1.0,
                      sqrt_in_var_to_out=''):
        """
        The function used for sampling from noise.
        """
        img = x_start
        device = x_start.device

        fire_runner = FIRE(model, self.alphas_cumprod_model, x_start, H, sqrt_in_var_to_out, self.clamp_denoise)
        fire_runner.rho = self.rho

        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        count = 0
        for idx in pbar:
            fire_iter = int(self.fire_iter_schedule[idx])
            fire_runner.max_iters = fire_iter

            time = torch.tensor([idx] * img.shape[0], device=device)
            img = self.p_sample(x=img, t=time, model=model, y=measurement, cond=True, fire=fire_runner, noise_sig=noise_sig, eta=eta)['sample'].detach()
            count += 1

        return img.clamp(min=-1., max=1.), 100

    def denoise(self, model, x, t, y, cond, fire, noise_sig):
        raise NotImplementedError

    def p_mean_variance(self, model, x, t):
        model_output = model(x, self._scale_timesteps(t))

        # In the case of "learned" variance, model will give twice channels.
        if model_output.shape[1] == 2 * x.shape[1]:
            model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
        else:
            # The name of variable is wrong.
            # This will just provide shape information, and
            # will not be used for calculating something important in variance.
            model_var_values = model_output

        model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, model_output)
        model_variance, model_log_variance = self.var_processor.get_variance(model_var_values, t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {'mean': model_mean,
                'variance': model_variance,
                'log_variance': model_log_variance,
                'pred_xstart': pred_xstart}

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t


def space_timesteps(num_timesteps, section_counts, ddpm_vars):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    K = int(section_counts[1])

    ddpm_prec = 1 / ddpm_vars
    ddpm_prec_log = np.log10(ddpm_prec)
    prec_vals = np.linspace(np.min(ddpm_prec_log), np.max(ddpm_prec_log), K)
    ddpm_prec_new = 10 ** prec_vals
    all_steps = []
    for i in range(K):
        t = np.argmin(np.abs(ddpm_prec - ddpm_prec_new[i]), axis=0)
        all_steps.append(t)

    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)

        self.fire_iter_schedule = []
        self.rho = 1.25
        one_iter_frac = kwargs["delta_1"]
        nfe_budget = kwargs["nfe_budget"]

        ddim_steps = list(use_timesteps)
        ddim_steps.sort()
        K = len(ddim_steps)

        ddpm_prec = base_diffusion.alphas_cumprod[ddim_steps] / (1 - base_diffusion.alphas_cumprod[ddim_steps])

        one_iter_num = int(np.round(K * one_iter_frac))
        new_gamma_tgt = -1
        new_diff = []
        for i in range(K):
            if i >= one_iter_num:
                if new_gamma_tgt == -1:
                    log_gam_tgt = np.log10(ddpm_prec[i])
                    new_gamma_tgt = ddpm_prec[i]

                new_diff.append(np.log10(new_gamma_tgt) - np.log10(ddpm_prec[i]))

        log_gam_ddim_ = np.log10(ddpm_prec)
        log_rho_min = 1e-5  # initialize bisection
        log_rho_max = 1e6  # initialize bisection

        NFE_min = np.sum(np.round(np.maximum((log_gam_tgt - log_gam_ddim_) / log_rho_min, 0) + 1))
        NFE_max = np.sum(np.round(np.maximum((log_gam_tgt - log_gam_ddim_) / log_rho_max, 0) + 1))
        NFE_tgt = nfe_budget
        if NFE_max > NFE_tgt:
            raise ValueError('NFE_tgt must be > ' + str(NFE_max))

        # bisection
        while True:
            log_rho_mid = 0.5 * (log_rho_min + log_rho_max)
            NFE_mid = np.sum(np.round(np.maximum((log_gam_tgt - log_gam_ddim_) / log_rho_mid, 0) + 1))
            # print('NFE_mid=',NFE_mid)
            if NFE_mid > NFE_tgt:
                log_rho_min = log_rho_mid
                NFE_min = NFE_mid
            elif NFE_mid < NFE_tgt:
                log_rho_max = log_rho_mid
                NFE_max = NFE_mid
            else:  # NFE_mid==NFE_tgt
                rho = 10 ** log_rho_mid
                self.rho = rho
                break

        iters_ire_ = np.round(np.maximum((log_gam_tgt - log_gam_ddim_) / log_rho_mid, 0) + 1).astype(int)
        self.fire_iter_schedule = iters_ire_.tolist()

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
            self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
            self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


@register_sampler(name='ddpm')
class DDPM(SpacedDiffusion):
    def p_sample(self, model, x, t, y, cond, fire, noise_sig, eta=1.):
        eta = 1. # This is DDPM, so force this to be safe..

        out = self.denoise(model, x, t, y, cond, fire, noise_sig)

        eps = self.predict_eps_from_x_start(x, t, out['pred_xstart'])

        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
        sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
                out['pred_xstart'] * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        sample = mean_pred
        if t[0] != 0:
            sample += sigma * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2

    def denoise(self, model, x, t, y, cond, fire, noise_sig):
        if not cond:
            pred_xstart = self.p_mean_variance(model, x, t)['pred_xstart']
        else:
            pred_xstart = fire.run_fire(x, y, extract_and_expand(self.alphas_cumprod, t, x)[0, 0, 0, 0], noise_sig=torch.tensor(noise_sig).to(x.device))

        return {'pred_xstart': pred_xstart}



@register_sampler(name='ddim')
class DDIM(SpacedDiffusion):
    def denoise(self, model, x, t, y, cond, fire, noise_sig):
        if not cond:
            pred_xstart = self.p_mean_variance(model, x, t)['pred_xstart']
        else:
            pred_xstart = fire.run_fire(x, y, extract_and_expand(self.alphas_cumprod, t, x)[0, 0, 0, 0], noise_sig=torch.tensor(noise_sig).to(x.device))

        return {'pred_xstart': pred_xstart}

    def p_sample(self, model, x, t, y, cond, fire, noise_sig, eta=0.85):
        out = self.denoise(model, x, t, y, cond, fire, noise_sig)
        pred_x0 = out['pred_xstart']
        if self.clamp_fire_ddim:
            pred_x0 = pred_x0.clamp(min=-1., max=1.)

        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)

        c1 = eta * ((1 - alpha_bar / alpha_bar_prev) * (1 - alpha_bar_prev) / (1 - alpha_bar)).sqrt()
        c2 = ((1 - alpha_bar_prev) - c1 ** 2).sqrt()

        eps_t = (x - pred_x0 * alpha_bar.sqrt()) / (1 - alpha_bar).sqrt()

        noise = torch.randn_like(x)

        sample = alpha_bar_prev.sqrt() * pred_x0.clamp(min=-1., max=1.)
        sample += c2 * eps_t
        if t[0] != 0:
            sample += c1 * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2


# =================
# Helper functions
# =================

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "poly_4":
        p = 4
        beta_start = 1e-8
        beta_end = 0.02
        return beta_start + (beta_end-beta_start) * np.linspace(0,1,num_diffusion_timesteps)**p
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


# ================
# Helper function
# ================

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.float):
        array = torch.tensor([array])

    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target).to(target.device)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))


def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img