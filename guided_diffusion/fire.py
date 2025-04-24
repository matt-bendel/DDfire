import torch
import numpy as np

from numpy.polynomial import Polynomial


class FIRE:
    def __init__(self, ref_tensor, gamma_model, model, A, rho, v_min, fire_config):
        self.A = A
        self.v_min = v_min
        self.rho = rho
        self.singular_match = A.s_star
        self.gamma_model = gamma_model
        self.var_model = 1 / gamma_model
        self.model = model

        # Ablation params
        self.use_stochastic_denoising = fire_config['use_stochastic_denoising']
        self.use_colored_noise = fire_config['use_colored_noise']
        self.estimate_nu = fire_config['estimate_nu']

        # CG params
        self.gam_w_correct = float(fire_config['gam_w_correct'])
        self.max_cg_iters = int(fire_config['max_cg_iters'])
        self.cg_tolerance = float(fire_config['cg_tolerance'])

        self.cg_initialization = torch.zeros_like(ref_tensor)

        with open(fire_config['nu_lookup'], 'rb') as f:
            self.scale_factor = np.load(f)

        # Quantized nu
        nu = np.sqrt(self.var_model) * self.scale_factor

        # find first t where sequence decreases
        first = np.argmin(np.diff(nu) >= 0)

        # start polynomial fit a bit earlier
        earlier = 100
        t_nofit = np.arange(first - earlier)
        t_fit = np.arange(first - earlier, 1000)

        # try polynomial fit in log domain
        logPoly_ffhq = Polynomial.fit(t_fit, np.log(nu[t_fit]), deg=10)
        self.nu_predict = lambda t: np.exp(logPoly_ffhq(t))

    def uncond_denoiser_function(self, noisy_im, noise_var, quantized_t):
        delta = np.minimum(noise_var / self.v_min, 1.)
        noise_var_clip = np.maximum(noise_var, self.v_min)

        if quantized_t:
            alpha_bars_model = 1 / (1 + 1 / self.gamma_model)
            diff = torch.abs(
                noise_var - (1 - torch.tensor(alpha_bars_model).to(noisy_im.device)) / torch.tensor(
                    alpha_bars_model).to(noisy_im.device))
            t = torch.argmin(diff).repeat(noisy_im.shape[0])
        else:
            t = torch.tensor(self.get_t_from_var(noise_var)).unsqueeze(0).repeat(noisy_im.shape[0]).to(noisy_im.device) # Need timestep for denoiser input

        alpha_bar = 1 / (1 + noise_var_clip)
        scaled_noisy_im = noisy_im * np.sqrt(alpha_bar)

        noise_predict = self.model(scaled_noisy_im, t)

        if noise_predict.shape[1] == 2 * noisy_im.shape[1]:
            noise_predict, _ = torch.split(noise_predict, noisy_im.shape[1], dim=1)

        noise_est = np.sqrt(noise_var_clip) * noise_predict

        x_0 = (1 - delta ** 0.5) * noisy_im + (delta ** 0.5) * (noisy_im - noise_est)

        return x_0, noise_var_clip, t

    def denoising(self, r, gamma, quantized_t):
        # Max var
        noise_var = 1 / gamma

        # Denoise
        x_bar, noise_var, t = self.uncond_denoiser_function(r.float(), noise_var, quantized_t)
        x_bar = x_bar.clamp(min=-1, max=1)

        if quantized_t:
            lookup_t = np.argmin(np.abs(self.gamma_model - gamma))
            one_over_nu = 1 / (self.scale_factor[lookup_t] * np.sqrt(noise_var))
        else:
            one_over_nu = 1 / self.nu_predict(t[0].cpu().numpy())

        return x_bar, torch.tensor(one_over_nu).unsqueeze(0).unsqueeze(0).repeat(x_bar.shape[0], 1).to(x_bar.device).float()

    def CG(self, A, scaled_x_bar, y, gamma_y):
        # solve Abar'Abar x = Abar' y

        x = self.cg_initialization.clone()
        x_store = torch.zeros_like(x)

        saved_x = [False for i in range(x_store.shape[0])]

        b = gamma_y[:, 0, None, None, None] * self.A.Ht(y).view(*scaled_x_bar.shape)
        b = b + scaled_x_bar

        b_norm = torch.sum(b ** 2, dim=(1, 2, 3))

        r = b - A(x)
        p = r.clone()

        num_cg_steps = 0
        while num_cg_steps < self.max_cg_iters:
            Ap = A(p)
            rsold = torch.sum(r ** 2, dim=(1, 2, 3))

            alpha = rsold / torch.sum(p * Ap, dim=(1, 2, 3))

            x = x + alpha[:, None, None, None] * p
            r = r - alpha[:, None, None, None] * Ap

            diff = (torch.sum(r ** 2, dim=(1, 2, 3)) / b_norm).sqrt()

            all_saved = True
            for i in range(diff.shape[0]):
                if diff[i] <= self.cg_tolerance and not saved_x[i]:
                    x_store[i] = x[i]
                    saved_x[i] = True

                if not saved_x[i]:
                    all_saved = False

            if all_saved:
                break

            beta = torch.sum(r ** 2, dim=(1, 2, 3)) / rsold

            p = r + beta[:, None, None, None] * p
            num_cg_steps += 1

        for i in range(x.shape[0]):
            if not saved_x[i]:
                x_store[i] = x[i]

        return x_store.clone()

    def linear_estimation(self, scaled_x_bar, y, gamma_y, one_over_nu):
        gamma_y_hat = gamma_y * torch.ones(scaled_x_bar.shape[0], 1).to(scaled_x_bar.device)
        gamma_y_hat[gamma_y / one_over_nu > self.gam_w_correct] = self.gam_w_correct * one_over_nu[
            gamma_y / one_over_nu > self.gam_w_correct]

        CG_A = lambda vec: gamma_y_hat[:, 0, None, None, None] * self.A.Ht(self.A.H(vec)).view(*scaled_x_bar.shape) + one_over_nu[:, 0, None, None, None] * vec

        x_hat = self.CG(CG_A, scaled_x_bar, y, gamma_y_hat)

        return x_hat

    def renoising(self, x_hat, one_over_nu, gamma, gamma_y):
        gamma = self.rho * gamma
        max_gamma = 1 / self.v_min
        gamma = gamma if gamma < max_gamma else max_gamma

        gamma_y_hat = gamma_y * torch.ones(x_hat.shape[0], 1).to(x_hat.device)
        gamma_y_hat[gamma_y / one_over_nu > self.gam_w_correct] = self.gam_w_correct * one_over_nu[
            gamma_y / one_over_nu > self.gam_w_correct]

        eps_1 = torch.randn_like(x_hat)
        transformed_eps_2 = self.A.Ht(torch.randn_like(self.A.H(x_hat))).view(*x_hat.shape)

        eps_1_scale_squared = torch.max(1 / gamma - 1 / one_over_nu, torch.zeros_like(1 / gamma - 1 / one_over_nu))
        eps_2_scale_squared = (1 / one_over_nu - ((gamma_y_hat ** 2) * self.singular_match ** 2 / gamma_y + one_over_nu) / (
                    (gamma_y_hat * self.singular_match ** 2 + one_over_nu) ** 2)) / (self.singular_match ** 2)
        eps_2_scale_squared = torch.max(eps_2_scale_squared, torch.zeros_like(eps_2_scale_squared))

        noise = eps_1_scale_squared.sqrt()[:, 0, None, None, None] * eps_1
        noise = noise + eps_2_scale_squared.sqrt()[:, 0, None, None, None] * transformed_eps_2

        r = x_hat + noise

        return r, gamma

    def get_t_from_var(self, noise_var):
        return np.minimum(999 * (np.sqrt(0.1 ** 2 + 2 * 19.9 * np.log(1 + noise_var)) - 0.1) / 19.9, 999)

    def reset(self):
        self.cg_initialization = torch.zeros_like(self.cg_initialization)

    def run_fire(self, x_t, y, sig_y, gamma_init, fire_iters, first_k=False, ve_init=False, quantized_t=False):
        # 0. Initialize Values
        gamma = gamma_init
        gamma_y = 1 / (sig_y ** 2)

        alpha_bar = 1 / (1 + 1 / gamma)

        if ve_init:
            x_hat = x_t / np.sqrt(gamma_init) if first_k else x_t
        else:
            x_hat = x_t / np.sqrt(alpha_bar)

        r = x_hat.clone()

        for i in range(fire_iters):
            # 1. Denoising
            x_bar, one_over_nu = self.denoising(r, gamma, quantized_t)
            # if self.use_stochastic_denoising:
            #     x_bar = x_bar + torch.randn_like(x_bar) / np.sqrt(one_over_nu)
            #     one_over_nu = torch.tensor(one_over_nu / 2).unsqueeze(0).repeat(x_bar.shape[0]).unsqueeze(1).to(x_bar.device)

            if self.estimate_nu:
                tr_approx = 0.
                num_samps = 50
                m = 0
                for k in range(num_samps):
                    out = self.A.H(torch.randn_like(x_hat))
                    m = out.shape[1]
                    tr_approx += torch.sum(out ** 2, dim=1).unsqueeze(1)

                tr_approx = tr_approx / num_samps
                y_m_A_x_bar = torch.sum((y - self.A.H(x_bar)) ** 2, dim=1).unsqueeze(1)
                one_over_nu = tr_approx / (y_m_A_x_bar - m / gamma_y)

            # 2. Linear Estimation
            x_hat = self.linear_estimation(x_bar * one_over_nu[:, 0, None, None, None], y, gamma_y, one_over_nu)

            # 3. Re-Noising
            r, gamma = self.renoising(x_hat, one_over_nu, gamma, gamma_y)

            self.cg_initialization = x_hat.clone()

        return x_hat.clamp(min=-1., max=1.).float()

