import torch
import numpy as np
import matplotlib.pyplot as plt

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


class FIRE:
    def __init__(self, model, alphas_cumprod, x_T, H, sqrt_in_var_to_out, clamp_denoise):
        self.model = model
        self.alphas_cumprod = alphas_cumprod
        self.power = 0.5
        self.H = H
        self.v_min = ((1 - self.alphas_cumprod) / self.alphas_cumprod)[0]

        self.singular_match = H.s_max

        self.tolerance = 1e-5
        self.gam_w_correct = 1e4
        self.max_cg_iters = 100
        self.cg_initialization = torch.zeros_like(x_T)

        self.rho = 1.25
        self.max_iters = 50

        self.clamp_denoise = clamp_denoise

        with open(sqrt_in_var_to_out, 'rb') as f:
            self.sqrt_in_variance_to_out = torch.from_numpy(np.load(f)).to(x_T.device)

    def uncond_denoiser_function(self, noisy_im, noise_var):
        diff = torch.abs(
            noise_var[:, 0, None] - (1 - torch.tensor(self.alphas_cumprod).to(noisy_im.device)) / torch.tensor(
                self.alphas_cumprod).to(noisy_im.device))
        t = torch.argmin(diff, dim=1)

        ones = torch.ones(noise_var.shape, device=noise_var.device)

        delta = torch.minimum(noise_var / self.v_min, ones)
        noise_var_clip = torch.maximum(noise_var, ones * self.v_min)

        scaled_noisy_im = noisy_im * torch.sqrt(1 / (1 + noise_var_clip[:, 0, None, None, None]))

        noise_predict = self.model(scaled_noisy_im, t)

        if noise_predict.shape[1] == 2 * noisy_im.shape[1]:
            noise_predict, _ = torch.split(noise_predict, noisy_im.shape[1], dim=1)

        noise_est = torch.sqrt(noise_var_clip)[:, 0, None, None, None] * noise_predict
        x_0 = (1 - delta ** self.power)[:, 0, None, None, None] * noisy_im + (delta ** self.power)[:, 0, None, None,
                                                                             None] * (noisy_im - noise_est)

        return x_0, ((1 - torch.tensor(self.alphas_cumprod).to(noisy_im.device)) / torch.tensor(
            self.alphas_cumprod).to(noisy_im.device))[t], t

    def CG(self, A, scaled_mu_2, y, gamma_w, eta):
        # solve Abar'Abar x = Abar' y

        x = self.cg_initialization.clone()

        b = gamma_w[:, 0, None, None, None] * self.H.Ht(y).view(scaled_mu_2.shape[0], scaled_mu_2.shape[1], scaled_mu_2.shape[2], scaled_mu_2.shape[3])
        b = b + scaled_mu_2

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

            if torch.mean(diff) <= self.tolerance:
                break

            beta = torch.sum(r ** 2, dim=(1, 2, 3)) / rsold

            p = r + beta[:, None, None, None] * p
            num_cg_steps += 1
            # if torch.isnan(x).any():
            #     print(num_cg_steps)
            #     exit()

        # print(num_cg_steps)
        # print(diff)
        return x.clone()

    def linear_estimation(self, scaled_mu_2, y, gamma_w, eta):
        # eta_np = eta
        gamma_w_hat = gamma_w * torch.ones(scaled_mu_2.shape[0], 1).to(scaled_mu_2.device)
        gamma_w_hat[gamma_w / eta > self.gam_w_correct] = self.gam_w_correct * eta[gamma_w / eta > self.gam_w_correct]

        CG_A = lambda vec: gamma_w_hat[:, 0, None, None, None] * self.H.Ht(self.H.H(vec)).view(scaled_mu_2.shape[0], scaled_mu_2.shape[1], scaled_mu_2.shape[2],
                                                                           scaled_mu_2.shape[3]) + eta[:, 0, None, None, None] * vec

        mu_1 = self.CG(CG_A, scaled_mu_2, y, gamma_w_hat, eta)

        return mu_1

    def denoising(self, mu_1, gamma_2):
        # Max var
        noise_var, _ = torch.max(1 / gamma_2, dim=1, keepdim=True)

        # Denoise
        mu_2, true_noise_var, used_t = self.uncond_denoiser_function(mu_1.float(), noise_var)
        if self.clamp_denoise:
            mu_2 = mu_2.clamp(min=-1, max=1)

        eta_2 = 1 / (self.sqrt_in_variance_to_out[used_t[0]] * true_noise_var[0].unsqueeze(0).sqrt().repeat(mu_2.shape[0], 1)).float()

        return mu_2, eta_2

    def renoising(self, mu_1, eta, gamma_r, gamma_w):
        gamma_r = self.rho * gamma_r
        # eta_np = eta
        gamma_w_hat = gamma_w * torch.ones(mu_1.shape[0], 1).to(mu_1.device)
        gamma_w_hat[gamma_w / eta > self.gam_w_correct] = self.gam_w_correct * eta[gamma_w / eta > self.gam_w_correct]

        max_prec = self.alphas_cumprod[0] / (1 - self.alphas_cumprod[0])
        gamma_r = torch.minimum(gamma_r, torch.ones(gamma_r.shape).to(gamma_r.device) * max_prec)

        eps_1 = torch.randn_like(mu_1)
        transformed_eps_2 = self.H.Ht(torch.randn_like(self.H.H(mu_1))).view(mu_1.shape[0], mu_1.shape[1], mu_1.shape[2], mu_1.shape[3])

        eps_1_scale_squared = torch.max(1 / gamma_r[0, 0] - 1 / (eta), torch.zeros_like(1 / gamma_r[0, 0] - 1 / (eta)))
        eps_2_scale_squared = (1 / eta - ((gamma_w_hat ** 2) * self.singular_match ** 2 / gamma_w + eta) / ((gamma_w_hat * self.singular_match ** 2 + eta) ** 2)) / (self.singular_match ** 2)
        eps_2_scale_squared = torch.max(eps_2_scale_squared, torch.zeros_like(eps_2_scale_squared))

        noise_approx = eps_1_scale_squared.sqrt()[:, 0, None, None, None] * eps_1
        noise_approx = noise_approx + eps_2_scale_squared.sqrt()[:, 0, None, None, None] * transformed_eps_2

        mu_1_noised = mu_1.clone() + noise_approx

        return mu_1_noised, gamma_r

    def run_fire(self, x_t, y, t_alpha_bar, noise_sig):
        # 0. Initialize Values
        gamma_r = torch.tensor([t_alpha_bar / (1 - t_alpha_bar)]).unsqueeze(0).repeat(x_t.shape[0], 1).to(
            x_t.device)
        gamma_w = 1 / (noise_sig ** 2)
        mu_1 = x_t / torch.sqrt(t_alpha_bar)
        mu_1_noised = mu_1.clone()

        for i in range(self.max_iters):
            # 1. Denoising
            mu_2, eta = self.denoising(mu_1_noised, gamma_r)
            mu_2 = mu_2 + torch.randn_like(mu_2) / (eta[:, 0, None, None, None]).sqrt()

            tr_approx = 0.
            num_samps = 50
            m = 0
            for k in range(num_samps):
                out = self.H.H(torch.randn_like(mu_1))
                m = out.shape[1]
                tr_approx += torch.sum(out ** 2, dim=1).unsqueeze(1)

            tr_approx = tr_approx / num_samps
            y_m_A_mu_2 = torch.sum((y - self.H.H(mu_2)) ** 2, dim=1).unsqueeze(1)
            eta = tr_approx / (y_m_A_mu_2 - m / gamma_w)
            eta = eta.float()

            # 2. Linear Estimation
            mu_1 = self.linear_estimation(mu_2 * eta[:, 0, None, None, None], y, gamma_w, eta)

            # 3. Re-Noising
            mu_1_noised, gamma_r = self.renoising(mu_1, eta, gamma_r, gamma_w)

            self.cg_initialization = mu_1.clone() # CG warm start

        return mu_1.float()


def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)
