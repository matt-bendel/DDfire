import yaml
import numpy as np
import torch
from tqdm.auto import tqdm
from guided_diffusion.fire import FIRE


class DDfire:
    def __init__(self, fire_config, ref_tensor, model, model_betas, A, K, delta, eta_ddim=1.0, N_tot=1000,
                 quantize_ddim=False):
        self.K = K
        self.delta = delta
        self.N_tot = N_tot
        self.quantize_ddim = quantize_ddim
        self.eta = eta_ddim

        self.A = A
        self.model = model
        self.model_alphas = 1 - model_betas
        self.model_alpha_bars = np.cumprod(self.model_alphas)
        self.model_vars = (1 - self.model_alpha_bars) / self.model_alpha_bars

        vp_prec = 1 / self.model_vars  # Precision is inverse variance TODO: Update to variance..
        vp_prec_log = np.log10(vp_prec)

        vp_ddim_prec_log = np.linspace(np.max(vp_prec_log), np.min(vp_prec_log), self.K)
        vp_ddim_prec = 10 ** vp_ddim_prec_log

        all_steps = []
        for i in range(K):
            t = np.argmin(np.abs(vp_prec - vp_ddim_prec[i]), axis=0)
            all_steps.append(t)

        t_list = list(set(all_steps))
        t_list.sort()
        if quantize_ddim:
            vp_ddim_prec = vp_prec[np.array(t_list)]
            self.K = len(vp_ddim_prec)

        self.N_k = []

        # set gam_tgt using delta, the fraction of 1-iter steps
        self.K_ddim1 = 1 + int(self.delta * (self.K - 1))  # number of 1-iter steps, in 1,...,K_ddim-1

        iters_ire_, rho = self.run_bisection_search(vp_ddim_prec)

        self.rho = rho
        self.N_k = iters_ire_.tolist()
        print(np.sum(self.N_k))

        self.vp_ddim_prec = vp_ddim_prec
        self.alpha_bars = 1 / (1 + 1 / self.vp_ddim_prec)
        self.fire_runner = FIRE(ref_tensor, vp_prec, model, A, rho, 1 / vp_prec[0], fire_config)
        self.edm_sample = fire_config['use_edm']

    def run_bisection_search(self, vp_ddim_prec):
        # set gam_tgt using delta, the fraction of 1-iter steps
        log_gam_tgt = np.log10(vp_ddim_prec[self.K_ddim1 - 1])  # log-precision of first 1-iter step

        # determine rho and FIRE waterfilling schedule to determine rho
        log_gam_ddim_ = np.log10(vp_ddim_prec)
        gam_ddim_ = 10 ** log_gam_ddim_
        log_rho_min = 1e-5  # initialize bisection
        log_rho_max = 1e6  # initialize bisection
        quant = lambda inp: np.ceil(inp - 1e-5)  # small correction due to imperfect bisection
        NFE_min = np.sum(quant(np.maximum((log_gam_tgt - log_gam_ddim_) / log_rho_min, 0) + 1))
        NFE_max = np.sum(quant(np.maximum((log_gam_tgt - log_gam_ddim_) / log_rho_max, 0) + 1))
        NFE_tgt = self.N_tot
        max_bisection_iters = 80

        if NFE_max > NFE_tgt:
            raise ValueError('NFE_tgt must be > ' + str(NFE_max))

        # bisection
        it = 1
        while it < max_bisection_iters:
            it = it + 1
            log_rho_mid = 0.5 * (log_rho_min + log_rho_max)
            NFE_mid = np.sum(quant(np.maximum((log_gam_tgt - log_gam_ddim_) / log_rho_mid, 0) + 1))

            if NFE_mid > NFE_tgt:
                log_rho_min = log_rho_mid
            elif NFE_mid < NFE_tgt:
                log_rho_max = log_rho_mid
            else:
                log_rho_max = log_rho_mid  # okay, but try to improve...

        iters_ire_ = quant(np.maximum((log_gam_tgt - log_gam_ddim_) / log_rho_mid, 0) + 1).astype(int)

        rho = 10 ** log_rho_mid

        return iters_ire_, rho

    def p_sample_loop(self, x_start, y, sig_y=0.001):
        """
        The function used for sampling from noise.
        """
        self.fire_runner.reset()

        x_t = x_start
        sig_prev = None

        pbar = tqdm(list(range(self.K))[::-1])
        for k in pbar:
            fire_iters = int(self.N_k[k])
            fire_prec = self.vp_ddim_prec[k]
            if self.edm_sample:
                # Abuse of variable, eta becomes DDIM gamma here...
                fire_var = 1 / fire_prec
                fire_sig = np.sqrt(fire_var)
                fire_sig_hat = (1 + self.eta) * fire_sig
                fire_prec = 1 / (fire_sig_hat ** 2)

            E_x_0_g_x_t_y = self.fire_runner.run_fire(x_t, y, sig_y, fire_prec, fire_iters, first_k=k == self.K - 1,
                                                      ve_init=self.edm_sample, quantized_t=self.quantize_ddim)
            if self.edm_sample:
                x_t = self.edm_update(x_t * fire_sig_hat if k == self.K - 1 else x_t, E_x_0_g_x_t_y, k)
            else:
                x_t = self.ddim_update(x_t, E_x_0_g_x_t_y, k)

        return x_t.clamp(min=-1., max=1.)

    def edm_update(self, x_t, E_x_0_g_x_t_y, k):
        alpha_bar = self.alpha_bars[k]
        alpha_bar_prev = 1. if k - 1 < 0 else self.alpha_bars[k - 1]

        sig = np.sqrt((1 - alpha_bar) / alpha_bar)
        sig_hat = (1 + self.eta) * sig
        sig_prev = np.sqrt((1 - alpha_bar_prev) / alpha_bar_prev)
        sig_hat_prev = (1 + self.eta) * sig_prev

        new_noise_sig = np.sqrt(sig_hat_prev ** 2 - sig_prev ** 2)
        noise = torch.randn_like(x_t)

        x_t = sig_prev * x_t / sig_hat
        x_t += (1 - sig_prev / sig_hat) * E_x_0_g_x_t_y

        if k != 0:
            x_t += new_noise_sig * noise

        return x_t

    def ddim_update(self, x_t, E_x_0_g_x_t_y, k):
        alpha_bar = self.alpha_bars[k]
        alpha_bar_prev = 1. if k - 1 < 0 else self.alpha_bars[k - 1]
        sigma = (
                self.eta
                * np.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * np.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        c = np.sqrt((1 - alpha_bar_prev - sigma ** 2) / (1 - alpha_bar))

        # Equation 12.
        noise = torch.randn_like(x_t)

        x_t = c * x_t + (np.sqrt(alpha_bar_prev) - c * np.sqrt(alpha_bar)) * E_x_0_g_x_t_y
        if k != 0:
            x_t += sigma * noise

        return x_t
