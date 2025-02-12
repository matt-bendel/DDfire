from functools import partial
import argparse
import yaml
import types
import lpips
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from guided_diffusion.unet import create_model
from guided_diffusion.fire_ddim import create_sampler

from util.img_utils import clear_color
from util.logger import get_logger
from data.ImageDataModule import ImageDataModule

from pytorch_lightning import seed_everything
from guided_diffusion.ddrm_svd import get_operator, Deblurring
from torchmetrics.functional import peak_signal_noise_ratio

def load_object(dct):
    return types.SimpleNamespace(**dct)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--problem_config', type=str)
    parser.add_argument('--noiseless', action='store_true')
    parser.add_argument('--clamp-denoise', action='store_true')
    parser.add_argument('--clamp-fire-ddim', action='store_true')
    parser.add_argument('--sig_y', type=float, default=0.05)
    parser.add_argument('--nfes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)

    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    problem_config = load_yaml(args.problem_config)
    data_config = load_yaml(args.data_config)

    diffusion_config["timestep_respacing"] = f'ddim{args.nfes}'
    diffusion_config["timestep_respacing"] = [f'ddim{args.nfes}', f'{problem_config["noiseless" if args.noiseless else "noisy"][args.nfes]["K"]}']

    if args.nfes < 100:
        diffusion_config["eta"] = 0.5

    sig_y = float(args.sig_y)
    if args.noiseless:
        sig_y = 0.001  # Set precision for DDfire

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config, delta_1=problem_config["noiseless" if args.noiseless else "noisy"][args.nfes]["delta_1"], nfe_budget=args.nfes, clamp_denoise=args.clamp_denoise, clamp_fire_ddim=args.clamp_fire_ddim)
    sample_fn = partial(sampler.p_sample_loop, model=model)

    dm = ImageDataModule(data_config)

    dm.setup()
    test_loader = dm.test_dataloader()

    H = get_operator(problem_config, data_config, device)

    os.makedirs(data_config["fire_out"] + f'/{args.nfes}/{problem_config["deg"]}{"" if args.noiseless else "_noisy"}/' + 'samples', exist_ok=True)

    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    lpips_vals = []
    psnr_vals = []

    for i, data in enumerate(test_loader):
        logger.info(f"Inference for image {i}")
        # Different motion blur operator for each batch
        if problem_config["deg"] == 'blur_motion':
            H = get_operator(problem_config, data_config, device)

        x = data[0]
        x = x.to(device)

        y_n = H.H(x)
        if not args.noiseless:
            y_n = y_n + torch.randn_like(y_n) * sig_y

        # Sampling
        with torch.no_grad():
            x_start = torch.randn(x.shape, device=device)
            sample, nfes = sample_fn(x_start=x_start, measurement=y_n, noise_sig=sig_y, eta=diffusion_config["eta"], H=H, sqrt_in_var_to_out=data_config["sqrt_in_var_to_out"])

            lpips_vals.append(loss_fn_vgg(sample, x).mean().detach().cpu().numpy())
            psnr_vals.append(peak_signal_noise_ratio(sample, x).mean().detach().cpu().numpy())

            for j in range(sample.shape[0]):
                plt.imsave(f'{data_config["fire_out"]}/{args.nfes}/{problem_config["deg"]}{"" if args.noiseless else "_noisy"}/samples/image_{i * data_config["batch_size"] + j}.png', clear_color(sample[j].unsqueeze(0)))


    print(f'Avg. LPIPS: {np.mean(lpips_vals)} +/- {np.std(lpips_vals) / len(lpips_vals)}')
    print(f'Avg. PSNR: {np.mean(psnr_vals)} +/- {np.std(psnr_vals) / len(psnr_vals)}')


if __name__ == '__main__':
    main()
