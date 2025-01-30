from functools import partial
import argparse
import yaml
import types
import torch
import matplotlib.pyplot as plt
import numpy as np

from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler

from util.logger import get_logger
from data.ImageDataModule import ImageDataModule

from pytorch_lightning import seed_everything

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
    data_config = load_yaml(args.data_config)

    diffusion_config["timestep_respacing"] = '1000'
    diffusion_config["sampler"] = 'ddpm'

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)

    dm = ImageDataModule(data_config)

    dm.setup()
    val_loader = dm.val_dataloader()

    scale_factors = [[] for i in range(1000)]

    for i, data in enumerate(val_loader):
        logger.info(f"Inference for image {i}")
        # Different motion blur operator for each batch
        x = data[0]
        x = x.to(device)

        # Compute Eta...
        with torch.no_grad():
            for j in range(1000):
                t = torch.tensor([j] * x.shape[0]).to(x.device)
                x_start = sampler.q_sample(x, j)

                x_0_hat = sampler.p_mean_variance(model, x_start, t)['pred_xstart']

                abar_t = extract_and_expand(sampler.alphas_cumprod, t, x_start)[0, 0, 0, 0]

                input_variance = (1 - abar_t) / abar_t
                output_variance = torch.mean((x_0_hat - x) ** 2)

                scale_factor = output_variance / input_variance.sqrt()
                scale_factors[j].append(scale_factor.cpu().numpy())

    scale_factors = np.array(scale_factors)
    scale_factors = np.mean(scale_factors, axis=1)

    with open(data_config["sqrt_in_var_to_out"], 'wb') as f:
        np.save(f, scale_factors)


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


if __name__ == '__main__':
    main()
