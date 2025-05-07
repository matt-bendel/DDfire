from functools import partial
import argparse
import yaml
import types
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from guided_diffusion.unet import create_model
from guided_diffusion.DDfire import DDfire

from util.img_utils import clear_color
from util.logger import get_logger
from data.ImageDataModule import ImageDataModule

from pytorch_lightning import seed_everything
from guided_diffusion.ddrm_svd import get_operator
from torch.multiprocessing import Process
import torch.multiprocessing as mp

def all_gather(tensor, log=None):
    if log: log.info("Gathering tensor across {} devices... ".format(dist.get_world_size()))
    gathered_tensors = [
        torch.zeros_like(tensor) for _ in range(dist.get_world_size())
    ]
    with torch.no_grad():
        dist.all_gather(gathered_tensors, tensor)
    return gathered_tensors


def collect_all_subset(psnr, lpips):
    gathered_psnr = all_gather(psnr)
    gathered_lpips = all_gather(lpips)

    return gathered_psnr, gathered_lpips


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = f'{args.port}'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


def load_object(dct):
    return types.SimpleNamespace(**dct)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main(args):
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
    fire_config = load_yaml(args.fire_config)

    if args.nfes < 100:
        diffusion_config["eta"] = 0.5

    sig_y = float(args.sig_y)
    if args.noiseless:
        sig_y = 0.001  # Set precision for DDfire

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    beta_start = 0.0001
    beta_end = 0.02
    model_betas = np.linspace(
        beta_start, beta_end, 1000, dtype=np.float64
    )

    A = get_operator(problem_config, data_config, device)

    # Load diffusion sampler
    sampler = DDfire(fire_config, torch.ones(data_config["batch_size"], 3, 256, 256).to(device), model, model_betas, A, problem_config['K'],
           problem_config['delta'], problem_config['eta'], N_tot=args.nfes, quantize_ddim=False)

    dm = ImageDataModule(data_config)

    dm.setup()
    test_loader = dm.test_dataloader()

    os.makedirs(data_config["fire_out"] + f'/{args.nfes}/{problem_config["deg"]}{"" if args.noiseless else "_noisy"}/' + 'samples', exist_ok=True)
    os.makedirs(data_config["fire_out"] + f'/{args.nfes}/{problem_config["deg"]}{"" if args.noiseless else "_noisy"}/' + 'x', exist_ok=True)

    for i, data in enumerate(test_loader):
        logger.info(f"Inference for image {i}")
        # Different motion blur operator for each batch - in paper we use fixed blur kernel
        if problem_config["deg"] == 'blur_motion':
            H = get_operator(problem_config, data_config, device)

        if i % args.gpus != args.local_rank and args.gpus > 1:
            continue

        x = data[0]
        x = x.to(device)

        y_n = A.H(x)
        if not args.noiseless:
            y_n = y_n + torch.randn_like(y_n) * sig_y

        # Sampling
        with torch.no_grad():
            x_start = torch.randn(x.shape, device=device)
            sample = sampler.p_sample_loop(x_start, y_n, sig_y)
            dist.barrier()

            for j in range(sample.shape[0]):
                plt.imsave(f'{data_config["fire_out"]}/{args.nfes}/{problem_config["deg"]}{"" if args.noiseless else "_noisy"}/samples/image_{i * data_config["batch_size"] + j}.png', clear_color(sample[j].unsqueeze(0)))
                plt.imsave(f'{data_config["fire_out"]}/{args.nfes}/{problem_config["deg"]}{"" if args.noiseless else "_noisy"}/x/x_{i * data_config["batch_size"] + j}.png', clear_color(x[j].unsqueeze(0)))


if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--problem_config', type=str)
    parser.add_argument('--fire_config', type=str)
    parser.add_argument('--noiseless', action='store_true')
    parser.add_argument('--sig_y', type=float, default=0.05)
    parser.add_argument('--nfes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--master_address', type=str, default='localhost')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument('--global_size', type=int, default=1)
    parser.add_argument('--port', type=int, default=6010)
    args = parser.parse_args()

    if args.gpus > 1:
        size = args.gpus

        processes = []
        for rank in range(size):
            args = copy.deepcopy(args)
            args.local_rank = rank
            global_rank = rank  # single node assumed
            global_size = size
            args.global_rank = rank
            args.global_size = size
            print('local proc %d, global proc %d, global_size %d, port %d' % (rank, global_rank, global_size, args.port))
            p = Process(target=init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        args.global_rank = 0
        args.local_rank = 0
        args.global_size = 1
        init_processes(0, 1, main, args)
