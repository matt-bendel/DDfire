from functools import partial
import argparse
import yaml
import types
import lpips
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import piq

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from PIL import Image


# this psnr function picked from fastmri utils

def compute_psnr(target, img2):
    target = np.clip(target, 0, 1)
    img2 = np.clip(img2, 0, 1)
    mse = np.mean((target - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def compute_psnr_altnorm(target, img2):
    target = np.clip(target, -1, 1)
    img2 = np.clip(img2, -1, 1)
    mse = np.mean((target - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(2.0 / np.sqrt(mse))

def compute_psnr_altnorm_2(target, img2):
    target = np.clip(target, -1, 1)
    img2 = np.clip(img2, -1, 1)
    psnr_base_e = 2. * np.log(2.) - np.log(np.mean((target - img2) ** 2))
    return psnr_base_e * (10. / np.log(10.))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--problem_config', type=str)
    parser.add_argument('--noiseless', action='store_true')
    parser.add_argument('--nfes', type=int, default=100)
    parser.add_argument('--num_ims', type=int, default=1000)
    args = parser.parse_args()

    # Device setting
    device_str = f"cuda:0" if torch.cuda.is_available() else 'cpu'
    print(f"Using device {device_str}")
    device = torch.device(device_str)

    data_config = load_yaml(args.data_config)
    problem_config = load_yaml(args.problem_config)

    lpips = piq.LPIPS(replace_pooling=True, reduction='none')
    total_num_of_images = args.num_ims

    print(" ")
    print("Finding performance metrics for algorithm: DDRM")
    print(" ")
    output_folder = data_config["fire_out"] + f'/{args.nfes}/{problem_config["deg"]}{"" if args.noiseless else "_noisy"}/' + 'samples/'
    gt_folder = data_config["fire_out"] + f'/{args.nfes}/{problem_config["deg"]}{"" if args.noiseless else "_noisy"}/' + 'x/'

    PSNR_all_imgs = torch.zeros(total_num_of_images, 1)
    LPIPS_all_imgs = torch.zeros(total_num_of_images, 1)

    for test_image_number in range(total_num_of_images):
        file_name1 = f"x_{test_image_number}.png"
        file_name2 = f"image_{test_image_number}.png"

        gt_img_path = os.path.join(gt_folder, file_name1)
        output_img_path = os.path.join(output_folder, file_name2)

        # Read images
        gt_image_0_255 = cv2.cvtColor(cv2.imread(gt_img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        recon_img_0_255 = cv2.cvtColor(cv2.imread(output_img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        gt_image = gt_image_0_255 / 255
        recon_img = recon_img_0_255 / 255

        PSNR_all_imgs[test_image_number, 0] = compute_psnr(gt_image, recon_img)

        x_0t1 = torch.from_numpy(gt_image_0_255 / 255.).permute(2, 0, 1).unsqueeze(
            0).contiguous().float().to(device)
        sample_0t1 = torch.from_numpy(recon_img_0_255 / 255.).permute(2, 0, 1).unsqueeze(
            0).contiguous().float().to(device)

        LPIPS_all_imgs[test_image_number, 0] = lpips(sample_0t1, x_0t1).mean().detach().cpu()

    # Average PSNR and LPIPS
    print("=====================================")
    PSNR_avg_np_round = np.round(PSNR_all_imgs[:-1].mean().numpy(), 2)
    LPIPS_avg_np_round = np.round(LPIPS_all_imgs[:-1].mean().numpy(), 4)

    # print(" ")
    # PSNR_all_imgs_mean = np.round(torch.mean(PSNR_all_imgs[:,0]).numpy(),2)
    print("Average PSNR  :", PSNR_avg_np_round)
    print("Average LPIPS :", LPIPS_avg_np_round)
    # print(" ")
    print("=====================================")


if __name__ == '__main__':
    main()
