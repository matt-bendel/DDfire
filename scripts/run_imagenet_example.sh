python main.py \
--model_config=configs/imagenet_model_config.yaml \
--diffusion_config=configs/imagenet_diffusion_config.yaml \
--data_config=configs/imagenet_config.yaml \
--problem_config=configs/problems/imagenet/blur_gauss_config.yaml \
--sig_y=0.05 --nfes=100 --clamp-denoise --clamp-fire-ddim