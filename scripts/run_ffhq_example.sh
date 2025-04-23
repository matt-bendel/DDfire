python main.py \
--model_config=configs/ffhq_model_config.yaml \
--diffusion_config=configs/ffhq_diffusion_config.yaml \
--data_config=configs/ffhq_config.yaml \
--problem_config=configs/problems/ffhq/blur_gauss_config.yaml \
--fire_config=configs/fire_config_imagenet.yaml
--sig_y=0.05 --nfes=1000 --gpus=1

python -m pytorch_fid /GT/PATH /SAMPLE/PATH # TODO: Enter appropriate path..

python evaluate_samples.py \
--data_config=configs/ffhq_config.yaml \
--problem_config=configs/problems/ffhq/blur_gauss_config.yaml \
--nfes=1000 --num_ims=1000