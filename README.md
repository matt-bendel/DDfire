# Solving Inverse Problems using Diffusion with Iterative Colored Renoising [[arXiv]](https://arxiv.org/pdf/2501.17468) [[OpenReview]](https://openreview.net/forum?id=RZv8FcQDPW)

Official PyTorch implementation of Solving Inverse Problems using Diffusion with Fast Iterative Renoising. Code modified from DPS.
Our code for Phase Retrieval can be found [here](https://github.com/Saurav-K-Shastri/DDfire-PR).

by Matthew Bendel, Saurav Shastri, Rizwan Ahmad, and Philip Schniter.

## Getting Started
### 1) Clone the repository
```
git clone https://github.com/matt-bendel/DDfire
cd DDfire
```

### 2) Download the Pretrained Checkpoints
Download the [FFHQ](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing) and [ImageNet](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) pretrained models.
Update `ffhq_model_config.yaml` and `imagenet_model_config.yaml` with the appropriate paths after downloading.

### 3) Download the Datasets
Download the [FFHQ](https://www.kaggle.com/datasets/rahulbhalley/ffhq-256x256) dataset and the [ImageNet validation set](https://www.image-net.org/download.php).
To work with the dataloader we use, store the images in a subfolder called '0'. I.e., the path to your data should be something like
```
/storage/FFHQ/0/00000.png
/storage/FFHQ/0/00001.png
.
.
```
for FFHQ and
```
/storage/ImageNet/0/ILSVRC2012_val_00000001.jpeg
/storage/ImageNet/0/ILSVRC2012_val_00000002.jpeg
.
.
```
for ImageNet. Once you have the data downloaded, update `full_data_path` in `ffhq_conig.yaml` and `imagenet_config.yaml`.

### 4) Setup the Environment
```
conda create -n ddfire python=3.8

conda activate ddfire

pip install -r requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

After this you are ready to go!

## Running the Code
The entry point is `main.py`. The arguments are as follows:
- `--model_config`: The config file for the pretrained model
- `--diffusion_config`: The config file for the diffusion process used by the pretrained model
- `--data_config`: The config for the dataset being used
- `--problem_config`: The config for the inverse problem being solved
- `--fire_config`: The config for DDfire for the current dataset
- `--noiseless`: If this argument is present no noise is added to the measurements
- `--sig_y`: The measurement noise standard deviation
- `--nfes`: The number of NFEs to run
- `--gpus`: The number of GPUs to use

An example invocation for Gaussian deblurring with noisy FFHQ data is
```
python main.py \
--model_config=configs/ffhq_model_config.yaml \
--diffusion_config=configs/ffhq_diffusion_config.yaml \
--data_config=configs/ffhq_config.yaml \
--problem_config=configs/problems/ffhq/blur_gauss_config.yaml \
--fire_config=configs/fire_config_imagenet.yaml
--sig_y=0.05 --nfes=1000 --gpus=1
```

Similarly, an example for Gaussian deblurring with noisy ImageNet data is
```
python main.py \
--model_config=configs/imagenet_model_config.yaml \
--diffusion_config=configs/imagenet_diffusion_config.yaml \
--data_config=configs/imagenet_config.yaml \
--problem_config=configs/problems/imagenet/blur_gauss_config.yaml \
--fire_config=configs/fire_config_imagenet.yaml
--sig_y=0.05 --nfes=1000 --gpus=1
```

Bash scripts that execute the above are available in the `scripts/` directory.

## Evaluating Performance
To evaluate performance for a given dataset/problem, run `evaluate_samples.py` as follows:
```
python evaluate_samples.py \
--data_config=configs/{DATA_CONFIG}.yaml \
--problem_config=configs/problems/{DATASET}/{PROBLEM_CONFIG}.yaml \
--nfes=1000 --num_ims=1000
```

To evaluate FID, we use the pytorch_fid module, which can be run by
```
python -m pytorch_fid /path/to/gt/images /path/to/reconstructed/images
```

## Extending the Code to New Inverse Problems
### Adding a New Forward Operator
First, implement your forward operator in `guided_diffusion/ddrm_svd.py`. Please follow the convention of
other operators in the file: extend the `H_funcs` class and implement the `H` and `Ht` functions. You will need to verify the
correctness of your forward operator implementation. An incorrect implementation will break things.

Once your forward operator is implemented, add a new `elif` block to the `get_operator` function in `guided_diffusion/ddrm_svd.py`.
You will need to assign your forward operator a `deg` keyword here. E.g., sr8x-bicubic for 8x Bicubic Super Resolution.
Next, add your `deg` keyword to the `accepted_operators` array in `get_operator`.

Finally, create a new problem config file in the appropriate directory. E.g., `configs/problems/ffhq/deg_config.yaml`. Please mode it after the existing files.

#### Tuning K, delta_1
Coming soon!

### Adding a New Dataset
To add a new dataset, you will need to create appropriate new configs in `configs/`. For instance, if one were
to add LSUN Churches, you would need to add `lsun_church_config.yaml`, `lsun_church_diffusion_config.yaml`, and `lsun_church_model_config.yaml` files.
Please model them after the existing config files.
If you store your data in the same way that you did for FFHQ and ImageNet, and properly set the config files, it should be relatively painless
to add a new dataset. It should also work out of the box with the `ImageDataModule` that the other datasets use, or it may require small modification.

#### Computing the Learned FIRE Scale Factor
The FIRE algorithm leverages a 'scale factor' computed from an additional validation set.
We use this to compute an initial estimate for the pretrained denoiser output error variance.
In `eta_scale/`, this scale factor is computed for FFHQ and ImageNet at all diffusion timesteps.
If you add a new dataset, or change the FFHQ/ImageNet denoiser, you will need to recompute these scale factors.
This is straightforward to do with `compute_eta_map.py`. Example scripts for computing the scale factors for FFHQ, ImageNet are included in 
`scripts/`.
