# CLC
Official Implementation of our CLC.

# Getting Started

## Prerequisites

```
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install matplotlib scipy opencv-python pillow scikit-image tqdm tensorflow-io
```

## Preparing Datasets

Update `configs/paths_config.py` with the necessary data paths and model paths for training and inference.

```
dataset_paths = {
    'train_data': '/path/to/train/data'
    'test_data': '/path/to/test/data',
}
```

## Auxiliary Models and Preparing Generator

You can download auxiliary models from [here](https://github.com/yuval-alaluf/restyle-encoder) and put it in the directory `/pretrained_models`.

We use rosinality's [StyleGAN2 implementation](https://github.com/rosinality/stylegan2-pytorch). You can download the 256px pretrained model in the project and put it in the directory `/pretrained_models ` as well.

# Training Model

```bash
python scripts/train_CLC.py \
--dataset_type=church_encode \
--encoder_type=ResNetBackboneEncoder \
--exp_dir=experiment/church \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--w_norm_lambda=0 \
--id_lambda=0 \
--moco_lambda=0.5 \
--output_size=256 \
--input_nc=6 \
--n_iters_per_batch=5 \
--max_steps 200000 \
--use_con \
--con_lambda=1 \
```

# Inference

```bash
python scripts/inference.py \
--exp_dir=experiment/church \
--checkpoint_path=checkpoints/best_model.pt \
--data_path=/your_test_data \
--test_batch_size=4 \
--test_workers=4 \
```
