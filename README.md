# Linearly Constrained Diffusion Implicit Models
![alt text](Teaser.jpg)

### Authors
[Vivek Jayaram](http://www.vivekjayaram.com/), [John Thickstun](https://johnthickstun.com/), [Ira Kemelmacher-Shlizerman](https://homes.cs.washington.edu/~kemelmi/), and [Steve Seitz](https://homes.cs.washington.edu/~seitz/)

### Links
[[Gradio Demo]](https://huggingface.co/spaces/vivjay30/cdim) [[Project Page]](https://grail.cs.washington.edu/projects/cdim/) [[Paper]](https://arxiv.org/abs/2411.00359)

### Summary
We solve noisy linear inverse problems with diffusion models. The method is fast and addresses many problems like inpainting, super-resolution, gaussian deblur, and poisson noise. 


## Getting started 

Recommended environment: Python 3.11, Cuda 12, Conda. For lower verions please adjust the dependencies below.

### 1) Clone the repository

```
git clone https://github.com/vivjay30/cdim

cd cdim
```

### 2) Install dependencies

```
conda create -n cdim python=3.11

conda activate cdim

pip install -r requirements.txt

pip install torch==2.4.1+cu124 torchvision-0.19.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

## Inference Examples

(The underlying diffusion models will be automatically downloaded on the first run).

#### CelebHQ Inpainting Example (T'=25 Denoising Steps)

`python inference.py sample_images/celebhq/00001.jpg 25 operator_configs/box_inpainting_config.yaml noise_configs/gaussian_noise_config.yaml google/ddpm-celebahq-256`
 
#### LSUN Churches Gaussian Deblur Example (T'=25 Denoising Steps)
`python inference.py sample_images/lsun_church.png 25 operator_configs/gaussian_blur_config.yaml noise_configs/gaussian_noise_config.yaml google/ddpm-church-256`

 
## FFHQ and Imagenet Models
These models are generally not as strong as the google ddpm models, but are used for comparisons with baseline methods.

From [this link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the checkpoints "ffhq_10m.pt" and "imagenet_256.pt" to models/

#### Imagenet Super Resolution Example
Here we set T'=50 to show the algorithm running slower
`python inference.py sample_images/imagenet_val_00002.png 50 operator_configs/super_resolution_config.yaml noise_configs/gaussian_noise_config.yaml models/imagenet_model_config.yaml`

#### FFHQ Random Inpainting (Faster)
Here we set T'=10 to show the algorithm running faster
`python inference.py sample_images/ffhq_00010.png 10 operator_configs/random_inpainting_config.yaml noise_configs/gaussian_noise_config.yaml models/ffhq_model_config.yaml`

#### A Note on Exact Recovery
If you set the measurement noise to 0 in gaussian_noise_config.yaml, then the recovered image should match the the observation y exactly (e.g. inpainting doesn't chance observed pixels). In practice, this doesn't happen because the diffusion schedule sets $\overline{\alpha}_0 = 0.999$ for numeric stability, meaning a tiny amount of noise is injected even at t=0.


