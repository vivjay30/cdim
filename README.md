# Constrained Diffusion Implicit Models
![alt text](Teaser.jpg)

## Authors
[Vivek Jayaram](http://www.vivekjayaram.com/), [John Thickstun](https://johnthickstun.com/), [Ira Kemelmacher-Shlizerman](https://homes.cs.washington.edu/~kemelmi/), and [Steve Seitz](https://homes.cs.washington.edu/~seitz/)

## [Project Page](www.google.com)
(Coming Soon)

### [Paper](www.google.com)
(Coming Soon)

### Summary
We solve noisy linear inverse problems with diffusion models. The method is fast and addresses many problems like inpainting, super-resolution, gaussian deblur, and poisson noise. 


## Getting started 

### 1) Clone the repository

```
git clone https://github.com/vivjay30/cdim

cd cdim

export PYTHONPATH=$PYTHONPATH:`pwd`
```

### 2) Install dependencies

```
conda create -n cdim python=3.11

conda activate cdim

pip install -r requirements.txt

pip install torch==2.4.1+cu124 torchvision-0.19.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```
