# Robust T-Loss for Medical Image Segmentation

<br>


<p align="center">
<img src="./assets/tloss.gif" width="320"/>
</p>

[**Project**](https://robust-tloss.github.io/) | [**Paper**](https://arxiv.org/abs/2306.00753) 

PyTorch implementation of **Robust T-Loss for Medical Image Segmentation** (MICCAI 2023).

[Alvaro Gonzalez-Jimenez](https://alvarogonjim.github.io), [Simone Lionetti](https://www.hslu.ch/en/lucerne-university-of-applied-sciences-and-arts/about-us/people-finder/profile/?pid=4484), Philippe Gottfrois, Fabian Gröger, [Marc Pouly](https://marcpouly.ch/) and [Alexander Navarini](https://navarinilab.com/)



This repository contains the implementation of the T-Loss, a novel robust loss function designed for semantic segmentation tasks in medical image analysis. The T-Loss is inspired by the negative log-likelihood of the Student-t distribution and offers a simple yet effective solution to mitigate the adverse effects of noisy labels on deep learning models. By adaptively learning an optimal tolerance level for label noise during backpropagation, the T-Loss eliminates the need for additional computations, making it highly efficient for practical applications. In this repository, you will find the T-Loss implementation, along with scripts to generate the noise masks and reproduce the experiments conducted on benchmark datasets for skin lesion segmentation and lung segmentation. 


## Getting Started

### Install Dependencies

Before proceeding, ensure that the required dependencies are installed. You can do this by using the following command:

```
pip install -r requirements.txt
```

### Generate Noise Masks

To reproduce the experiments, you first need to generate noise masks using the official ISIC 2017 dataset and the Shenzhen dataset. Follow these steps:

1. Download the official [ISIC 2017 dataset](https://challenge.isic-archive.com/data/#2017) and the [Shenzhen dataset](https://www.kaggle.com/datasets/yoctoman/shcxr-lung-mask).

2. Use the script located in `datasets/noise/data_preprocessing.py` to generate the noise masks. This scripted was adapted from [https://github.com/gaozhitong/SP_guided_Noisy_Label_Seg](https://github.com/gaozhitong/SP_guided_Noisy_Label_Seg). Run the script, replacing the `root` with the path where you stored the datasets:

```bash
python3 data_preprocessing.py --dataname isic --dataset_noise_ratio 0.3 0.5 0.7 --sample_noise_ratio 0.5 0.7 --root /data/ISIC/
```


This script will generate noise masks with dataset noise ratios ($\alpha$) of 0.3, 0.5, and 0.7 and strength ($\beta$) of 0.5 and 0.7. The resulting files will be stored in the parent folder where the datasets are located, in our example, `/data/ISIC_noise`.

## Experiments

### Run Experiments Without Docker 

To train the model without Docker, first, you have to adapt the configuration files in `./configs/` with the corresponding paths where the generated noise masks are located. Then you can run the following training script:

```
python3 train.py --config ./configs/isic/isic_alpha07_beta07.py --workdir Results_ISIC
```
In this example, we are training using the ISIC dataset with noise masks of 0.7 $\alpha$ and 0.7 $\beta$ using the configuration file `./configs/isic/isic_alpha07_beta07.py`. Please adapt to the desired configuration, and `Results_ISIC` with the desired work directory where you want to store the results.

### Run Experiments Using Docker

For reproducibility and isolation, we provide a Docker image with the necessary environment to run the experiments, with the Docker option there is no need to make modifications in the config files. 

1. Build the Docker image:
```
docker build -t t-loss .
```

2. Run the Docker container, specifying the GPU if available, and replace `/root/` with the parent path where the datasets are located:

```
docker run --gpus all -d -t --shm-size 16G --name tloss -v /root/:/data/ tloss-container /bin/bash
```

3. Finally, run the experiments with the desired configuration file, dataset noise ratio, and sample noise ratio using the Docker container: 

```
# 0.3 Alpha 0.5 Beta
docker exec -d tloss-container python3 train.py --config ./configs/isic/isic_alpha03_beta05.py --workdir Results_ISIC

# 0.3 Alpha 0.7 Beta
docker exec -d tloss-container python3 train.py --config ./configs/isic/isic_alpha03_beta07.py --workdir Results_ISIC

# 0.5 Alpha 0.5 Beta
docker exec -d tloss-container python3 train.py --config ./configs/isic/isic_alpha05_beta05.py --workdir Results_ISIC

# 0.5 Alpha 0.7 Beta
docker exec -d tloss-container python3 train.py --config ./configs/isic/isic_alpha05_beta07.py --workdir Results_ISIC

# 0.7 Alpha 0.7 Beta
docker exec -d tloss-container python3 train.py --config ./configs/isic/isic_alpha07_beta07.py --workdir Results_ISIC

# 0.3 Alpha 0.7 Beta
docker exec -d tloss-container python3 train.py --config ./configs/isic/isic_alpha07_beta07.py --workdir Results_ISIC
```

In this example we are storing the results in a directory called `Results_ISIC`. You can copy the results from the Docker container to your local machine using:

```
docker cp tloss-container:/app/Results_ISIC ./
```


## References
If you find this repository useful for your research, please cite the following work.

```bib
@inproceedings{gonzalezjimenezRobustTLoss2023,
  title     = {Robust T-Loss for Medical Image Segmentation},
  author    = {Gonzalez-Jimenez, Alvaro and Lionetti, Simone and Gottfrois, Philippe and Gröger, Fabian and Pouly, Marc and Navarini, Alexander},
  journal   = {Medical {{Image Computing}} and {{Computer Assisted Intervention}} – {{MICCAI}} 2023},
  publisher = {{Springer International Publishing}},
  year      = {2023},
}
```

## License
The code in this repository is released under the Apache License 2.0
