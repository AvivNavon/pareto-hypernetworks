# Pareto HyperNetworks 

Official implementation of ["_Learning The Pareto Front With HyperNetworks_"](https://arxiv.org/abs/2010.04104).

<p align="center"> 
    <img src="https://github.com/AvivNavon/pareto-hypernetworks/blob/master/resources/phn.png" width="350">
</p>

Pareto HyperNetworks (PHN) learn the entire pareto front using a single model.

<p align="center"> 
    <img src="https://github.com/AvivNavon/pareto-hypernetworks/blob/master/resources/toy_pf_ours.png" width="350">
</p>  

## Install

```bash
git clone https://github.com/AvivNavon/pareto-hypernetworks.git
cd pareto-hypernetworks
pip install -e .
```

## Run Experiments

To run the experiment follow the `README.md` files within each experiment folder.

## Citation

If you find PHN to be useful in your own research, please consider citing the following paper:

```bib
@inproceedings{
navon2021learning,
title={Learning the Pareto Front with Hypernetworks},
author={Aviv Navon and Aviv Shamsian and Gal Chechik and Ethan Fetaya},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=NjF772F4ZZR}
}
```
