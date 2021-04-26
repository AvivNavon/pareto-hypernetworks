# MultiMNIST

<p align="center"> 
    <img src="https://github.com/AvivNavon/pareto-hypernetworks/blob/master/resources/mmnist_fashion_and_mnist_evolve.png" width="600">
</p>

We use the data provided in [Pareto Multi-Task Learning](https://papers.nips.cc/paper/9374-pareto-multi-task-learning).
The data is available [here](https://drive.google.com/drive/folders/1VnmCmBAVh8f_BKJg1KYx-E137gBLXbGG).

Please create a `data` folder and download the `.pickle` files. 

## Run experiments

We support two model variants: `resnet` and `lenet` controlled by the `--model` flag. To run the LeNet experiment, use e.g.,

```bash
python trainer.py --datapath data/multi_fashion_and_mnist.pickle --model lenet
```

For the ResNet experiment with 5M trainable parameters, use e.g.,

```bash
python trainer.py --datapath data/multi_fashion_and_mnist.pickle --model resnet --resnet-size 5M
```

We also support 1M, 2M and 11M trainable parameters.