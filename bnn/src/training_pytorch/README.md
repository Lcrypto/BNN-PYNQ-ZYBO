# BNN training in pytorch

Create results directory

```bash
	$ mkdir results
```

## For Traning cnv with WxAy

```bash
	$ python cnv.py --wb x --ab y
```
To export trained model as npz to use with finnthesizer

```bash
	$ python cnv.py --wb x --ab y --export
```
Copy the generated .npz

```bash
	$ cp results/cifar10-wxay.npz ../../weights
```

## For Traning lfc with WxAy

```bash
	$ python lfc.py --wb x --ab y
```
To export trained model as npz to use with finnthesizer

```bash
	$ python lfc.py --wb x --ab y --export
```
Copy the generated .npz

```bash
	$ cp results/mnist-wxay.npz ../../weights
```

Binarized Neural Network (BNN) for pytorch
This is the pytorch version for the BNN code, fro VGG and resnet models
Link to the paper: https://papers.nips.cc/paper/6573-binarized-neural-networks

The code is based on https://github.com/eladhoffer/convNet.pytorch
Please install torch and torchvision by following the instructions at: http://pytorch.org/
To run resnet18 for cifar10 dataset use: python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10
