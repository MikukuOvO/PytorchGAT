# Pytorch GAT

A pytorch implementation of the GAT model based on the officail code, adding code for plots & using mps for accelerate running on MacOS.

The cora data set is used for evaluating the learning results.

To run training & testing process:
```
python main.py
```
The requirements are as following:
```
torch==2.1.2
torchaudio==2.1.2
torchvision==0.16.2
matplotlib
scipy==1.12.0
```