## 2023-MS-Project-Long-Tailed-Learning-with-Noisy-Labels

This repository contains the code for Jinghao Shen's MS Project

The framework comprises all the basic stages: feature extraction, training, inference and evaluation. After loading the CIFAR10/CIFAR100 dataset, a resnet baseline is trained and evaluated. The code also allows to test four noise-robust loss functions. 

## Dependencies
This framework is tested on Ubuntu 20.04.5. To duplicate the environment:

`Under construction`
<!---`conda create --name <envname> --file requirements.txt`--->


## Directories and files

`data/` folder where to include downloaded datasets
`results/` folder where to include output files per experiment  

`main.py` is the main script  
`dataset.py` contains the data generators  
`bias_cifar.py` contains bias generateors 
`metrics.py` contains functions for matric estimators 
`utils.py` some basic utilities  
`resnet.py` contains models 
`losses.py` definition of several loss functions  



## Usage

#### (0) Download the dataset:

Download CIFAR through the <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">dataset companion site</a>, unzip it and locate it in a given directory.

#### (1) Adjust parameters:

The goal is to define the parameters of the experiment. The most important parameters are: 
   
`noise_type`: type of noise 
`lt_type`: type of long-tailed distribution
`loss`: defines the loss function. To be decided among:
`dual_t`: apply dual T estimator or not

  - `cross_entropy`: cross entropy loss
  - `focal_loss`: 
  - `logits_adjustment`: 
  - `cores`: 
  - `gce`: 
  - `cb_ce`: 
  - `cb_focal`: 
  - `cores_no_select`: 
  - `cores_logits_adjustment`: 
  - `erl`:
  - `coteaching`: 
  - `coteaching_plus`: 
  - `cls`: 


The rest of the parameters should be rather intuitive.


#### (2) Execute the code by:
- run, for instance: `python main.py --dataset cifar100 --loss cross_entropy`


#### (3) See results:

You can check the `results/*.txt`. Results are shown in a table.


## Reproducing the baseline

Under construction
<!--
#### (1) Edit `config/*.yaml` file

  - `ctrl.train_data: all` # (or any other train subset)
  - `loss.type: CCE` # this is standard cross entropy loss
 
#### (2) Execute the code.
-->
## Baseline system details

Under construction
 
## Contact

You are welcome to contact me privately should you have any question/suggestion or if you have any problems running the code at jshen30@ucsc.edu.