# 2023-MS-Project-Long-Tailed-Learning-with-Noisy-Labels

This repository contains the code for Jinghao Shen's MS Project

The framework comprises all the basic stages: feature extraction, training, inference and evaluation. After loading the CIFAR10/CIFAR100 dataset, a resnet baseline is trained and evaluated. The code also allows to test four noise-robust loss functions. 

### Content
- [1. Type of Learning Approach](#1-type-of-learning-approach)
- [2. Top-tier Conference Papers](#2-top-tier-conference-papers)
  - [Long-Tailed Learning with Noisy Labels](#long-tailed-learning-with-noisy-labels)
  - [Noisy Labels](#noisy-labels)
  - [Long-tailed Learning](#long-tailed-learning)
  - [Benchmark Datasets](#benchmark-datasets)
- [3. Our codebase](#3-our-codebase)
  - [Example execution code for cifar100](#2-example-execution-code-for-cifar100)
- [4. Survey References](#4-survey-references)


## 1. Type of Learning Approach

| Symbol | `Sampling`  |          `CSL`          |       `LA`       |       `TL`        |       `Aug`       |
| :----- | :---------: | :---------------------: | :--------------: | :---------------: | :---------------: |
| Type   | Re-sampling | Cost-sensitive Learning | Logit Adjustment | Transfer Learning | Data Augmentation |

| Symbol |          `RL`           |       `CD`        |        `DT`        |    `Ensemble`     |   `other`   |
| :----- | :---------------------: | :---------------: | :----------------: | :---------------: | :---------: |
| Type   | Representation Learning | Classifier Design | Decoupled Training | Ensemble Learning | Other Types |

## 2. Top-tier Conference Papers

### Long-Tailed Learning with Noisy Labels

| Title                                                        |  Venue  | Year |       Type       |                             Code                             |
| :----------------------------------------------------------- | :-----: | :--: | :--------------: | :----------------------------------------------------------: |
| [Fairness Improves Learning from Noisily Labeled Long-Tailed Data](https://arxiv.org/pdf/2303.12291.pdf) |  ArXiv   | 2023 |        |                   |
| [Meta-learning advisor networks for long-tail and noisy labels in social image classification](https://dl.acm.org/doi/pdf/10.1145/3584360) |  ACM   | 2023 |        |                   |
| [Identifying Hard Noise in Long-Tailed Sample Distribution](https://arxiv.org/pdf/2207.13378.pdf) |  ECCV   | 2022 |        |         [Official](https://github.com/yxymessi/H2E-Framework)          |
| [Combating Noisy Labels in Long-Tailed Image Classification](https://arxiv.org/pdf/2209.00273.pdf) |  ICLR   | 2022 |         |                                                              |
| [Learning from Long-Tailed Noisy Data with Sample Selection and Balanced Loss](https://arxiv.org/pdf/2211.10906.pdf) |  ArXiv   | 2022 |         |            |
| [Sample Selection with Uncertainty of Losses for Learning with Noisy Labels](https://arxiv.org/pdf/2106.00445.pdf) |  ICLR   | 2022 |  `Sampling`  |                          [Official](https://github.com/xiaoboxia)                                    |
| [Prototypical Classifier for Robust Class-Imbalanced Learning](https://arxiv.org/pdf/2110.11553.pdf) |  PAKDD  | 2021 |  |    [Official](https://github.com/Stomach-ache/PCL)     |
| [Robust Long-Tailed Learning Under Label Noise](https://arxiv.org/pdf/2108.11569.pdf) |  ArXiv  | 2021 |     `Aug`       |      [Official](https://github.com/Stomach-ache/RoLT)      |
| [Learning From Long-Tailed Data With Noisy Labels](https://arxiv.org/pdf/2108.11096.pdf) |  ArXiv | 2021 |   |                 |

### Noisy Labels

| Title                                                        |  Venue  | Year |       Type       |                             Code                             |
| :----------------------------------------------------------- | :-----: | :--: | :--------------: | :----------------------------------------------------------: |
| [Learning with Instance-Dependent Label Noise: A Sample Sieve Approach](https://openreview.net/pdf?id=2VXyy9mIyU3) | ICLR | 2021 |  |    [Official](https://github.com/haochenglouis/cores)     |
| [Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels](https://arxiv.org/pdf/1805.07836.pdf) | NeurIPS  | 2018 |         |    [Unofficial](https://github.com/edufonseca/icassp19/blob/master/losses.py)    |
| [L_DMI: A Novel Information-theoretic Loss Function for Training Deep Nets Robust to Label Noise](https://arxiv.org/pdf/1909.03388.pdf) |  NeurIPS   | 2019 |             |      [Official](https://github.com/Newbeeer/L_DMI)      |
| [Early-Learning Regularization Prevents Memorization of Noisy Labels](https://arxiv.org/pdf/2007.00151.pdf) |  NeurIPS   | 2020 |        |         [Official](https://github.com/shengliu66/ELR)          |
| [Dual T: Reducing Estimation Error for Transition Matrix in Label-noise Learning](https://arxiv.org/pdf/2006.07805.pdf) |  NeurIPS   | 2020 |         |    [Official](https://github.com/a5507203/dual-t-reducing-estimation-error-for-transition-matrix-in-label-noise-learning)    |
| [DualGraph: A graph-based method for reasoning about label noise](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_DualGraph_A_Graph-Based_Method_for_Reasoning_About_Label_Noise_CVPR_2021_paper.pdf) |  CVPR   | 2021 |         |            |
| [DivideMix: Learning with Noisy Labels as Semi-supervised Learning](https://arxiv.org/pdf/2002.07394.pdf) |  ICLR   | 2020 |    |        [Official](https://github.com/LiJunnan1992/DivideMix)         |
| [Peer Loss Functions: Learning from Noisy Labels without Knowing Noise Rates](https://arxiv.org/pdf/1910.03231.pdf) |  ICML   | 2020 |           |          [Unofficial](https://github.com/weijiaheng/Multi-class-Peer-Loss-functions)           |
| [Superloss: A generic loss for robust curriculum learning](https://proceedings.neurips.cc/paper_files/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf) |  NeurIPS   | 2020 |           |          [Official](https://github.com/AlanChou/Super-Loss)           |
| [How does Disagreement Help Generalization against Label Corruption?](https://arxiv.org/pdf/1901.04215.pdf) |  ICML   | 2019 |              | [Official](https://github.com/bhanML/coteaching_plus) |
| [Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels](https://arxiv.org/pdf/1804.06872.pdf) |  NeurIPS   | 2018 |           |          [Official](https://github.com/bhanML/Co-teaching)           |

### Long-tailed Learning

| Title                                                        |  Venue  | Year |       Type       |                             Code                             |
| :----------------------------------------------------------- | :-----: | :--: | :--------------: | :----------------------------------------------------------: |
| [Escaping Saddle Points for Effective Generalization on Class-Imbalanced Data](https://arxiv.org/pdf/2212.13827.pdf) |  NeurIPS   | 2022 |       | [Official](https://github.com/val-iisc/Saddle-LongTail) |
| [Label-Imbalanced and Group-Sensitive Classification under Overparameterization](https://openreview.net/pdf?id=UZm2IQhgIyB) | NeurIPS | 2021 |             `LA`             |      [Official](https://github.com/orparask/VS-Loss)      |
| [Balanced Knowledge Distillation for Long-tailed Learning](https://arxiv.org/pdf/2203.09081.pdf) | ArXiv | 2021 |        |       [Official](https://github.com/EricZsy/BalancedKnowledgeDistillation)         |
| [FASA: Feature augmentation and sampling adaptation for long-tailed instance segmentation](https://arxiv.org/pdf/2102.12867.pdf) |  ICCV   | 2021 |       `Sampling`,`CSL`       |     [Official](https://github.com/yuhangzang/FASA)      |
| [Disentangling label distribution for long-tailed visual recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Hong_Disentangling_Label_Distribution_for_Long-Tailed_Visual_Recognition_CVPR_2021_paper.pdf) |  CVPR   | 2021 |    `CSL`,`LA`    |       [Official](https://github.com/hyperconnect/LADE)       |
| [Influence-Balanced Loss for Imbalanced Visual Classification](https://arxiv.org/pdf/2110.02444.pdf) |  ICCV   | 2021 |            `CSL`             |        [Official](https://github.com/pseulki/IB-Loss)        |
| [Long-tail learning via logit adjustment](https://openreview.net/pdf?id=37nvvqkCo5) |  ICLR   | 2021 |             `LA`             | [Official](https://github.com/google-research/google-research/tree/master/logit_adjustment) |
| [Adjusting Decision Boundary for Class Imbalanced Learning](https://arxiv.org/pdf/1912.01857.pdf) | ArXiv | 2020 |  |    [Official](https://github.com/feidfoe/AdjustBnd4Imbalance)     |
| [To Balance or Not to Balance: A Simple-yet-Effective Approach for Learning with Long-Tailed Distributions](https://arxiv.org/pdf/1912.04486.pdf) |  CVPR   | 2020 |             |            |
| [Decoupling representation and classifier for long-tailed recognition](https://openreview.net/pdf?id=r1gRTCVFvB) |  ICLR   | 2020 | `Sampling`,`CSL`,`RL`,`CD`,`DT` | [Official](https://github.com/facebookresearch/classifier-balancing) |
| [Deep representation learning on long-tailed data: A learnable embedding augmentation perspective](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Deep_Representation_Learning_on_Long-Tailed_Data_A_Learnable_Embedding_Augmentation_CVPR_2020_paper.pdf) |  CVPR   | 2020 |         `TL`,`Aug`,`RL`         |                           |
| [Identifying and Compensating for Feature Deviation in Imbalanced Deep Learning](https://arxiv.org/pdf/2001.01385.pdf) |  NeurIPS   | 2020 |      `LA`      |      |
| [Learning imbalanced datasets with label-distribution-aware margin loss](https://proceedings.neurips.cc/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf) | NeurIPS | 2019 |   `CSL`    |        [Official](https://github.com/kaidic/LDAM-DRW)        |
| [Focal loss for dense object detection](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf) |  ICCV   | 2017 | `CSL` |      |

### Benchmark Datasets

| Title                                                        |  Venue  | Year |       Type       |                             Code                             |
| :----------------------------------------------------------- | :-----: | :--: | :--------------: | :----------------------------------------------------------: |
| [A Benchmark of Long-tailed Instance Segmentation with Noisy Labels (Short Version)](https://arxiv.org/pdf/2211.13435.pdf) | ArXiv | 2022 |  |    [Official](https://github.com/GuanlinLee/Noisy-LVIS)     |

## 3. Our codebase
### Dependencies
This framework is tested on Ubuntu 20.04.5.

### Usage
#### (0) Download the dataset:

Download CIFAR through the <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">dataset companion site</a>, unzip it and locate it in 'data' directory.

#### (1) Adjust parameters:

The goal is to define the parameters of the experiment. The default parameters and explanations are: 

```
usage: main.py [--resume] [-a] [--batch_size] [--lr] [--start-epoch] [--epochs] [--num_classes]
                [--noise_rate] [--noise_type] [--num_gradual]  [--dataset] [--lt_type] [--lt_rate]
                [--momentum] [--weight-decay] [--loss] [--random_state] [--WVN_RS] [--model_dir] [--save_dir]
                [--train_rule] [--gpu]

  options:
    --resume                      path to latest checkpoint (default: None)
    --arch, -a                    model architecture (default: ResNet34)
    --bs                          batch size (default: 64)
    --lr                          learning rate (default: 0.1)
    --start-epoch                 initial epoch (default: 0)
    --epochs                      total epoches (default: 200)
    --num_classes                 num of classes (default: 10)
    --noise_rate                  noise level (default: 0.3 for 30%)
    --noise_type                  noise type (default: symmetric)
    --num_gradual                 only for co-teaching approaches, how many epochs for linear drop rate (default: 10)
    --dataset                     dataset (default: cifar10)
    --lt_type                     long-tailed type (default: exp)
    --lt_rate                     long-tailed ratio (default: 0.02 for factor = 50)
    --momentum                    momentum (default: 0.9)
    --weight-decay                weight decay (default: 1e-4)
    --loss                        loss (default: CE)
    --random_state                random state (default: 0)
    --WVN_RS                      WVN+RS is used when parameter called
    --model_dir                   only for approaches with teacher model (default: None)
    --save_dir                    save directory path (default: None)
    --train_rule                  model training strategy (default: None)
    --gpu                         GPU id to use (default: 0)     

```

#### (2) Example execution code for cifar100:
* CE:
  ```
  cd ./LT 
  python main.py --dataset cifar100 --loss CE --train_rule None --epochs 200 --num_classes 100 --gpu 0
  ```
* CB_CE:
  ```
  cd ./LT 
  python main.py --dataset cifar100 --loss CB_CE --train_rule Reweight --epochs 200 --num_classes 100 --gpu 0
  ```
* CB_Focal:
  ```
  cd ./LT 
  python main.py --dataset cifar100 --loss CB_Focal --train_rule Reweight --epochs 200 --num_classes 100 --gpu 0
  ```
* Focal:
  ```
  cd ./LT 
  python main.py --dataset cifar100 --loss Focal --train_rule None --epochs 200 --num_classes 100 --gpu 0
  ```
* LADE:
  ```
  cd ./LT 
  python main.py --dataset cifar100 --loss LADE --train_rule None --epochs 200 --num_classes 100 --gpu 0
  ```
* LDAM:
  ```
  cd ./LT 
  python main.py --dataset cifar100 --loss LDAM --train_rule DRW --epochs 200 --num_classes 100 --gpu 0
  ```
* logits_adjustment:
  ```
  cd ./LT 
  python main.py --dataset cifar100 --loss logits_adjustment --train_rule None --epochs 200 --num_classes 100 --gpu 0
  ```
* IB:
  ```
  cd ./LT 
  python main.py --dataset cifar100 --loss IB --train_rule IBReweight --epochs 200 --num_classes 100 --gpu 0
  ```
* IB_Focal:
  ```
  cd ./LT 
  python main.py --dataset cifar100 --loss IBFocal --train_rule IBReweight --epochs 200 --num_classes 100 --gpu 0
  ```
* BKD:
  ```
  cd ./LT 
  Stage 1: python main.py --dataset cifar100 --loss CE --train_rule None --lt_type exp --lt_rate 0.02 --noise_rate 0.3 --noise_type symmetric --epochs 200 --num_classes 100 --gpu 0
  Stage 2: python main.py --dataset cifar100 --loss BKD --train_rule None --lt_type exp --lt_rate 0.02 --noise_rate 0.3 --noise_type symmetric --epochs 200 --num_classes 100 --gpu 0 --model_dir results/cifar100/CE
  ```
* VS:
  ```
  cd ./LT 
  python main.py --dataset cifar100 --loss VS --train_rule None --epochs 200 --num_classes 100 --gpu 0
  ```
* WVN+RS:
  ```
  cd ./LT 
  python main.py --dataset cifar100 --loss CE --train_rule None --epochs 200 --num_classes 100 --gpu 0 --WVN_RS
  ```
* Co-teaching:
  ```
  cd ./NL 
  python main.py --dataset cifar100 --loss coteaching --train_rule None --epochs 200 --num_classes 100 --gpu 0 --drop_last
  ```
* Co-teaching_plus:
  ```
  cd ./NL 
  python main.py --dataset cifar100 --loss coteaching_plus --train_rule None --epochs 200 --num_classes 100 --gpu 0 --drop_last
  ```
* Dual_T estimator:
  ```
  cd ./NL 
  Stage 1: python main.py --dataset cifar100 --loss CE --train_rule None --lt_type exp --lt_rate 0.02 --noise_rate 0.3 --noise_type symmetric --epochs 40 --num_classes 100 --gpu 0 --save_dir results/cifar100/CE/est_t
  Stage 2: python main.py --dataset cifar100 --loss CLS --train_rule Dual_t --lt_type exp --lt_rate 0.02 --noise_rate 0.3 --noise_type symmetric --epochs 200 --num_classes 100 --gpu 0 --model_dir results/cifar100/CE/est_t
  ```
* Dual_T Co-teaching:
  ```
  cd ./NL 
  python main.py --dataset cifar100 --loss_type coteaching --train_rule Dual_t --epochs 200 --num_classes 100 --gpu 0 --drop_last
  ```
* ELR:
  ```
  cd ./NL 
  python main.py --dataset cifar100 --loss ELR --train_rule None --epochs 200 --num_classes 100 --gpu 0
  ```
* Super:
  ```
  cd ./NL 
  python main.py --dataset cifar100 --loss super --train_rule None --epochs 200 --num_classes 100 --gpu 0
  ```
* CORES:
  ```
  cd ./NL 
  python main.py --dataset cifar100 --loss cores --train_rule CORES --epochs 200 --num_classes 100 --gpu 0
  ```
* CORES_logits_adjustment:
  ```
  cd ./NL 
  python main.py --dataset cifar100 --loss cores_logits_adjustment --train_rule CORES --epochs 200 --num_classes 100 --gpu 0
  ```
* DivideMix:
  ```
  cd ./NL 
  python main.py --dataset cifar100 --loss Semi --train_rule None --epochs 300 --num_classes 100 --gpu 0 --arch ResNet18
  ```
* CNLCU_soft:
  ```
  cd ./LT_NL 
  python main.py --dataset cifar100 --loss CNLCU_soft --train_rule None --train_opt CNLCU_soft --epochs 200 --num_classes 100 --gpu 0 --drop_last
  ```
* CNLCU_hard:
  ```
  cd ./LT_NL 
  python main.py --dataset cifar100 --loss CNLCU_hard --train_rule None --train_opt CNLCU_hard --epochs 200 --num_classes 100 --gpu 0 --drop_last
  ```
* RoLT:
  ```
  cd ./LT_NL 
  python main.py --dataset cifar100 --loss CE --train_rule None --train_opt RoLT --epochs 200 --num_classes 100 --gpu 0
  ```
* PCL:
  ```
  cd ./LT_NL
  python main.py --dataset cifar100 --loss PCL --train_rule None --train_opt PCL --epochs 200 --num_classes 100 --gpu 0 --arch ResNet18 --low_dim --data_aug
  ```
* LA+SL:
  ```
  cd ./LT_NL
  Stage 1: python main.py --dataset cifar100 --loss cos --train_rule None --train_opt SimSiam --epochs 200 --num_classes 100 --gpu 0 --arch ResNet18 --batch_size 512
  Stage 2: python main.py --dataset cifar100 --loss super_logits_adjustment --train_rule None --epochs 400 --num_classes 100 --gpu 0 --arch SimSiam_SSL --resume results/cifar100/cos/symmetric0.3exp0.02SimSiam.pth
  ```  

#### (3) See results:

You can check the `results/*.txt`. Results are shown in a table.
You can load `results/*.best.pth` to resume the trained model.

## 4. Survey References

- [Vanint/Awesome-LongTailed-Learning] (https://github.com/Vanint/Awesome-LongTailed-Learning)
- [weijiaheng/Advances-in-Label-Noise-Learning] (https://github.com/weijiaheng/Advances-in-Label-Noise-Learning)

## 5. Contact

You are welcome to contact me privately should you have any question/suggestion or if you have any problems running the code at jshen30@ucsc.edu.