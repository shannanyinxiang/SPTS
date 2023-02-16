# SPTS: Single-Point Text Spotting (ACM MM 2022 Oral)

The official implementation of [SPTS: Single-Point Text Spotting](https://arxiv.org/abs/2112.07917) (ACM MM 2022 Oral). The SPTS tackles scene text spotting as an end-to-end sequence prediction task and requires only extremely low-cost single-point annotations. Below is the overall architecture of SPTS. SPTS v2 will be released at [link](https://github.com/Yuliang-Liu/SPTSv2).

![SPTS](figures/method.svg)

## Environment
We recommend using [Anaconda](https://www.anaconda.com/) to manage environments. Run the following commands to install dependencies.
```
conda create -n spts python=3.7 -y
conda activate spts
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
git clone https://github.com/shannanyinxiang/SPTS
cd SPTS
pip install -r requirements.txt
```

## Dataset 

- CurvedSynText150k [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_ABCNet_Real-Time_Scene_Text_Spotting_With_Adaptive_Bezier-Curve_Network_CVPR_2020_paper.pdf): 
  - Part1 (94,723) Download (15.8G) ([Origin](https://universityofadelaide.box.com/s/xyqgqx058jlxiymiorw8fsfmxzf1n03p), [Google](https://drive.google.com/file/d/1OSJ-zId2h3t_-I7g_wUkrK-VqQy153Kj/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1Y5pqVqfjcc4FKxW4y8R5jw) password: 4k3x) 
  - Part2 (54,327) Download (9.7G) ([Origin](https://universityofadelaide.box.com/s/e0owoic8xacralf4j5slpgu50xfjoirs), [Google](https://drive.google.com/file/d/1EzkcOlIgEp5wmEubvHb7-J5EImHExYgY/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1gRv-IjqAUu6qnXN5BXlOzQ) password: a5f5)

- Totaltext [[paper]](https://ieeexplore.ieee.org/abstract/document/8270088/) [[source]](https://github.com/cs-chan/Total-Text-Dataset). 
  - Download (0.4G) ([Google](https://drive.google.com/file/d/1jfBYrAmh6Zshb7Jc0bctRjQKpK839SFq/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/18brRQAwnqGd4A_uwPRYRng) password: 5nhw) 
  
- SCUT-CTW1500 [[paper]](https://www.sciencedirect.com/science/article/pii/S0031320319300664) [[source]](https://github.com/Yuliang-Liu/Curve-Text-Detector).
  - Download (0.8G) ([Google](https://drive.google.com/file/d/1yjpsNmcjNHBPAeFNvSpYJOQPb1gRkV0K/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/193y6N_Ek1184PZ7PbEljmA) password: 82vs)
   
- MLT [[paper]](https://ieeexplore.ieee.org/abstract/document/8270168).
  - Download (6.8G) ([Origin](https://universityofadelaide.box.com/s/qu2wctdcsxh73bb94krdredpmx9nzf8m), [Google](https://drive.google.com/file/d/1nE2d_sIfcAejgVIv6-UjGNcBXgxc4QfD/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1rjqmb3uuki_Ppcxq-tl7oQ) password: zqrm)

- ICDAR2013 [[paper]](https://rrc.cvc.uab.es/?ch=2) [[source]](https://rrc.cvc.uab.es/?ch=2). 
  - Download (0.2G) ([Google](https://drive.google.com/file/d/1dMffINYhIRa9UD_3pzTFllVwL6PK7KXD/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1PiSZxZlG38qjj7Xb05cXdg) password: 5ddh) 
 
- ICDAR2015 [[paper]](https://rrc.cvc.uab.es/?ch=4) [[source]](https://rrc.cvc.uab.es/?ch=4). 
  - Download (0.1G) ([Google](https://drive.google.com/file/d/1THhzo_WH1RY5DlGdBfjRA_dwu9tAmQUE/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1x3EpYLRa4EtSMNg5JqszVg) password: wjrh) 

Please download and extract the above datasets into the `data` folder following the file structure below.

```
data
├─CTW1500
│  ├─annotations
│  │      test_ctw1500_maxlen25.json
│  │      train_ctw1500_maxlen25_v2.json
│  ├─ctwtest_text_image
│  └─ctwtrain_text_image
├─icdar2013
│  │  ic13_test.json
│  │  ic13_train.json
│  ├─test_images
│  └─train_images
├─icdar2015
│  │  ic15_test.json
│  │  ic15_train.json
│  ├─test_images
│  └─train_images
├─mlt2017
│  │  train.json
│  └─MLT_train_images
├─syntext1
│  │  train.json
│  └─syntext_word_eng
├─syntext2
│  │  train.json
│  └─emcs_imgs
└─totaltext
    │  test.json
    │  train.json
    ├─test_images
    └─train_images
```

## Train

The model training in the original paper uses 32 GPUs (4 nodes, 8 GPUs per node). Below are the instructions for the training using a single machine with 8 GPUs, which can be simply modified to multi-node training following [PyTorch Distributed Docs](https://pytorch.org/docs/1.8.0/distributed.html).
### 1. Pretrain
```
python -m torch.distributed.launch \
       --nproc_per_node=8 \
       main.py \
       --data_root ./data/ \
       --train_dataset totaltext_train mlt_train ic13_train ic15_train syntext1_train syntext2_train \
       --output_folder ./output/pretrain/ \
       --lr 0.0005 \
       --epochs 150 \
       --checkpoint_freq 2 \
       --freeze_bn \
       --batch_aug \
       --tfm_pre_norm
```
### 2. Finetune
For example, the command for finetuning on Total-Text is:
```
python -m torch.distributed.launch \
       --nproc_per_node=8 \
       main.py \
       --data_root ./data/ \
       --train_dataset totaltext_train\
       --output_folder ./output/totaltext/ \
       --resume ./output/pretrain/checkpoints/checkpoint_ep0149.pth \
       --lr 0.00001 \
       --epochs 350 \
       --checkpoint_freq 10 \
       --freeze_bn \
       --batch_aug \
       --tfm_pre_norm \
       --finetune 
```
Set "--train_dataset" to "ctw1500_train", "ic13_train" or "ic15_train" to finetune on other datasets.

## Inference
The trained models can be obtained after finishing the above steps. You can also download the models for the Total-Text, SCUT-CTW1500, ICDAR2013, and ICDAR2015 datasets from [GoogleDrive](https://drive.google.com/drive/folders/1qLPOpwOV40RyEB8ZWd-0LZ_HkozombnO?usp=sharing) or [BaiduNetDisk](https://pan.baidu.com/s/17OpkuZr_59vVEgAoz2nsmQ) password: b9u5.

For example, the command for the inference on Total-Text is:
```
python main.py \
       --eval \
       --data_root ./data/ \
       --val_dataset totaltext_val \
       --output_folder ./output/totaltext/ \
       --resume ./output/totaltext/checkpoints/checkpoint_ep0349.pth \
       --tfm_pre_norm \
       --visualize
```

## Evaluation

First, download the ground-truth files ([GoogleDrive](https://drive.google.com/file/d/1ztyjczfn3YdBf6hpLuV2Vs2UJPlRdAjm/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1ERkKR8L58ZVlB12SpCwEVQ) password: 35tr) and lexicons ([GoogleDrive](https://drive.google.com/file/d/1JxmuDsOZ-x_WO5lck2ZQZHRcjoUtUiLo/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1so_s94_XysLjlcWasos8mA) password: 9eml), and extracted them into the `evaluation` folder.

```
evaluation
│  eval.py
├─gt
│  ├─gt_ctw1500
│  ├─gt_ic13
│  ├─gt_ic15
│  └─gt_totaltext
└─lexicons
    ├─ctw1500
    ├─ic13
    ├─ic15
    └─totaltext
```

Then the command for evaluating the inference result of Total-Text is:
```
python evaluation/eval.py \
       --result_path ./output/totaltext/results/ep349/totaltext_val.json \
       # --with_lexicon \ # uncomment this line if you want to evaluate with lexicons.
       # --lexicon_type 0 # used for ICDAR2013 and ICDAR2015. 0: Generic; 1: Weak; 2: Strong.
```

## Performance

The end-to-end recognition performances of SPTS on four public benchmarks are:

| Dataset | Strong | Weak | Generic |
| ------- | ------ | ---- | ------- |
| ICDAR 2013 | 93.3 | 91.7 | 88.5 |
| ICDAR 2015 | 77.5 | 70.2 | 65.8 |

| Dataset | None | Full |
| ------- | ---- | ---- |
| Total-Text | 74.2 | 82.4 |
| SCUT-CTW1500 | 63.6 | 83.8 |

## Citation
```
@inproceedings{peng2022spts,
  title={SPTS: Single-Point Text Spotting},
  author={Peng, Dezhi and Wang, Xinyu and Liu, Yuliang and Zhang, Jiaxin and Huang, Mingxin and Lai, Songxuan and Zhu, Shenggao and Li, Jing and Lin, Dahua and Shen, Chunhua and Bai, Xiang and Jin, Lianwen},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
```

## Copyright
This repository can only be used for non-commercial research purpose.

For commercial use, please contact Prof. Lianwen Jin (eelwjin@scut.edu.cn).

Copyright 2022, [Deep Learning and Vision Computing Lab](http://www.dlvc-lab.net), South China University of Technology. 

## Acknowledgement
We sincerely thank [Stable-Pix2Seq](https://github.com/gaopengcuhk/Stable-Pix2Seq), [Pix2Seq](https://github.com/google-research/pix2seq), [DETR](https://github.com/facebookresearch/detr), and [ABCNet](https://github.com/aim-uofa/AdelaiDet) for their excellent works.
