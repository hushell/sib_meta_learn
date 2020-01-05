# sib_meta_learn
This repo contains the implementation of the *synthetic information bottleneck* algorithm for few-shot classification on Mini-ImageNet,
which is described in the paper 
[Empirical Bayes Transductive Meta-Learning with Synthetic Gradients](https://openreview.net/forum?id=Hkg-xgrYvH).


## Authors of the code
[Shell Xu Hu](http://hushell.github.io/), [Xi Shen](https://xishen0220.github.io/) and [Yang Xiao](https://youngxiao13.github.io/)


## Dependencies
The code is tested under **Pytorch > 1.0 + Python 3.6** environment. 


## How to use the code
### **Step 0**: Download Mini-ImageNet dataset

``` Bash
cd data
bash download_miniimagenet.sh 
cd ..
```

### **Step 1** (optional): train a WRN-28-10 feature network (aka backbone)
The weights of the feature network is downloaded in step 0, but you may also train from scracth by running

``` Bash
python main_feat.py --outDir miniImageNet_WRN_60Epoch --cuda --dataset miniImageNet --nbEpoch 60
```

### **Step 2**: Few-shot classification on Mini-ImageNet, e.g., 5-way-1-shot:

``` Bash
python main.py --config config/miniImageNet_1shot.yaml --seed 100 --gpu 0
```

