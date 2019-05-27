# A simple Pytorch implementation of Single Shot MultiBox Object Detector (SSD)

This repository is a pytorch implementation of [Single Shot MultiBox Object Detector (SSD)](https://arxiv.org/pdf/1606.06160.pdf) and it's modified based on another pytorch implementation [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch).
The main difference is that I reorgainze the code structure and make it as simple as possible, 
if you always get lost in the ocean of code (like me), I promise you will like this repository :) 

The plain python implemented nms in ssd.pytorch is relatively slow and I don't want to compile the cuda nms (which is not *simple*, at least for me). 
So I use the tensorflow nms function for evaluation, I know it sounds ridiculous but it works pretty well ! 
Evaluation speed jumps from ~1FPS to ~10FPS, the only drawback is that you can't use tensorflow and pytorch in one file 
(more precisely when using nn.DataParallel), so the there is *train.py* for training and *eval.py* for evaluation.    

There are three main parts: *./dataset*, *./nets* and *./utils*. 
* *./dataset* contains the data reading and augmentation code
* *./nets* contains the backbone, detection head as well as the anchor box code
* *./utils* contains boundingbox processing, loss function, nms and mAP evaluation utils.    
 
## Requirements:
- python >= 3.5
- pytorch >= 1.0
- tensorflow >= 1.0
- tensorboardX
- tqdm

## Train
* ```python train.py --data_dir YOUR_VOC_DIR ```

    (currently I only tested the training with batch size 64 on 2 GPUs, 
you may need to change some arguments (batch size, lr, etc) to run it on a single GPU)

## Evaluate
* ```python eval.py --eval_data_dir YOUR_VOC_DIR ```

## Pascal VOC results:
Method|Train set|Eval set|Paper mAP|Reproduced mAP
:---:|:---:|:---:|:---:|:---:
SSD300|VOC07+12|VOC07 test|77.2|77.42

