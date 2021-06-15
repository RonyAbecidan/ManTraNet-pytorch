[![Generic badge](https://img.shields.io/badge/Library-Pytorch-<>.svg)](https://shields.io/) [![Ask Me Anything !](https://img.shields.io/badge/Official%20-No-1abc9c.svg)](https://GitHub.com/Naereen/ama) ![visitors](https://visitor-badge.glitch.me/badge?page_id=RonyAbecidan.ManTraNet-pytorch)

Who has never met a forged picture on the web ? No one ! Everyday we are constantly facing fake pictures touched up in Photoshop but it is not always easy to detect it.

In this repo, you will find an implementation of ManTraNet, a manipulation tracing network for detection and localization of image forgeries with anomalous features. 
With this algorithm, you may find if an image has been falsified and even identify suspicious regions. A little example is displayed below.

![](https://i.imgur.com/OyErscI.png)

It's a faifthful replica of the [official implementation](https://github.com/ISICV/ManTraNet) using however the library Pytorch. To learn more about this network, I suggest you to read the paper that describes it [here](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_ManTra-Net_Manipulation_Tracing_Network_for_Detection_and_Localization_of_Image_CVPR_2019_paper.pdf).

On top of the MantraNet, there is also a file containing pre-trained weights obtained by the authors which is compatible with this pytorch version.

There is a slight discrepancy between the architecture depicted in the paper compared to the real one implemented and shared on the official repo. I put below the real architecture which is implemented here.

![](https://i.imgur.com/htcP41B.png)

Please note that the rest of the README is largely inspired by the original repo.



--- 
## What is ManTraNet ?

ManTraNet is an end-to-end image forgery detection and localization solution, which means it takes a testing image as input, and predicts pixel-level forgery likelihood map as output. Comparing to existing methods, the proposed ManTraNet has the following advantages:

- **Simplicity**: ManTraNet needs no extra pre- and/or post-processing
- **Fast**: ManTraNet puts all computations in a single network, and accepts an image of arbitrary size.
- **Robustness**: ManTraNet does not rely on working assumptions other than the local manipulation assumption, i.e. some region in a testing image is modified differently from the rest.


Technically speaking, ManTraNet is composed of two sub-networks as shown below:

- The **Image Manipulation Trace Feature Extractor**: It's a feature extraction network for the image manipulation classification task, which is sensitive to different manipulation types, and encodes the image manipulation in a patch into a fixed dimension feature vector.

- The **Local Anomaly Detection Network**: It's a network that is designed following the intuition that we need to inspect more and more locally our extracted features if we want to be able to detect many kind of forgeries efficiently.


## Where are the pre-trained weights coming from  ?

- The authors have first pretrained the Image Manipulation Trace Feature Extractor with an homemade database containing 385 types of forgeries. Unfortunately, their database is not shared publicly. Then, they trained the ManTraNet with four types of synthetic data, i.e. copy-move, splicing, removal, and enhancement.

**_The pre-trained weights available in this repo are the results of these two trainings achieved by the authors_**

**Remarks** : I provide also a way to train ManTraNet in the demo notebook. Of course, to do it you need your own (relevant) dataset and, certainly play with some hyperparameters like the learning rate (see the details of the `ForgeryDetector` class in `mantranet.py` ).


## Dependency
- **Pytorch** : 1.8.1
- **Pytorch-lightning : 1.2.10**

## Demo
One may simply download the repo and play with the provided ipython notebook.

## N.B. :
- Considering that there is some differences between the implementation of common functions between Tensorflow/Keras and Pytorch, some particular methods of Pytorch (like batch normalization or hardsigmoid) are re-implemented here to match perfectly with the original Tensorflow version

- MantraNet is an architecture difficult to train without GPU/Multi-CPU. Even in "eval" mode, if you want to use it for detecting forgeries in one image it may take some minutes
using only your CPU. It depends on the size of your input image.

- There is also a slightly different version of MantraNet that uses ConvGRU instead of ConvLSTM in the repo. It enables to speed up a bit the training of the MantraNet without losing efficiency.

## Citation :

```
@InProceedings{Wu_2019_CVPR,
author = {Wu, Yue and AbdAlmageed, Wael and Natarajan, Premkumar},
title = {ManTra-Net: Manipulation Tracing Network for Detection and Localization of Image Forgeries With Anomalous Features},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
