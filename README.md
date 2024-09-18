
![net_arch](docs/lsk.png)

## This repository is the official implementation of ArXiv "SM3Det: A Single Remote Sensing Object Detection Model for Multi-Modal Datasets and Multi-Task Joint Training" at: [Here]( )

## Abstract

With the rapid advancement of remote sensing technology, high-resolution, multi-modal imagery is now more widely accessible. Traditionally, object detection models are trained on a single dataset, often restricted to a specific imaging modality. However, real-world applications increasingly demand a more versatile approach—one capable of detecting objects across diverse modalities. This paper introduces a new task called Multi-Modal Datasets and Multi-Task Object Detection (M3Det) for remote sensing, designed to accurately detect horizontal or oriented objects from any sensor modality. This task poses challenges due to the trade-offs involved in managing multi-modal data and the complexities of multi-task optimization. To address these, we establish a benchmark dataset and propose a unified model, SM3Det (Single Model for Multi-Modal datasets and Multi-Task object Detection) in remote sensing images. SM3Det leverages a sparse MoE backbone, allowing for joint knowledge learning while enabling distinct feature representation for different modalities. It also employs a dynamic optimization strategy to manage varying learning difficulties across tasks and modalities. Extensive experiments demonstrate the effectiveness and generalizability of SM3Det, consistently outperforming individual models on each dataset. 

## Introduction

## This repository is the official implementation of ArXiv "SM3Det: A Single Remote Sensing Object Detection Model for Multi-Modal Datasets and Multi-Task Joint Training" at: [Here]( )

The master branch is built on MMRotate which works with **PyTorch 1.6+**.

Main configuration files are put under configs/SM3Det/


## Results and models

![Main_results](docs/results.png)

<table><thead>
  <tr>
    <th>Model</th>
    <th>FLOPs</th>
    <th>#P</th>
    <th>Test on</th>
    <th>mAP</th>
    <th>@50</th>
    <th>@75</th>
  </tr></thead>
<tbody>
  <tr>
    <td>3 models</td>
    <td>403G</td>
    <td>126M</td>
    <td>Overall</td>
    <td>48.23</td>
    <td>79.39</td>
    <td>51.26</td>
  </tr>
  <tr>
    <td>GFL</td>
    <td>131G</td>
    <td>36M</td>
    <td>\scriptsizeSARDet-50K</td>
    <td>57.31</td>
    <td>87.44</td>
    <td>61.99</td>
  </tr>
  <tr>
    <td>O-RCNN</td>
    <td>136G</td>
    <td>45M</td>
    <td>\scriptsizeDOTA</td>
    <td>45.31</td>
    <td>77.70</td>
    <td>46.45</td>
  </tr>
  <tr>
    <td>O-RCNN</td>
    <td>136G</td>
    <td>45M</td>
    <td>\scriptsizeDroneVehicle</td>
    <td>46.09</td>
    <td>74.78</td>
    <td>52.79</td>
  </tr>
  <tr>
    <td>Simple<br>Joint<br>Training</td>
    <td>403G</td>
    <td>66M</td>
    <td>Overall</td>
    <td>47.05</td>
    <td>77.56</td>
    <td>50.11</td>
  </tr>
  <tr>
    <td>DA<br></td>
    <td>403G</td>
    <td>66M</td>
    <td>Overall</td>
    <td>48.37</td>
    <td>79.76</td>
    <td>51.66</td>
  </tr>
  <tr>
    <td>UniDet<br></td>
    <td>403G</td>
    <td>66M</td>
    <td>Overall</td>
    <td>48.47</td>
    <td>79.55</td>
    <td>52.01</td>
  </tr>
  <tr>
    <td>Uncertainty <br>loss</td>
    <td>403G</td>
    <td>66M</td>
    <td>Overall</td>
    <td>48.79</td>
    <td>79.99</td>
    <td>52.50</td>
  </tr>
  <tr>
    <td>SM3Det <br>lightweighted</td>
    <td>403G</td>
    <td>66M</td>
    <td>Overall</td>
    <td>49.40</td>
    <td>80.19</td>
    <td>52.93</td>
  </tr>
  <tr>
    <td>SM3Det</td>
    <td>487G</td>
    <td>178M</td>
    <td>Overall</td>
    <td>50.20</td>
    <td>80.68</td>
    <td>53.79</td>
  </tr>
</tbody></table>



![vis](docs/vis.png)



## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install -U openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/zcablii/SM3Det.git
cd SM3Det
pip install -v -e .
```

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
We provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)
 


## Acknowledgement

MMRotate is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@InProceedings{Li_2023_ICCV,
    author    = {Li, Yuxuan and Hou, Qibin and Zheng, Zhaohui and Cheng, Ming-Ming and Yang, Jian and Li, Xiang},
    title     = {Large Selective Kernel Network for Remote Sensing Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {16794-16805}
}
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.
