## This repository is the official implementation of ArXiv "[SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](http://arxiv.org/abs/2412.20665 )"

![meme](docs/meme.png)


## [Unifying Heterogeneous Multi-Modal Remote Sensing Detection Via language-pivoted pretraining](https://github.com/zcablii/SM3Det/blob/main/docs/BabelRS.pdf)

Heterogeneous multi-modal remote sensing object detection aims to accurately detect objects from diverse sensors (e.g., RGB, SAR, Infrared). Existing approaches largely adopt a late alignment paradigm, in which modality alignment and task-specific optimization are entangled during downstream fine-tuning. This tight coupling complicates optimization and often results in unstable training and suboptimal generalization. To address these limitations, we propose BabelRS, a unified language-pivoted pretraining framework that explicitly decouples modality alignment from downstream task learning. BabelRS comprises two key components: Concept-Shared Instruction Aligning (CSIA) and Layerwise Visual-Semantic Annealing (LVSA). CSIA aligns each sensor modality to a shared set of linguistic concepts, using language as a semantic pivot to bridge heterogeneous visual representations. To further mitigate the granularity mismatch between high-level language representations and dense detection objectives, LVSA progressively aggregates multiscale visual features to provide fine-grained semantic guidance. Extensive experiments demonstrate that BabelRS stabilizes training and consistently outperforms state-of-the-art methods without bells and whistles.

![net_arch](docs/BabelRS_motivation.png)


![net_arch](docs/BabelRS.png)

## BabelRS Model pretraining and finetuning
Main code and configuration files are put under BabelRS_pretrain/ and BabelRS_configs/

* [Pretrained BabelRS-ViT-Large](https://huggingface.co/datasets/GreatBird/BabelRS/resolve/main/ckpts/BabelRS_ViT-300M.safetensors?download=true) 


## [SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](http://arxiv.org/abs/2412.20665 )

With the rapid advancement of remote sensing technology, high-resolution multi-modal imagery is now more widely accessible. Conventional Object detection models are trained on a single dataset, often restricted to a specific imaging modality and annotation format. However, such an approach overlooks the valuable shared knowledge across multi-modalities and limits the model's applicability in more versatile scenarios. This paper introduces a new task called Multi-Modal Datasets and Multi-Task Object Detection (M2Det) for remote sensing, designed to accurately detect horizontal or oriented objects from any sensor modality. This task poses challenges due to 1) the trade-offs involved in managing multi-modal modelling and 2) the complexities of multi-task optimization. To address these, we establish a benchmark dataset and propose a unified model, SM3Det (Single Model for Multi-Modal datasets and Multi-Task object Detection). SM3Det leverages a grid-level sparse MoE backbone to enable joint knowledge learning while preserving distinct feature representations for different modalities. Furthermore, it integrates a consistency and synchronization optimization strategy using dynamic learning rate adjustment, allowing it to effectively handle varying levels of learning difficulty across modalities and tasks. Extensive experiments demonstrate SM3Det's effectiveness and generalizability, consistently outperforming specialized models on individual datasets.

![net_arch](docs/SM3Det.png)

## SM3Det Model 

**Model Architecture:**

- We propose integrating a plug-and-play grid-level sparse Mixture of Experts (MoE) architecture into backbone networks, enabling the model to capture both shared knowledge and modality-specific representations. Through dynamic routing, the experts operate on local spatial features, allowing the model to adaptively process information at a grid level, which is crucial for object detection tasks. 

**Model Optimization:**

- We propose a novel Dynamic Learning Rate Adjustment (DLA) method that adaptively adjusts the learning rates of different network components with tailored policies. DLA accommodates the varying learning complexities across different tasks and modalities by balancing the relative convergence rate and guaranteeing optimization direction consistency. 
Unlike traditional techniques that primarily modify loss weights or gradients—often lacking precise manipulation over specific network submodules or suffering from inefficiencies—our DLA provides fine-grained control while maintaining optimization efficiency.

Main configuration files are put under configs/SM3Det/


**SOI-Det DATASET DOWNLOAD at:** 

* [Dataset](https://www.kaggle.com/datasets/greatbird/soi-det) 

## Results and models

![Main_results](docs/results.png)

-----

![vis](docs/vis.png)

-----

<table><thead>
  <tr>
    <th>Model</th>
    <th>FLOPs</th>
    <th>#P</th>
    <th>Test on</th>
    <th>@50</th>
    <th>mAP</th>
    <th>H-mAP</th>
    <th>config</th>
    <th>log/ckpt</th>
  </tr></thead>
<tbody>
  <tr>
    <td>3 models</td>
    <td>403G</td>
    <td>126M</td>
    <td>Overall</td>
    <td>79.39</td>
    <td>48.23</td>
    <td>49.01</td>
    <td><a href="local_configs/sardet50k_convnext_t_gfl.py"> 1 </a> <a href="local_configs/dota_convnext_t_orcnn.py"> 2 </a> <a href="local_configs/dronevehicle_convnext_t_orcnn.py"> 3 </a> </td>
    <td><a href="https://www.kaggle.com/models/greatbird/sm3det"> here </a></td>
  </tr>
  <tr>
    <td>Simple<br>Joint<br>Training</td>
    <td>403G</td>
    <td>66M</td>
    <td>Overall</td>
    <td>77.56</td>
    <td>47.05</td>
    <td>47.57</td>
    <td><a href="main_convnext_t_orcnn_gfl_simple_joint.py"> here </a></td>
    <td><a href="https://www.kaggle.com/models/greatbird/sm3det"> here </a></td>
  </tr>
  <tr>
    <td>DA<br></td>
    <td>403G</td>
    <td>66M</td>
    <td>Overall</td>
    <td>79.76</td>
    <td>48.37</td>
    <td>49.23</td>
    <td><a href="local_configs/main_DA_convnext_t_orcnn_gfl.py"> here </a></td>
    <td><a href="https://www.kaggle.com/models/greatbird/sm3det"> here </a></td>
  </tr>
  <tr>
    <td>UniDet<br></td>
    <td>403G</td>
    <td>66M</td>
    <td>Overall</td>
    <td>79.55</td>
    <td>48.47</td>
    <td>49.24</td>
    <td><a href="local_configs/main_unidet_convnext_t_orcnn_gfl.py"> here </a></td>
    <td><a href="https://www.kaggle.com/models/greatbird/sm3det"> here </a></td>
  </tr>
  <tr>
    <td>Uncertainty <br>loss</td>
    <td>403G</td>
    <td>66M</td>
    <td>Overall</td>
    <td>79.99</td>
    <td>48.79</td>
    <td>49.57</td>
    <td><a href="local_configs/main_uncertainty_convnext_t_orcnn_gfl.py"> here </a></td>
    <td><a href="https://www.kaggle.com/models/greatbird/sm3det"> here </a></td>
  </tr>
  <tr>
    <td>SM3Det <br>lightweighted</td>
    <td>403G</td>
    <td>66M</td>
    <td>Overall</td>
    <td>80.19</td>
    <td>49.40</td>
    <td>50.39</td>
    <td><a href="local_configs/main_SM3Det_convnext_t_orcnn_gfl_wo_moe.py"> here </a></td>
    <td><a href="https://www.kaggle.com/models/greatbird/sm3det"> here </a></td>
  </tr>
  <tr>
    <td>SM3Det</td>
    <td>487G</td>
    <td>178M</td>
    <td>Overall</td>
    <td>80.68</td>
    <td>50.20</td>
    <td>51.31</td>
    <td><a href="configs/SM3Det_convnext_t.py"> here </a></td>
    <td><a href="https://www.kaggle.com/models/greatbird/sm3det"> here </a></td>
  </tr>
  <tr>
    <td>BabelRS</td>
    <td>3395G</td>
    <td>738M</td>
    <td>Overall</td>
    <td>81.32</td>
    <td>51.57</td>
    <td>53.02</td>
    <td><a href="BabelRS_configs/BabelRS_20kstep.py"> here </a></td>
    <td><a href="https://huggingface.co/datasets/GreatBird/BabelRS/tree/main/ckpts/BabelRS_20kstep"> here </a></td>
  </tr>
</tbody></table>




## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

# Usage
- Clone this repository:
```
git clone https://github.com/zcablii/SM3Det.git
```
## Object Detection
### Installation

- Create a conda environment:
```
cd SM3Det
conda create -n SM3Det python==3.10
conda activate SM3Det
```

- Install the required packages:
```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install -r requirements.txt
```

- Install mmcv:
```
cd ../mmcv
python setup.py install
cd ../mmrotate
```
- Install mmrotate:
```
pip install -e .
```

### Train
```
sh ./tools/dist_train.sh BabelRS_configs/BabelRS_20kstep.py 8 
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
@inproceedings{Li_2026_sm3det,
    author    = {Li, Yuxuan and Li, Xiang and Li, Yunheng and Zhang Yicheng and Dai, Yimian and Hou, Qibin and Cheng, Ming-Ming and Yang, Jian},
    title     = {SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection},
    booktitle = {AAAI},
    year      = {2026}
}
```

```bibtex
@inproceedings{Li_2026_babelrs,
    author    = {Li, Yuxuan and Chen, Yuming and Li, Yunheng and Ming-Ming and Li, Xiang and Yang, Jian},
    title     = {Unifying Heterogeneous Multi-Modal Remote Sensing Detection Via Language-Pivoted Pretraining},
    booktitle = {Arxiv},
    year      = {2026}
}
```
## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.
