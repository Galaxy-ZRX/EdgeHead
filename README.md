# EdgeHead - Detect Closer Surfaces That Can be Seen: New Modeling and Evaluation in Cross-Domain 3D Object Detection

This repo is the implementation of the [EdgeHead](https://arxiv.org/abs/2407.04061) project accepted by ECAI 2024. Codes are based on the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 

**Authors:** Ruixiao Zhang, Yihong Wu, Juheon Lee, Xiaohao Cai, Adam Prugel-Bennett

## Abstract

The performance of domain adaptation technologies has not yet reached an ideal level in the current 3D object detection field for autonomous driving, which is mainly due to significant differences in the size of vehicles, as well as the environments they operate in when applied across domains. These factors together hinder the effective transfer and application of knowledge learned from specific datasets. Since the existing evaluation metrics are initially designed for evaluation on a single domain by calculating the 2D or 3D overlap between the prediction and ground-truth bounding boxes, they often suffer from the overfitting problem caused by the size differences among datasets. This raises a fundamental question related to the evaluation of the 3D object detection models’ cross-domain performance: Do we really need models to maintain excellent performance in their original 3D bounding boxes after being applied across domains? From a practical application perspective, one of our main focuses is actually on preventing collisions between vehicles and other obstacles, especially in cross-domain scenarios where correctly predicting the size of vehicles is much more difficult. In other words, as long as a model can accurately identify the closest surfaces to the ego vehicle, it is sufficient to effectively avoid obstacles. In this paper, we propose two metrics to measure 3D object detection models’ ability of detecting the closer surfaces to the sensor on the ego vehicle, which can be used to evaluate their cross-domain performance more comprehensively and reasonably. Furthermore, we propose a refinement head, named EdgeHead, to guide models to focus more on the learnable closer surfaces, which can greatly improve the cross-domain performance of existing models not only under our new metrics, but even also under the original BEV/3D metrics.


## To reproduce the results:

1. Install OpenPCDet and prepare datasets following the official instructions. 
   (Tested version: PCDet 0.6.0)

2. Train the original models first - such as SECOND, CenterPoint, etc.
```
cd tools
python train.py --cfg_file cfgs/waymo_kitti_models_edgehead/centerpoint.yaml --ckpt_save_interval 30
```

3. Train the edgehead with original model frozen by setting "--roihead_only True":

```
python train.py --cfg_file cfgs/waymo_kitti_models_edgehead/centerpoint_edgehead.yaml --ckpt_save_interval 30 --roihead_only True --pretrained_model ../output/waymo_kitti_models_edgehead/centerpoint/default/ckpt/checkpoint_epoch_30.pth

```

4. Eval the model:
```
python test.py --cfg_file cfgs/waymo_kitti_models_edgehead/centerpoint_edgehead.yaml --batch_size 8 --eval_all
```

To eval cross-domain performance, uncomment the "DATA_CONFIG_TAR" in model configs.


## Acknowledgement

Thanks to the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) codebase. Thanks to [SECOND](https://github.com/traveller59/second.pytorch), [CenterPoint](https://github.com/tianweiy/CenterPoint), [PV-RCNN](https://github.com/sshaoshuai/PV-RCNN) and [VoxelRCNN](https://github.com/djiajunustc/Voxel-R-CNN) for their valuable inspiration.

## Citation
If you find this work useful in your research, please consider cite:
```
@incollection{zhang2024detect,
  title={Detect closer surfaces that can be seen: New modeling and evaluation in cross-domain 3d object detection},
  author={Zhang, Ruixiao and Wu, Yihong and Lee, Juheon and Cai, Xiaohao and Prugel-Bennett, Adam},
  booktitle={ECAI 2024},
  pages={65--72},
  year={2024},
  publisher={IOS Press}
}
```