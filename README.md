# EdgeHead - Detect Closer Surfaces That Can be Seen: New Modeling and Evaluation in Cross-Domain 3D Object Detection

This repo is the implementation of the [EdgeHead](https://arxiv.org/abs/2407.04061) project accepted by ECAI 2024. Codes are based on the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 


## To reproduce the results:

1. Install OpenPCDet and prepare datasets following the official instructions. 
   (Tested version: PCDet 0.6.0)

2. Train the original models first - such as SECOND, CenterPoint, etc.
```
cd tools
python train.py --cfg_file cfgs/waymo_kitti_models_edgehead/centerpoint.yaml --ckpt_save_interval 30
```

1. Train the edgehead with original model frozen by setting "--roihead_only True":

```
python train.py --cfg_file cfgs/waymo_kitti_models_edgehead/centerpoint_edgehead.yaml --ckpt_save_interval 30 --roihead_only True --pretrained_model ../output/waymo_kitti_models_edgehead/centerpoint/default/ckpt/checkpoint_epoch_30.pth

```

4. Eval the model:
```
python test.py --cfg_file cfgs/waymo_kitti_models_edgehead/centerpoint_edgehead.yaml --batch_size 8 --eval_all
```

To eval cross-domain performance, uncomment the "DATA_CONFIG_TAR" in model configs.
