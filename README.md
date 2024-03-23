# OED
Official implementation of paper "OED: Towards One-stage End-to-End Dynamic Scene Graph Generation".

## Dataset
### Data preperation
We use the dataset Action Genome to train/evaluate our method. 
Please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome) and put the processed annotation [files](https://drive.google.com/drive/folders/1tdfAyYm8GGXtO2okAoH1WgVHVOTl1QYe?usp=share_link) with COCO style into `annotations` folder. 
The directories of the dataset should look like:
```
|-- action-genome
    |-- annotations   # gt annotations
        |-- ag_train_coco_style.json
        |-- ag_test_coco_style.json
        |-- ...
    |-- frames        # sampled frames
    |-- videos        # original videos
```

## Evaluation
Please download the [checkpoints](https://drive.google.com/drive/folders/12zh9ocGmbV8aOFPzUfp8ezP0pMTlpzJl?usp=sharing) used in the paper and put it into `exps` folder.
You can use the scripts below to evaluate the performance of OED.

In order to reduce the training cost, we firstly train the spatial module and then train the temporal module after loading trained spatial paramters.
+ For SGDET task
```
python scripts/eval_spatial_sgdet.py   # spatial module
python scripts/eval_temporal_sgdet.py  # temporal module
```
+ For PredCLS task
```
python scripts/eval_spatial_predcls.py   # spatial module
python scripts/eval_temporal_predcls.py  # temporal module
```

## Train
You can follow the scripts below to train OED in both SGDET and PredCLS tasks.

Notably, manually tuning LR drop may be needed to obtain the best performance.
+ For SGDET task
```
python scripts/train_spatial_sgdet.py   # spatial module
python scripts/trian_temporal_sgdet.py  # temporal module
```
+ For PredCLS task
```
python scripts/train_spatial_predcls.py   # spatial module
python scripts/train_temporal_predcls.py  # temporal module
```
