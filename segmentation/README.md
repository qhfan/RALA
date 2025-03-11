## Installation

1. 

```bash
bash install.sh
```

2. Download ADE20K dataset from the [official website](https://groups.csail.mit.edu/vision/datasets/ADE20K/). The directory structure should look like

   ```
   ade
   └── ADEChallengeData2016
       ├── annotations
       │   ├── training
       │   └── validation
       └── images
           ├── training
           └── validation
   ```



## Training

To train a model, run:

```bash
bash tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.pretrained=<PRETRAIN_MODEL>
```


## Benchmark

To get the FLOPs, run

```bash
python tools/get_flops.py configs/.../....py
```

