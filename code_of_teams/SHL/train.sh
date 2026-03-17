#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python basicsr/train.py -opt options/Uformer.yml --auto_resume
CUDA_VISIBLE_DEVICES=1 python basicsr/train.py -opt options/restormer.yml --auto_resume
CUDA_VISIBLE_DEVICES=1 python basicsr/train.py -opt options/mstpp.yml --auto_resume

## 修改`option/*_step2.yml`中的`network_g->pretrained`权重路径为上一步迭代的最佳权重路径。执行第二次迭代
# CUDA_VISIBLE_DEVICES=1 python basicsr/train.py -opt options/mstpp_step2.yml --auto_resume
# CUDA_VISIBLE_DEVICES=1 python basicsr/train.py -opt options/restormer_step2.yml --auto_resume
# CUDA_VISIBLE_DEVICES=1 python basicsr/train.py -opt options/Uformer_step2.yml --auto_resume