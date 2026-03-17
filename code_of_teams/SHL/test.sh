#!/bin/bash

python test_crop.py \
    --input_dir /home/notebook/data/personal/S9059954/HDR-Competition/datasets/test-stage2-wo-gt \
    --output_dir output_Uformer \
    --weights checkpoints/uformer_56.87_net_g_38000.pth \
    --opt options/Uformer.yml \
    --crop_size 1528 \
    --crop_shave 128 \
    --factor 16 --save_suffix png


python test_crop.py \
    --input_dir /home/notebook/data/personal/S9059954/HDR-Competition/datasets/test-stage2-wo-gt \
    --output_dir output_restormer \
    --weights checkpoints/restormer_57.19_net_g_25000.pth \
    --opt options/restormer.yml \
    --crop_size 1528 \
    --crop_shave 128 \
    --factor 16 --save_suffix png


python test_crop.py \
    --input_dir /home/notebook/data/personal/S9059954/HDR-Competition/datasets/test-stage2-wo-gt \
    --output_dir output_mstpp \
    --weights checkpoints/mstpp_58.03_net_g_25000.pth \
    --opt options/mstpp.yml \
    --crop_size 1528 \
    --crop_shave 128 \
    --factor 16 --save_suffix png

python postdeal.py