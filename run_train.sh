#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python main_rot.py --dataset multi --target painting --num 3 --net resnet34 --save_check
