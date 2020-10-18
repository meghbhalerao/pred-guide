#!/bin/sh
CUDA_VISIBLE_DEVICES=1 python main_rot.py --dataset multi --target sketch --net resnet34 --save_check
