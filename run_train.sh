#!/bin/sh
#CUDA_VISIBLE_DEVICES=0 python main_rot.py --dataset multi --target real --num 3 --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main_rot.py --dataset multi --target sketch --num 3 --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main_rot.py --dataset multi --target clipart --num 3 --net resnet34 --save_check
CUDA_VISIBLE_DEVICES=0 python main_rot.py --dataset multi --target painting --num 3 --net resnet34 --save_check
