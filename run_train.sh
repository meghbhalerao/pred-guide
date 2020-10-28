#!/bin/sh
#CUDA_VISIBLE_DEVICES=0 python main_rot.py --dataset multi --target real --num 3 --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main_rot.py --dataset multi --target sketch --num 3 --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main_rot.py --dataset multi --target clipart --num 3 --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=0 python main_rot.py --dataset multi --target painting --num 3 --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset multi --source real --target sketch --num 3 --net resnet34 --pretrained_ckpt ./save_model_ssda/model_sketch_step_22500.pth.tar --save_check

#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset multi --source real --target clipart --num 3 --net resnet34 --pretrained_ckpt ./save_model_ssda/model_clipart_step_28000.pth.tar --save_check

#CUDA_VISIBLE_DEVICES=0 python main.py --method MME --dataset multi --source painting --target real --num 3 --net resnet34 --pretrained_ckpt ./save_model_ssda/model_real_step_21000.pth.tar --save_check

#CUDA_VISIBLE_DEVICES=0 python main.py --method MME --dataset multi --source real --target painting --num 3 --net resnet34 --pretrained_ckpt ./save_model_ssda/model_painting_step_22500.pth.tar --save_check
