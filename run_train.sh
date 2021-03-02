#!/bin/sh
#export CUDA_VISIBLE_DEVICES=1
#CUDA_VISIBLE_DEVICES=1 python main_match.py --method ENT --dataset multi --source real --target sketch --num 3 --net resnet34 --save_check

#CUDA_VISIBLE_DEVICES=0,1 python main_match.py --method MME --dataset multi --source real --target sketch --num 3 --net resnet34 --augmentation_policy rand_augment --save_check
#CUDA_VISIBLE_DEVICES=0,1 python main_max_acc.py --method MME --dataset multi --source real --target sketch --num 3 --net resnet34 --augmentation_policy rand_augment --save_check

#CUDA_VISIBLE_DEVICES=0,1 python main_match_majvot.py --method MME --dataset multi --source real --target sketch --num 3 --net resnet34 --augmentation_policy rand_augment --save_check

CUDA_VISIBLE_DEVICES=0 python main_classwise.py \
--method MME \
--dataset office_home \
--source Art \
--target Real \
--num 3 \
--net alexnet \
--augmentation_policy rand_augment \
--which_method SEW \
--uda 1 \
--use_bank 1 \
--use_new_features 0 \
--patience 10 \
--data_parallel 0  \
--weigh_using target_acc \
--num_to_weigh 1 \
--save_interval 80
--label_target_iteration 1000 \
--SEW_iteration 280 \
--SEW_interval 140 \
--thresh 0.9 \
--save_check

#CUDA_VISIBLE_DEVICES=0,1 python main_classwise.py --method MME --dataset multi --source real --target painting --num 3 --net resnet34 --augmentation_policy rand_augment --which_method SEW --uda 1 --use_bank 1 --use_cb 0 --use_new_features 0 --patience 5 --data_parallel 1 --weigh_using target_acc --num_to_weigh 5 --save_check


#CUDA_VISIBLE_DEVICES=0,1 python main_match_knn_analysis.py --method MME --dataset multi --source real --target sketch --num 3 --net resnet34 --augmentation_policy rand_augment --save_check

#CUDA_VISIBLE_DEVICES=0,1 python main_match_confident_source.py --method MME --dataset multi --source real --target sketch --num 3 --net resnet34 --augmentation_policy rand_augment --save_check

# Useful command to kill all nvidia processes
#fuser -v /dev/nvidia*

#CUDA_VISIBLE_DEVICES=0,1 python main_match_bank_analysis.py --method MME --dataset multi --source real --target sketch --num 3 --net resnet34 --augmentation_policy rand_augment --save_check

#CUDA_VISIBLE_DEVICES=1 python main_rot.py --dataset multi --target sketch --num 3 --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main_rot.py --dataset multi --target clipart --num 3 --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=0 python main_rot.py --dataset multi --target painting --num 3 --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset multi --source real --target sketch --num 3 --net resnet34 --pretrained_ckpt ./save_model_ssda/model_sketch_step_22500.pth.tar --save_check

#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset multi --source real --target clipart --num 3 --net resnet34 --pretrained_ckpt ./save_model_ssda/model_clipart_step_28000.pth.tar --save_check

#CUDA_VISIBLE_DEVICES=0 python main.py --method MME --dataset multi --source painting --target real --num 3 --net resnet34 --pretrained_ckpt ./save_model_ssda/model_real_step_21000.pth.tar --save_check

#CUDA_VISIBLE_DEVICES=0 python main.py --method MME --dataset multi --source real --target painting --num 3 --net resnet34 --pretrained_ckpt ./save_model_ssda/model_painting_step_22500.pth.tar --save_check
