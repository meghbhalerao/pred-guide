
CUDA_VISIBLE_DEVICES=1 python main_classwise.py \
--method MME \
--dataset multi \
--source real \
--target clipart \
--num 3 \
--use_new_features 0 \
--net alexnet \
--which_method SEW \
--patience 20 \
--data_parallel 0 \
--weigh_using target_acc \
--num_to_weigh 3 \
--save_interval 500 \
--log_interval 100 \
--label_target_iteration 8000 \
--SEW_iteration 2000 \
--SEW_interval 1000 \
--thresh 0.9 \
--phi 0.5 \


