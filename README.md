# Pred&Guide - Label Target Class Prediction for Guiding Semi-supervised Domain Adaptation
![alt text](https://user-images.githubusercontent.com/33202960/134644560-2bcf9d33-3251-4b5a-bda4-da16a5ec114b.png)
## Brief Description
This repository contains the codes and instructions to run the codes for Pred&Guide.
## Steps to run the code
### Dataset Download
1. To download the **DomainNet** Dataset, run `bash download_data.sh`. You will need to modify it a little to download the domains which you specifically want. Please follow the point below to download all the domains at once.
2. Alternatively **DomainNet** Dataset can be downloaded from [here](http://ai.bu.edu/M3SDA/#dataset). 
3. The **Office-Home** data can be downloaded from [here](https://www.hemanthdv.org/officeHomeDataset.html). 
3. Place the downloaded datasets into either `./data/multi` for DomainNet and `./data/office-home` for Office-Home.
### Dataset format 
1. The datasets are stored in the format - `./data/$dataset_name/$domain_name/$category_name/$image_name.png`
2. The `./data/txt/` folder contains the image lists used for training, validation and testing purposes.
3. `./data/txt/multi/labeled_source_images_real.txt` indicates real being used as a source and the images are all labeled.
4.  `./data/txt/multi/unlabeled_target_images_real_3.txt` indicates the real begin used as a target domain in a 3 shot setting. This file would contain all labeled target images except for the 3 labeled images per class. 
5. `./data/txt/multi/labeled_target_images_real_3.txt` would contain the 3 labeled target images of real domain, when real domain is being used as a target domain in the experiment. 
6. The same will apply for the **1 shot setting** except that 3 would be replaced by 1. 
7. The **Office-Home** dataset is also formatted in the same way as the **DomainNet** Dataset. 
8. In points 3 and 4, `multi` is the name given to the `DomainNet` dataset. This is done according to the previous baselines.
### Experiment and Producing Results
1. The `main_classwise.py` is the main file to train the domain adaptation model. The same script also prints out the results which we report, and hence no other script is needed to run for testing/inference.
2. Run the following command and change the hyper-parameters according to your requirements - 
```
python main_classwise.py \ # main file to run
--method MME \ # SSDA Method, Options - ENT, S+T
--dataset multi \  # SSDA Dataset, Options - multi, office-home
--source real \ # Source Domain, varies according to the dataset
--target sketch \ # Target Domain, varies according to the dataset
--num 3 \ # Number of Target Examples to be chosen, Options - 1, 3
--net resnet34 \ # Backbone architecture, Options - resnet34, alexnet
--which_method SEW \ # Whether to perform source example weighing
--patience 10 \ # Training patience
--data_parallel 0 \ # Whether to use parallel GPU training
--weigh_using target_acc \ # Which weighing scheme to follow
--num_to_weigh 2 \ # Number of source examples per class to be weighed 
--save_interval 500 \ # Iteration interval after which model to be saved
--log_interval 100 \ # Iteration interval after which training status to be logged
--label_target_iteration 8000 \ # Iteration after which labeled target examples brought in training
--SEW_iteration 2000 \ # Iteration after which source example weighing is started
--SEW_interval 1000 \ # Iterations after which source examples are to be reweighed
--thresh 0.9 \ # Pseudo labeling confidence threshold
--phi 0.5 \ # Parameter in the source weighing formula
--save_check # Whether to save the model weights
```
3. After running this script your model checkpoints will be saved in `./save_model_ssda` folder and the accuracy will be printed out.
4. Just a note, for `Office-Home` dataset you will have to reduce some of the iterations and intervals to smaller values (scale-down by a factor of **7** approximately since it is smaller then `DomainNet` approximately and on-average by a factor of 7) to get reproducible results.
## Dependenceis
 - [`pytorch v1.7.0`](https://pytorch.org)

## Other issues
Please feel free to post any issues, questions or concerns regarding the code.
