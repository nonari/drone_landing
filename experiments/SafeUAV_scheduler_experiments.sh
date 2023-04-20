#!/bin/bash

#SUAV_Ll_XXXX
#Ll-> Sigmoid | Softmax
#XXXX -> Loss

#Class probability distribution 
#[0.24186207, 0.2048791, 0.28634467, 0.09043592, 0.01872263, 0.03582632, 0.08700141, 0.02159951, 0.00299932, 0.00311403, 0.00093436, 0.00628066]

dataset_rural=datasets.ruralscapes.RuralscapesOrigSplit
dataset_uavid=datasets.uavid.UAVid
dataset_tugraz=datasets.tugraz_sort.TUGrazSortedDataset

ruralroot=/home/xxx/tfm/ruralscapes_light
uavidroot=/home/xxx/tfm/uavid_v1.5
tugrazroot=/home/xxx/tfm/semantic_drone_dataset

params="-override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 -augment=True \
        -num_threads=4 -max_epochs=150 -delta=-0.01 -model_config.optimizer.params.lr=0.001 \
        -model_config.lr_scheduler.name=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts \
        -model_config.lr_scheduler.params.T_0=10 "

 
# TUgraz
python3 drone_landing/main.py train -tugraz_root="$tugrazroot" -dataset_name="$dataset_tugraz" \
-name=SUAV_tugraz $params \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss

 
# Uavid
python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset_uavid" \
-name=SUAV_uavid $params \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss


# Ruralscapes
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset_rural" \
-name=SUAV_ruralscapes $params \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss



# TESTS

python3 drone_landing/main.py test \
-rural_root="$tugrazroot" -model_config=safeuav_base \
-name=SUAV_tugraz -dataset_name="$dataset_tugraz"

python3 drone_landing/main.py test \
-rural_root="$uavidroot" -model_config=safeuav_base \
-name=SUAV_uavid -dataset_name="$dataset_uavid"

python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_rural -dataset_name="$dataset_rural"
