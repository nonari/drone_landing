#!/bin/bash

#SUAV_Ll_XXXX
#Ll-> Sigmoid | Softmax
#XXXX -> Loss


dataset_orig=datasets.ruralscapes.RuralscapesOrigSplit
dataset_segprop=datsets.ruralscapes.RuralscapesOrigSegprop

ruralroot=/home/xxx/tfm/ruralscapes_light
params="-override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
        -num_threads=4 -max_epochs=150 -delta=-0.01"


# Segprop augment
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset_segprop" \
-name=SUAV_BCEDA_seg_aug $params \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-augment=True \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss

# Segprop no augment
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset_segprop" \
-name=SUAV_BCEDA_seg_noaug $params \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-augment=False \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss

# Base augment
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset_orig" \
-name=SUAV_BCEDA_base_aug $params \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-augment=True \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss

# Base no augment
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset_orig" \
-name=SUAV_BCEDA_base_noaug $params \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-augment=False \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss
