#!/bin/bash

#SUAV_Ll_XXXX
#Ll-> Sigmoid | Softmax
#XXXX -> Loss


dataset_ruralseg=datasets.ruralscapes.RuralscapesOrigSegprop
dataset_uavid=datasets.uavid.UAVid

ruralroot=/home/xxx/tfm/ruralscapes_light
uavidroot=/home/xxx/tfm/uavid_v1.5
params="-override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
        -num_threads=4 -max_epochs=200 -delta=-0.01 \
        -model_config.optimizer.params.lr=1e-4"


# UAVid
python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset_uavid" \
-name=uavid_base_noaug $params \
-model_config=safeuav_base \
-augment=False \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss


params="-override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=1 \
        -num_threads=4 -max_epochs=150 -delta=-0.01 \
        -model_config.optimizer.params.lr=1e-4"

# UAVid
python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset_uavid" \
-name=uavid_nb36 $params \
-model_config=safeuav_base \
-model_config.net.params.init_nb=36 \
-augment=True \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss

# Ruralscapes more nb
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset_ruralseg" \
-name=rural_nb36 $params \
-model_config=safeuav_base \
-model_config.net.params.init_nb=36 \
-augment=True \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss


python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=safeuav_base \
-name=uavid_base_noaug -dataset_name="$dataset_uavid"


python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=safeuav_base \
-name=uavid_nb36 -dataset_name="$dataset_uavid"


python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=rural_nb36 -dataset_name="$dataset_ruralseg"


