#!/bin/bash

#SUAV_Ll_XXXX
#Ll-> Sigmoid | Softmax
#XXXX -> Loss


dataset_rural=datasets.ruralscapes.RuralscapesOrigSplit
dataset_uavid=datsets.uavid.UAVid
dataset_tugraz=datasets.tugraz_sort.TUGrazSortedDataset

ruralroot=/home/xxx/tfm/ruralscapes_light
uavidroot=/home/xxx/tfm/uavid_v1.5
tugrazroot=/home/xxx/tfm/semantic_drone_dataset
params="-override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
        -num_threads=4 -max_epochs=150 -delta=-0.01 \
        -model_config.optimizer.params.lr=1e-4"

weights="[1.08, 1.39, 1., 3.16, 15.29, 7.99, 3.29, 1, 95.46, 91.95, 306.46, 45.59]"

# Ruralscapes more nb
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset_rural" \
-name=rural_nb48 $params \
-model_config.net.params.init_nb=48 \
-model_config=safeuav_base \
-augment=True \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss

# Ruralscapes BCELL
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset_rural" \
-name=rural_bcell $params \
-model_config=safeuav_base \
-augment=True \
-model_config.net.params.last=identity \
-model_config.loss.name=custom_models.losses.BCELL \
-model_config.loss.params.config=!CONFIG \
-model_config.loss.params.w="$weights"


# UAVid
python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset_uavid" \
-name=uavid_base $params \
-model_config=safeuav_base \
-augment=True \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss

# Tugraz sorted
python3 drone_landing/main.py train -tugraz_root="$tugrazroot" -dataset_name="$dataset_tugraz" \
-name=tugraz_base $params \
-model_config=safeuav_base \
-augment=True \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss


python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=rural_nb48 -dataset_name="$dataset_rural"


python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=rural_bcell -dataset_name="$dataset_rural"


python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=uavid_base -dataset_name="$dataset_uavid"


python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=tugraz_base -dataset_name="$dataset_tugraz"
