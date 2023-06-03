#!/bin/bash

dataset=datasets.uavid.UAVid
uavidroot=/home/xxx/tfm/uavid_v1.5

weights="[1.72, 1., 2.47, 1.20, 2.17, 14.71, 1, 40.32]"
params="-override=True -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
        -num_threads=4 -max_epochs=200 -delta=-0.01 -model_config=base \
        -model_config.optimizer.params.lr=0.0001"


python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=uavid_bcelldice_old $params -model_config.net.name=unet -model_config.net.params.encoder_name=resnet18 \
-model_config.loss.name=custom_models.losses.BCELLDiceAvgOld

python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=uavid_focal $params -model_config.net.name=unet -model_config.net.params.encoder_name=resnet18 \
-model_config.loss.name=custom_models.losses.FocalDiceAvg

python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=uavid_focalw $params -model_config.net.name=unet -model_config.net.params.encoder_name=resnet18 \
-model_config.loss.name=custom_models.losses.FocalDiceAvg \
-model_config.loss.params.w="$weights" -model_config.loss.params.config=!CONFIG

python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=uavid_bcelldice_w $params -model_config.net.name=unet -model_config.net.params.encoder_name=resnet18 \
-model_config.loss.name=custom_models.losses.BCELLWDiceAvg \
-model_config.loss.params.w="$weights" -model_config.loss.params.config=!CONFIG

python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=uavid_bcelldice_mobilenet $params -model_config.net.name=unet -model_config.net.params.encoder_name=mobilenet_v2 \
-model_config.loss.name=custom_models.losses.BCELLDiceAvg


# TESTS

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_bcelldice_old -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_focal -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_focalw -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_bcelldice_w -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_bcelldice_mobilenet -dataset_name="$dataset"
