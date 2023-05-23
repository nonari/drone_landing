#!/bin/bash

#SUAV_Ll_XXXX
#Ll-> Sigmoid | Softmax
#XXXX -> Loss

#Class probability distribution 
#[0.24186207, 0.2048791, 0.28634467, 0.09043592, 0.01872263, 0.03582632, 0.08700141, 0.02159951, 0.00299932, 0.00311403, 0.00093436, 0.00628066]

dataset=datasets.uavid.UAVid
uavidroot=/home/xxx/tfm/uavid_v1.5

weights="[1.72, 1., 2.47, 1.20, 2.17, 14.71, 1, 50.32]"
weightsone="[1., 1., 1., 1., 1., 1., 1., 1.]"
params="-override=True -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
        -num_threads=4 -max_epochs=200 -delta=-0.01 -model_config=base -model_config.net.name=unet \
        -model_config.net.params.encoder=resnet18 -model_config.net.params.last=identity"


python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=uavid_bcelldice_old $params \
-model_config.optimizer.params.lr=0.001 \
-model_config.loss.name=custom_models.losses.BCELLDiceAvgOld

python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=uavid_bcelldice_512 $params -model_config.size=[512,736] \
-model_config.optimizer.params.lr=0.001 \
-model_config.loss.name=custom_models.losses.BCELLDiceAvg

python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=uavid_bcelldice_416 $params -model_config.size=[416,608] \
-model_config.optimizer.params.lr=0.001 \
-model_config.loss.name=custom_models.losses.BCELLDiceAvg

python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=uavid_bcelldice $params \
-model_config.optimizer.params.lr=0.001 \
-model_config.loss.name=custom_models.losses.BCELLDiceAvg

python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=uavid_bcelldice_lowlr $params \
-model_config.optimizer.params.lr=0.0001 \
-model_config.loss.name=custom_models.losses.BCELLDiceAvg

python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=uavid_focal $params \
-model_config.optimizer.params.lr=0.001 \
-model_config.loss.name=custom_models.losses.FocalDiceAvg

python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=uavid_focalw $params \
-model_config.loss.name=custom_models.losses.FocalDiceAvg \
-model_config.optimizer.params.lr=0.001 \
-model_config.loss.params.w="$weights" -model_config.loss.params.config=!CONFIG

python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=uavid_bcelldice_w $params \
-model_config.loss.name=custom_models.losses.BCELLWDiceAvg \
-model_config.optimizer.params.lr=0.001 \
-model_config.loss.params.w="$weights" -model_config.loss.params.config=!CONFIG


# TESTS

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_bcelldice_old -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_bcelldice -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_bcelldice -dataset_name="$dataset"

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
-name=uavid_bcelldice_512 -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_bcelldice_416 -dataset_name="$dataset"