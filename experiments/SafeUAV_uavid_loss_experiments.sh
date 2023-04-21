#!/bin/bash

#SUAV_Ll_XXXX
#Ll-> Sigmoid | Softmax
#XXXX -> Loss

#Class probability distribution 
#[0.24186207, 0.2048791, 0.28634467, 0.09043592, 0.01872263, 0.03582632, 0.08700141, 0.02159951, 0.00299932, 0.00311403, 0.00093436, 0.00628066]

dataset=datasets.uavid.UAVid
uavidroot=/home/xxx/tfm/uavid_v1.5
weights="[1.72, 1., 2.47, 1.20, 2.17, 14.71, 1, 227.32]"
weightsone="[1., 1., 1., 1., 1., 1., 1., 1.]"
params="-override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
        -num_threads=4 -max_epochs=200 -delta=-0.01 -model_config.optimizer.params.lr=0.001"

# BCEDice Sigmoid
python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=SUAV_Si_BCED $params \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.BCEDiceLoss
 
# BCEDiceAvg Sigmoid
python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=SUAV_Si_BCEDA $params \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss
 
 # BCEDice Softmax
python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=SUAV_So_BCED $params \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.BCEDiceLoss
 
# BCEDiceAvg Softmax
python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=SUAV_So_BCEDA $params \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss

# CEWeightDiceAvgLoss Softmax
python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=SUAV_So_CEWDA $params \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.CEWeightDiceAvgLoss \
-model_config.loss.params.w="$weights" \
-model_config.loss.params.config=!CONFIG


# CEWeightDiceAvgLoss Sigmoid
python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=SUAV_Si_CEWDA $params \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.CEWeightDiceAvgLoss \
-model_config.loss.params.w="$weights" \
-model_config.loss.params.config=!CONFIG


# CEWeightDiceLoss Softmax
python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=SUAV_So_CEWD $params \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.CEWeightDiceLoss \
-model_config.loss.params.w="$weights" \
-model_config.loss.params.config=!CONFIG


# CEWeightDiceLoss Sigmoid
python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=SUAV_Si_CEWD $params \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.CEWeightDiceLoss \
-model_config.loss.params.w="$weights" \
-model_config.loss.params.config=!CONFIG

# CELoss Softmax
python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=SUAV_So_CE $params \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.CELoss


# CELoss Sigmoid
python3 drone_landing/main.py train -uavid_root="$uavidroot" -dataset_name="$dataset" \
-name=SUAV_Si_CE $params \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.CELoss


# TESTS

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=safeuav_base \
-name=SUAV_Si_BCED -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=safeuav_base \
-name=SUAV_Si_BCEDA -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=safeuav_base \
-name=SUAV_So_BCED -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=safeuav_base \
-name=SUAV_So_BCEDA -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=safeuav_base \
-name=SUAV_So_CEWDA -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=safeuav_base \
-name=SUAV_Si_CEWDA -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=safeuav_base \
-name=SUAV_So_CEWD -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=safeuav_base \
-name=SUAV_Si_CEWD -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=safeuav_base \
-name=SUAV_So_CE -dataset_name="$dataset"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=safeuav_base \
-name=SUAV_Si_CE -dataset_name="$dataset"
