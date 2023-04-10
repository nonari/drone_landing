#!/bin/bash

#SUAV_Ll_XXXX
#Ll-> Sigmoid | Softmax
#XXXX -> Loss

#Class probability distribution
#[0.24186207, 0.2048791, 0.28634467, 0.09043592, 0.01872263, 0.03582632, 0.08700141, 0.02159951, 0.00299932, 0.00311403, 0.00093436, 0.00628066]

dataset=datasets.ruralscapes.RuralscapesOrigSplit
ruralroot=/home/xxx/tfm/ruralscapes_light
weights="[0.04134588,0.04880927,0.03492295,0.11057553,0.534113,0.2791244,0.11494067,0.46297347,1,1,1,1]"
params="-override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
        -num_threads=4 -max_epochs=150 -delta=-0.01 \
        -model_config.optimizer.params.eps=1e-8 \
        -model_config.optimizer.params.alpha=0.99 \
        -model_config.optimizer.params.lr=1e-3"

# BCEDice Sigmoid
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_Si_BCED_lr "$params" \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.BCEDiceLoss

# BCEDiceAvg Sigmoid
#python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
#-name=SUAV_Si_BCEDA_lr "$params" \
#-model_config=safeuav_base \
#-model_config.net.params.last=sigmoid \
#-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss

 # BCEDice Softmax
#python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
#-name=SUAV_So_BCED_lr "$params" \
#-model_config=safeuav_base \
#-model_config.net.params.last=softmax \
#-model_config.loss.name=custom_models.losses.BCEDiceLoss

# BCEDiceAvg Softmax
#python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
#-name=SUAV_So_BCEDA_lr "$params" \
#-model_config=safeuav_base \
#-model_config.net.params.last=softmax \
#-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss


# DiceAvg Softmax
#python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
#-name=SUAV_So_DA_lr "$params" \
#-model_config=safeuav_base \
#-model_config.net.params.last=softmax \
#-model_config.loss.name=custom_models.losses.DiceAvgLoss


# DiceAvg Sigmoid
#python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
#-name=SUAV_Si_DA_lr "$params" \
#-model_config=safeuav_base \
#-model_config.net.params.last=sigmoid \
#-model_config.loss.name=custom_models.losses.DiceAvgLoss


# Dice Sigmoid
#python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
#-name=SUAV_Si_D_lr "$params" \
#-model_config=safeuav_base \
#-model_config.net.params.last=sigmoid \
#-model_config.loss.name=custom_models.losses.DiceAvgLoss

# Dice Softmax
#python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
#-name=SUAV_So_D_lr "$params" \
#-num_threads=4 -max_epochs=150 -delta=-0.01 \
#-model_config=safeuav_base \
#-model_config.net.params.last=softmax \
#-model_config.loss.name=custom_models.losses.DiceLoss


# CEWeightDiceAvgLoss Softmax
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_So_CEWDA_lr "$params" \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.CEWeightDiceAvgLoss \
-model_config.loss.params.w="$weights" \
-model_config.loss.params.config=!CONFIG


# CEWeightDiceAvgLoss Sigmoid
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_Si_CEWDA_lr "$params" \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.CEWeightDiceAvgLoss \
-model_config.loss.params.w="$weights" \
-model_config.loss.params.config=!CONFIG


# CEWeightDiceLoss Softmax
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_So_CEWD_lr "$params" \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.CEWeightDiceLoss \
-model_config.loss.params.w="$weights" \
-model_config.loss.params.config=!CONFIG


# CEWeightDiceLoss Sigmoid
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_Si_CEWD_lr "$params" \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.CEWeightDiceLoss \
-model_config.loss.params.w="$weights" \
-model_config.loss.params.config=!CONFIG


# CELoss Softmax
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_So_CE_lr "$params" \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.CELoss


# CELoss Sigmoid
python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_Si_CE_lr "$params" \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.CELoss


# TESTS

python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_Si_BCED_lr -dataset_name="$dataset"

#python3 drone_landing/main.py test \
#-rural_root="$ruralroot" -model_config=safeuav_base \
#-name=SUAV_Si_BCEDA_lr -dataset_name="$dataset"

#python3 drone_landing/main.py test \
#-rural_root="$ruralroot" -model_config=safeuav_base \
#-name=SUAV_So_BCED_lr -dataset_name="$dataset"

#python3 drone_landing/main.py test \
#-rural_root="$ruralroot" -model_config=safeuav_base \
#-name=SUAV_So_BCEDA_lr -dataset_name="$dataset"

#python3 drone_landing/main.py test \
#-rural_root="$ruralroot" -model_config=safeuav_base \
#-name=SUAV_So_DA_lr -dataset_name="$dataset"

#python3 drone_landing/main.py test \
#-rural_root="$ruralroot" -model_config=safeuav_base \
#-name=SUAV_Si_DA_lr -dataset_name="$dataset"

#python3 drone_landing/main.py test \
#-rural_root="$ruralroot" -model_config=safeuav_base \
#-name=SUAV_Si_D_lr -dataset_name="$dataset"

#python3 drone_landing/main.py test \
#-rural_root="$ruralroot" -model_config=safeuav_base \
#-name=SUAV_So_D_lr -dataset_name="$dataset"

python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_So_CEWDA_lr -dataset_name="$dataset"

python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_Si_CEWDA_lr -dataset_name="$dataset"

python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_So_CEWD_lr -dataset_name="$dataset"

python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_Si_CEWD_lr -dataset_name="$dataset"

python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_So_CE_lr -dataset_name="$dataset"

python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_Si_CE_lr -dataset_name="$dataset"
