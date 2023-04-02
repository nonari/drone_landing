#!/bin/bash

#SUAV_Ll_XXXX
#Ll-> Sigmoid | Softmax
#XXXX -> Loss

#Class probability distribution 
#[0.24186207, 0.2048791, 0.28634467, 0.09043592, 0.01872263, 0.03582632, 0.08700141, 0.02159951, 0.00299932, 0.00311403, 0.00093436, 0.00628066]

dataset=datasets.ruralscapes.RuralscapesOrigSplit
ruralroot=/home/xxx/tfm/ruralscapes_light
weights="[0.04134588, 0.04880927, 0.03492295, 0.11057553, 0.534113, 0.2791244, 0.11494067, 0.46297347, 1, 1, 1, 1]"

# BCEDice Sigmoid
echo python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_Si_BCED -override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
-num_threads=4 -max_epochs=150 -delta=-0.01 \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.BCEDiceLoss
 
# BCEDiceAvg Sigmoid
echo python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_Si_BCEDA -override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
-num_threads=4 -max_epochs=150 -delta=-0.01 \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss
 
 # BCEDice Softmax
echo python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_So_BCED -override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
-num_threads=4 -max_epochs=150 -delta=-0.01 \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.BCEDiceLoss
 
# BCEDiceAvg Softmax
echo python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_So_BCEDA -override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
-num_threads=4 -max_epochs=150 -delta=-0.01 \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.BCEDiceAvgLoss
 
 
# DiceAvg Softmax
echo python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_So_DA -override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
-num_threads=4 -max_epochs=150 -delta=-0.01 \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.DiceAvgLoss


# DiceAvg Sigmoid
echo python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_Si_DA -override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
-num_threads=4 -max_epochs=150 -delta=-0.01 \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.DiceAvgLoss


# Dice Sigmoid
echo python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_Si_D -override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
-num_threads=4 -max_epochs=150 -delta=-0.01 \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.DiceAvgLoss

# Dice Softmax
echo python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_So_D -override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
-num_threads=4 -max_epochs=150 -delta=-0.01 \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.DiceLoss


# CEWeightDiceAvgLoss Softmax
echo python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_So_CEWDA -override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
-num_threads=4 -max_epochs=150 -delta=-0.01 \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.CEWeightDiceAvgLoss \
-model_config.loss.params.w="$weights"


# CEWeightDiceLoss Softmax
echo python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_So_CEWD -override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
-num_threads=4 -max_epochs=150 -delta=-0.01 \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.CEWeightDiceLoss \
-model_config.loss.params.w="$weights"

# CELoss Softmax
echo python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_So_CE -override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
-num_threads=4 -max_epochs=150 -delta=-0.01 \
-model_config=safeuav_base \
-model_config.net.params.last=softmax \
-model_config.loss.name=custom_models.losses.CELoss \
-model_config.loss.params.w="$weights"


# CELoss Sigmoid
echo python3 drone_landing/main.py train -rural_root="$ruralroot" -dataset_name="$dataset" \
-name=SUAV_Si_CE -override=False -validation_epochs=10 -stop_after_miss=3 -batch_size=4 \
-num_threads=4 -max_epochs=150 -delta=-0.01 \
-model_config=safeuav_base \
-model_config.net.params.last=sigmoid \
-model_config.loss.name=custom_models.losses.CELoss \
-model_config.loss.params.w="$weights"



# TESTS

echo python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_Si_BCED -dataset_name="$dataset"

echo python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_Si_BCEDA -dataset_name="$dataset"

echo python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_So_BCED -dataset_name="$dataset"

echo python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_So_BCEDA -dataset_name="$dataset"

echo python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_So_DA -dataset_name="$dataset"

echo python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_Si_DA -dataset_name="$dataset"

echo python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_Si_D -dataset_name="$dataset"

echo python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_So_D -dataset_name="$dataset"

echo python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_So_CEWDA -dataset_name="$dataset"

echo python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_So_CEWD -dataset_name="$dataset"

echo python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_So_CE -dataset_name="$dataset"

echo python3 drone_landing/main.py test \
-rural_root="$ruralroot" -model_config=safeuav_base \
-name=SUAV_Si_CE -dataset_name="$dataset"

