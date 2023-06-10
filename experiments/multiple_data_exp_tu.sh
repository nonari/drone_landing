#!/bin/bash


dataset_rural=datasets.synthetic.RuralscapesOrigToSynthetic
dataset_uavid=datasets.synthetic.UAVidToSynthetic
dataset_tugraz=datasets.synthetic.TUGrazToSynthetic
dataset_aero=datasets.synthetic.AeroscapesToSynthetic
dataset_uav123=datasets.synthetic.UAV123ToSynthetic
dataset_ruralaero=datasets.synthetic.RuralAero


home=/home/xxx/tfm
executions=/home/xxx/tfm/executions

ruralroot="$home"/ruralscapes_light
uavidroot="$home"/uavid_v1.5
tugrazroot="$home"/semantic_drone_dataset
aeroroot="$home"/aeroscapes
uav123root="$home"/uav123

params="-override=False -stop_after_miss=3 -batch_size=4 -augment=True \
        -uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
        -num_threads=4 -max_epochs=200 -delta=-0.01 -model_config.optimizer.params.lr=0.0001 \
        -model_config.net.name=unet -model_config.net.params.encoder_name=resnet18 -model_config=base
        -model_config.loss.name=custom_models.losses.BCELLDiceAvg "


#TEST ----------------------------------------------------------
cp -r "$executions"/uavid "$executions"/uavid_tu
cp -r "$executions"/ruralscapes "$executions"/rural_tu
cp -r "$executions"/aeroscapes "$executions"/aero_tu
cp -r "$executions"/ruralbyuavid "$executions"/ruralbyuavid_tu
cp -r "$executions"/aerobyuavid "$executions"/aerobyuavid_tu
cp -r "$executions"/aeroandrural "$executions"/aeroandrural_tu
cp -r "$executions"/ruralbyuavid "$executions"/ruralbyuavid_taero

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=uavid_tu -dataset_name="$dataset_tugraz"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=rural_tu -dataset_name="$dataset_tugraz"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=aero_tu -dataset_name="$dataset_tugraz"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=ruralbyuavid_tu -dataset_name="$dataset_tugraz"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=aerobyuavid_tu -dataset_name="$dataset_tugraz"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=aeroandrural_tu -dataset_name="$dataset_tugraz"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=ruralbyuavid_taero -dataset_name="$dataset_aero"