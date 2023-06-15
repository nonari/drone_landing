#!/bin/bash


dataset_rural=datasets.synthetic.RuralscapesOrigToSynthetic
dataset_uavid=datasets.synthetic.UAVidToSynthetic
dataset_tugraz=datasets.synthetic.TUGrazToSynthetic
dataset_aero=datasets.synthetic.AeroscapesToSynthetic
dataset_uav123=datasets.synthetic.UAV123ToSynthetic
dataset_ruralaero=datasets.synthetic.RuralAero
dataset_uavid_ori=datasets.uavid.UAVid

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
-model_config=base -name=aeroandruralbase_tu -dataset_name="$dataset_tugraz"



python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_bcelldice_old -dataset_name="$dataset_uavid_ori"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_bcelldice -dataset_name="$dataset_uavid_ori"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_bcelldice_lowlr -dataset_name="$dataset_uavid_ori"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_focal -dataset_name="$dataset_uavid_ori"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_focalw -dataset_name="$dataset_uavid_ori"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_bcelldice_w -dataset_name="$dataset_uavid_ori"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_bcelldice_512 -dataset_name="$dataset_uavid_ori"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_bcelldice_416 -dataset_name="$dataset_uavid_ori"

python3 drone_landing/main.py test \
-uavid_root="$uavidroot" -model_config=base \
-name=uavid_bcelldice_352 -dataset_name="$dataset_uavid_ori"