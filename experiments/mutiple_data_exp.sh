#!/bin/bash


dataset_rural=datasets.synthetic.RuralscapesOrigToSynthetic
dataset_uavid=datasets.synthetic.UAVidToSynthetic
dataset_tugraz=datasets.tugraz_sort.TUGrazSortedDataset
dataset_aero=datasets.synthetic.AeroscapesToSynthetic
dataset_uav123=datasets.synthetic.UAV123ToSynthetic
dataset_ruralaero=datasets.synthetic.RuralAero

home=xxx
ruralroot=/home/"$home"/tfm/ruralscapes_light
uavidroot=/home/"$home"/tfm/uavid_v1.5
tugrazroot=/home/"$home"/tfm/semantic_drone_dataset
aeroroot=/home/"$home"/tfm/aeroscapes
uav123root=/home/"$home"/tfm/uav123

params="-override=False -stop_after_miss=3 -batch_size=4 -augment=True \
        -uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
        -num_threads=4 -max_epochs=150 -delta=-0.01 -model_config.optimizer.params.lr=0.0001 \
        -model_config.net.name=unet -model_config.net.params.encoder_name=resnet18 -model_config=base
        -model_config.loss.name=custom_models.losses.BCELLDiceAvg "

 
# Uavid
python3 drone_landing/main.py train  -dataset_name="$dataset_uavid" \
-name=uavid -validation_epochs=10 $params

# Ruralscapes
python3 drone_landing/main.py train -dataset_name="$dataset_rural" \
-name=ruralscapes -validation_epochs=10 $params

# Aeroscapes
python3 drone_landing/main.py train -dataset_name="$dataset_aero" \
-name=aeroscapes -validation_epochs=10 $params

# Rural by uavid
python3 drone_landing/main.py train -dataset_name="$dataset_rural" \
-name=ruralbyuavid -reuse=True -reuse_path=/home/$home/tfm/executions/uavid/models/0 \
-validation_epochs=2 -data_factor=4 $params

# Aero by uavid
python3 drone_landing/main.py train -dataset_name="$dataset_rural" \
-name=aerobyuavid -reuse=True -reuse_path=/home/$home/tfm/executions/uavid/models/0 \
-validation_epochs=2 -data_factor=4 $params

# Aeroandrural by uavid
python3 drone_landing/main.py train -dataset_name="$dataset_rural" \
-name=aeroandrural -reuse=True -reuse_path=/home/$home/tfm/executions/uavid/models/0 \
-validation_epochs=2 -data_factor=4 $params


#TEST ----------------------------------------------------------
cp -r /home/$home/tfm/executions/uavid /home/$home/tfm/executions/uavid_t123
cp -r /home/$home/tfm/executions/ruralscapes /home/$home/tfm/executions/rural_t123
cp -r /home/$home/tfm/executions/aeroscapes /home/$home/tfm/executions/aero_t123
cp -r /home/$home/tfm/executions/ruralbyuavid /home/$home/tfm/executions/ruralbyuavid_t123
cp -r /home/$home/tfm/executions/aerobyuavid /home/$home/tfm/executions/aerobyuavid_t123
cp -r /home/$home/tfm/executions/aeroandrural /home/$home/tfm/executions/aeroandrural_t123

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=uavid -dataset_name="$dataset_uavid"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=ruralscapes -dataset_name="$dataset_rural"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=aeroscapes -dataset_name="$dataset_aero"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=ruralbyuavid -dataset_name="$dataset_rural"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=aerobyuavid -dataset_name="$dataset_aero"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=aeroandrural -dataset_name="$dataset_ruralaero"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=uavid_t123 -dataset_name="$dataset_uav123"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=rural_t123 -dataset_name="$dataset_uav123"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=aero_t123 -dataset_name="$dataset_uav123"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=ruralbyuavid_t123 -dataset_name="$dataset_uav123"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=aerobyuavid_t123 -dataset_name="$dataset_uav123"

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=aeroandrural_t123 -dataset_name="$dataset_uav123"