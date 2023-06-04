#!/bin/bash


dataset_rural=datasets.synthetic.RuralscapesOrigToSynthetic
dataset_uavid=datasets.synthetic.UAVidToSynthetic
dataset_tugraz=datasets.tugraz_sort.TUGrazSortedDataset
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

 
# Uavid
python3 drone_landing/main.py train  -dataset_name="$dataset_uavid" \
-name=uavid -data_factor=1 -validation_epochs=10 $params

# Ruralscapes
python3 drone_landing/main.py train -dataset_name="$dataset_rural" \
-name=ruralscapes -data_factor=1 -validation_epochs=10 $params

# Aeroscapes
python3 drone_landing/main.py train -dataset_name="$dataset_aero" \
-name=aeroscapes -data_factor=2 -validation_epochs=10 $params

# Rural by uavid
python3 drone_landing/main.py train -dataset_name="$dataset_rural" \
-name=ruralbyuavid -reuse=True -reuse_path="$executions"/uavid/models/0 \
-validation_epochs=2 -data_factor=4 $params

# Aero by uavid
python3 drone_landing/main.py train -dataset_name="$dataset_rural" \
-name=aerobyuavid -reuse=True -reuse_path="$executions"/uavid/models/0 \
-validation_epochs=2 -data_factor=4 $params

# Aeroandrural by uavid
python3 drone_landing/main.py train -dataset_name="$dataset_rural" \
-name=aeroandrural -reuse=True -reuse_path="$executions"/uavid/models/0 \
-validation_epochs=2 -data_factor=4 $params


#TEST ----------------------------------------------------------
cp -r "$executions"/uavid "$executions"/uavid_t123
cp -r "$executions"/ruralscapes "$executions"/rural_t123
cp -r "$executions"/aeroscapes "$executions"/aero_t123
cp -r "$executions"/ruralbyuavid "$executions"/ruralbyuavid_t123
cp -r "$executions"/aerobyuavid "$executions"/aerobyuavid_t123
cp -r "$executions"/aeroandrural "$executions"/aeroandrural_t123
cp -r "$executions"/ruralbyuavid "$executions"/ruralbyuavid_taero

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

python3 drone_landing/main.py test \
-uavid_root=$uavidroot -rural_root=$ruralroot -tugraz_root=$tugrazroot -aeroscapes_root=$aeroroot -uav123_root=$uav123root \
-model_config=base -name=ruralbyuavid_taero -dataset_name="$dataset_aero"