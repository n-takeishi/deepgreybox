#!/bin/bash

exptname=$1
trialname=$2
mode=$3
seed=$4
coeff=$5
device=$6

# common configs
if [ ${exptname} == "toy1" ]; then
    optimconfig0="--batch-size 10 --max-epochs 2000 --learning-rate 1e-2 1e-4 --weight-decay 1e-2"
    optimconfig1="--batch-size 999999 --max-epochs 2000 --learning-rate 1e-2 1e-4 --weight-decay 0"
elif [ ${exptname} == "pendulum" ]; then
    optimconfig0="--batch-size 50 --max-epochs 500 --learning-rate 1e-3 1e-5 --weight-decay 1e-2"
    optimconfig1="--num-grids 400 --adaptive-pred-only"
elif [ ${exptname} == "reaction-diffusion" ]; then
    optimconfig0="--batch-size 20 --max-epochs 1000 --learning-rate 1e-3 1e-5 --weight-decay 1e-2"
    optimconfig1="--batch-size 999999 --max-epochs 100 --learning-rate 1e-3 1e-5 --weight-decay 0"
elif [ ${exptname} == "predator-prey" ]; then
    optimconfig0="--batch-size 100 --max-epochs 1000 --learning-rate 1e-3 1e-3 --weight-decay 1e-2"
    optimconfig1="--batch-size 100 --max-epochs 50 --learning-rate 1e-4 1e-4 --weight-decay 0"
else
    echo "unknown exptname"
    return 1
fi

export MKL_THREADING_LAYER=sequential

# prepare original data if it does not exist
if [ ! -f ./${exptname}/out/data_raw.npz ]; then
    python ${exptname}/make_dataset.py
fi

# tr-va-te split
python ${exptname}/build_features.py ${trialname} --seed ${seed}

# main procedures
if [ ${mode} == "adaptive" ] || [ ${mode} == "inductive" ]; then
    python common/train.py ${exptname} ${trialname} --mode ${mode} --seed ${seed} --coeff-R ${coeff} ${optimconfig0} --device ${device}
    for target in "te" #"va"
    do
        python common/predict.py ${exptname} ${trialname} --mode ${mode} --seed ${seed} --target ${target} ${optimconfig1} --device ${device}
    done
elif [ ${mode} == "transductive" ]; then
    python common/train.py ${exptname} ${trialname} --mode ${mode} --seed ${seed} --device ${device}
    for target in "te" #"va"
    do
        python common/predict.py ${exptname} ${trialname} --mode ${mode} --seed ${seed} --target ${target} --coeff-R ${coeff} ${optimconfig0} --device ${device}
    done
else
    echo "unknown mode"
    return 1
fi

