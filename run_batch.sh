#!/bin/bash

ROOT_DIR=~/Source/TOCI/TOCI/55.K_Fold_Thumb/
BATCHS="005 003 004 016 020 026"
CUDA_VISIBLE_DEVICES=0

echo Running in $ROOT_DIR and looping over $BATCHS

for i in ${BATCHS}

do
    echo Doing $ROOT_DIR/$i...
    /home/lwong/Toolkits/Anaconda2/bin/python main.py $ROOT_DIR/$i/Training \
    --train $ROOT_DIR/$i/Training  \
    --useCUDA -e 500 -b 55 -d 0.005 --train-params "{'lr': 1e-5, 'momentum':0.1}" \
    --stage 2 --checkpoint $ROOT_DIR/$i/checkpoint_WNET.pt --load $ROOT_DIR/$i/checkpoint_WNET.pt
done
