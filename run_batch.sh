#!/bin/bash

ROOT_DIR=~/Source/Repos/TOCI/TOCI/55.K_Fold_Thumb/
BATCHS=`ls $ROOT_DIR`

echo Running in $ROOT_DIR and looping over $BATCHS

for i in $BATCHS
do
    echo Doing $ROOT_DIR/$i...
    /home/lwong/Toolkits/Anaconda2/bin/python main.py $ROOT_DIR/$i/Training \
    --train $ROOT_DIR/$i/Training  \
    --useCUDA -e 500 -b 200 -d 0.005 --train-params "{'lr': 5e-5, 'momentum':0.1}" \
    --stage 2 --checkpoint $ROOT_DIR/$i/checkpoint_WNET.pt --load $ROOT_DIR/$i/checkpoint_WNET.pt
done
