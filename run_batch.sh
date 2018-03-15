#!/bin/bash

ROOT_DIR=~/Source/Repos/TOCI/TOCI/53.K_Fold/
BATCHS=`ls $ROOT_DIR`

for i in $BATCHS
do
    /home/lwong/Toolkits/Anaconda2/bin/python main.py $ROOT_DIR/$i/Training/ROIs \
    --train $ROOT_DIR/$i/Training  \
    --useCUDA -e 7500 -b 800 -d 0.005 --train-params "{'lr': 5e-5, 'momentum':0.1}" \
    --stage 2 --checkpoint $ROOT_DIR/$i/checkpoint_WNET.pt --load $ROOT_DIR/$i/checkpoint_WNET.pt
done
