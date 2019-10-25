#!/bin/bash
clear
keras_retinanet/bin/train.py csv /home/carbor/code/keras-retinanet/train.csv /home/carbor/code/keras-retinanet/classes.csv --weights /home/carbor/code/keras-retinanet/weights/resnet50_coco_best_v2.1.0.h5 --val-annotations /home/carbor/code/keras-retinanet/valid.csv --batch-size 3 --lr 0.001 --random-transform