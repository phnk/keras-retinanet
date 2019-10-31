#!/bin/bash
clear
keras_retinanet/bin/train.py --weights /home/carbor/code/keras-retinanet/weights/resnet50_coco_best_v2.1.0.h5 --random-transform csv /home/carbor/code/keras-retinanet/train.csv /home/carbor/code/keras-retinanet/classes.csv --val-annotations /home/carbor/code/keras-retinanet/valid.csv