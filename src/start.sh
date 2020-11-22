#!/bin/sh

cd /home/src && python2 api.py --gpu 0 --net resnet34s --ghost_cluster 2 --vlad_cluster 8 --warmup_file "chunk-00.wav" --resume "/home/weights-15-0.873.h5"