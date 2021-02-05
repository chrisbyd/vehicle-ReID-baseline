#!/usr/bin/env bash

# vehicleID  and veri

python train_xent_tri.py \
-s vehicleID \
-t vehicleID \
--height 128 \
--width 256 \
--train-batch-size  120 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
-a resnet50 \
--eval-freq 10 \
--save-dir log/resnet50-veri \
--gpu-devices 0