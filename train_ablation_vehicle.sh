#!/usr/bin/env bash

# vehicleID  and veri

python train_hash.py \
-s vehicleID \
-t vehicleID \
--height 128 \
--width 256 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 96 \
--test-batch-size 100 \
--hash-bit-number 2048 \
-a reshash \
--eval-freq 10 \
--save-dir log/resnet50-veri \
--gpu-devices 0 \
--lambda-xent 1 \
--lambda-htri 1 \
--lambda-quant  0
