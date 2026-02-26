#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HDF5_USE_FILE_LOCKING=FALSE

for FE in musk uni_v1 conch_v15 phikon_v2 resnet50 uni_v2
do
  echo "===== Running with feature extractor: $FE ====="
  python main.py --train-mode grading --seed 20,40,60,80,100 --train-dataset-name panda --test-dataset-name panda --resolution-list x10 --feature-extractor $FE --mil-model DTFD-MIL --distill AFS --epochs 50 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.03 --mil-patch-drop-max 0.1 --gpu-id 0 --dataset-root /data/juhyeon/
  python main.py --train-mode grading --seed 20,40,60,80,100 --train-dataset-name panda --test-dataset-name panda --resolution-list x10 --feature-extractor $FE --mil-model ILRA --epochs 50 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.03 --mil-patch-drop-max 0.1 --gpu-id 0 --dataset-root /data/juhyeon/
  python main.py --train-mode grading --seed 20,40,60,80,100 --train-dataset-name panda --test-dataset-name panda --resolution-list x10 --feature-extractor $FE --mil-model RRTMIL --epochs 50 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.03 --mil-patch-drop-max 0.1 --gpu-id 0 --dataset-root /data/juhyeon/
  python main.py --train-mode grading --seed 20,40,60,80,100 --train-dataset-name panda --test-dataset-name panda --resolution-list x10 --feature-extractor $FE --mil-model WiKG --epochs 50 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.03 --mil-patch-drop-max 0.1 --gpu-id 0 --dataset-root /data/juhyeon/
  python main.py --train-mode grading --seed 20,40,60,80,100 --train-dataset-name panda --test-dataset-name panda --resolution-list x10 --feature-extractor $FE --mil-model CLAM-MB --epochs 50 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.03 --mil-patch-drop-max 0.1 --gpu-id 0 --dataset-root /data/juhyeon/
  python main.py --train-mode grading --seed 20,40,60,80,100 --train-dataset-name panda --test-dataset-name panda --resolution-list x10 --feature-extractor $FE --mil-model TransMIL --epochs 50 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.03 --mil-patch-drop-max 0.1 --gpu-id 0 --dataset-root /data/juhyeon/
  python main.py --train-mode grading --seed 20,40,60,80,100 --train-dataset-name panda --test-dataset-name panda --resolution-list x10 --feature-extractor $FE --mil-model ABMIL --epochs 50 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.03 --mil-patch-drop-max 0.1 --gpu-id 0 --dataset-root /data/juhyeon/
  python main.py --train-mode grading --seed 20,40,60,80,100 --train-dataset-name panda --test-dataset-name panda --resolution-list x10 --feature-extractor $FE --mil-model DSMIL --epochs 50 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.03 --mil-patch-drop-max 0.1 --gpu-id 0 --dataset-root /data/juhyeon/
  python main.py --train-mode grading --seed 20,40,60,80,100 --train-dataset-name panda --test-dataset-name panda --resolution-list x10 --feature-extractor $FE --mil-model meanpooling --epochs 50 --opt radam --lr 5e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.03 --mil-patch-drop-max 0.1 --gpu-id 0 --dataset-root /data/juhyeon/
done

# --opt lookahead_radam

#  python main.py --train-mode grading --seed 20,40,60,80,100 --train-dataset-name panda --test-dataset-name panda --resolution-list x10 --feature-extractor $FE --mil-model maxpooling --epochs 50 --opt radam --lr 1e-3 --weight-decay 2e-5 --mil-patch-drop-min 0.03 --mil-patch-drop-max 0.1 --gpu-id 0 --dataset-root /data/juhyeon/
#   python main.py --train-mode grading --seed 20,40,60,80,100 --train-dataset-name panda --test-dataset-name panda --resolution-list x10 --feature-extractor $FE --mil-model Transformer --epochs 50 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.03 --mil-patch-drop-max 0.1 --gpu-id 0 --dataset-root /data/juhyeon/
#   python main.py --train-mode grading --seed 20,40,60,80,100 --train-dataset-name panda --test-dataset-name panda --resolution-list x10 --feature-extractor $FE --mil-model CLAM-SB --epochs 50 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.03 --mil-patch-drop-max 0.1 --gpu-id 0 --dataset-root /data/juhyeon/


echo "모든 실험이 완료되었습니다!"