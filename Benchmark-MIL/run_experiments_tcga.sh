#!/bin/bash
# 

for FE in hibou_b vit-ssl-dino-p16 virchow2 conch_v1 musk uni_v1 conch_v15 phikon_v2 resnet50 uni_v2
do
  echo "===== Running with feature extractor: $FE ====="
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-RCC --test-dataset-name TCGA-RCC --resolution-list x10 --feature-extractor $FE --mil-model DTFD-MIL --distill AFS --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-RCC --test-dataset-name TCGA-RCC --resolution-list x10 --feature-extractor $FE --mil-model ILRA --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-RCC --test-dataset-name TCGA-RCC --resolution-list x10 --feature-extractor $FE --mil-model RRTMIL --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-RCC --test-dataset-name TCGA-RCC --resolution-list x10 --feature-extractor $FE --mil-model WiKG --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-RCC --test-dataset-name TCGA-RCC --resolution-list x10 --feature-extractor $FE --mil-model CLAM-MB --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-RCC --test-dataset-name TCGA-RCC --resolution-list x10 --feature-extractor $FE --mil-model TransMIL --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-RCC --test-dataset-name TCGA-RCC --resolution-list x10 --feature-extractor $FE --mil-model ABMIL --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-RCC --test-dataset-name TCGA-RCC --resolution-list x10 --feature-extractor $FE --mil-model DSMIL --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-RCC --test-dataset-name TCGA-RCC --resolution-list x10 --feature-extractor $FE --mil-model meanpooling --epochs 30 --opt radam --lr 5e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
done

for FE in hibou_b vit-ssl-dino-p16 virchow2 conch_v1 musk uni_v1 conch_v15 phikon_v2 resnet50 uni_v2
do
  echo "===== Running with feature extractor: $FE ====="
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-GLIOMA --test-dataset-name TCGA-GLIOMA --resolution-list x10 --feature-extractor $FE --mil-model DTFD-MIL --distill AFS --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-GLIOMA --test-dataset-name TCGA-GLIOMA --resolution-list x10 --feature-extractor $FE --mil-model ILRA --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-GLIOMA --test-dataset-name TCGA-GLIOMA --resolution-list x10 --feature-extractor $FE --mil-model RRTMIL --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-GLIOMA --test-dataset-name TCGA-GLIOMA --resolution-list x10 --feature-extractor $FE --mil-model WiKG --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-GLIOMA --test-dataset-name TCGA-GLIOMA --resolution-list x10 --feature-extractor $FE --mil-model CLAM-MB --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-GLIOMA --test-dataset-name TCGA-GLIOMA --resolution-list x10 --feature-extractor $FE --mil-model TransMIL --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-GLIOMA --test-dataset-name TCGA-GLIOMA --resolution-list x10 --feature-extractor $FE --mil-model ABMIL --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-GLIOMA --test-dataset-name TCGA-GLIOMA --resolution-list x10 --feature-extractor $FE --mil-model DSMIL --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-GLIOMA --test-dataset-name TCGA-GLIOMA --resolution-list x10 --feature-extractor $FE --mil-model meanpooling --epochs 30 --opt radam --lr 5e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
done



for FE in hibou_b vit-ssl-dino-p16 virchow2 conch_v1 musk uni_v1 conch_v15 phikon_v2 resnet50 uni_v2
do
  echo "===== Running with feature extractor: $FE ====="
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-NSCLC --test-dataset-name TCGA-NSCLC --resolution-list x10 --feature-extractor $FE --mil-model DTFD-MIL --distill AFS --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-NSCLC --test-dataset-name TCGA-NSCLC --resolution-list x10 --feature-extractor $FE --mil-model ILRA --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-NSCLC --test-dataset-name TCGA-NSCLC --resolution-list x10 --feature-extractor $FE --mil-model RRTMIL --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-NSCLC --test-dataset-name TCGA-NSCLC --resolution-list x10 --feature-extractor $FE --mil-model WiKG --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-NSCLC --test-dataset-name TCGA-NSCLC --resolution-list x10 --feature-extractor $FE --mil-model CLAM-MB --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-NSCLC --test-dataset-name TCGA-NSCLC --resolution-list x10 --feature-extractor $FE --mil-model TransMIL --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-NSCLC --test-dataset-name TCGA-NSCLC --resolution-list x10 --feature-extractor $FE --mil-model ABMIL --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-NSCLC --test-dataset-name TCGA-NSCLC --resolution-list x10 --feature-extractor $FE --mil-model DSMIL --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-NSCLC --test-dataset-name TCGA-NSCLC --resolution-list x10 --feature-extractor $FE --mil-model meanpooling --epochs 30 --opt radam --lr 5e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
done






# --opt lookahead_radam

#  python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-GLIOMA --test-dataset-name TCGA-GLIOMA --resolution-list x10 --feature-extractor $FE --mil-model maxpooling --epochs 30 --opt radam --lr 1e-3 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
#   python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-GLIOMA --test-dataset-name TCGA-GLIOMA --resolution-list x10 --feature-extractor $FE --mil-model Transformer --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/
#   python main.py --seed 20,40,60,80,100 --train-dataset-name TCGA-GLIOMA --test-dataset-name TCGA-GLIOMA --resolution-list x10 --feature-extractor $FE --mil-model CLAM-SB --epochs 30 --opt radam --lr 1e-4 --weight-decay 2e-5 --mil-patch-drop-min 0.2 --mil-patch-drop-max 0.4 --gpu-id 0 --dataset-root /home/super/Desktop/codes/juhyeon/data1/benchmark_data/


echo "모든 실험이 완료되었습니다!"