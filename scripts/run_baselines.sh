#!/usr/bin/env bash
# 기존 방어 baseline: FLAME / FLTrust / multi-Krum / RLR / FLARE
# 사용법:  GPU=0 SEED=1 bash scripts/run_baselines.sh
set -u
PY=${PY:-python}; GPU=${GPU:-0}; SEED=${SEED:-1}
cd "$(dirname "$0")/.."; mkdir -p logs
for def in flame fltrust mkrum rlr flare; do
for mdl in resnet VGG; do
for ds in cifar cifar100 mnist fmnist; do
for atk in LGA LPA; do
  lr=0.1; lep=2; tg=27
  [ "$ds" = cifar100 ] && { lr=0.05; lep=5; }
  if [ "$ds" = mnist ] || [ "$ds" = fmnist ]; then
    [ "$mdl" = resnet ] && tg=23 || tg=27
  fi
  tag="bl_${def}_${mdl}_${ds}_${atk}_s${SEED}"
  echo "[run] $tag"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u main_fed.py \
    --dataset "$ds" --model "$mdl" --attack "$atk" --defence "$def" \
    --epochs 300 --local_ep $lep --lr $lr \
    --malicious 0.1 --p 0.5 --seed $SEED --gpu 0 --triggerX $tg --triggerY $tg \
    > "logs/${tag}.log" 2>&1
done; done; done; done
