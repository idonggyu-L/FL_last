#!/usr/bin/env bash
# 메인 표 재현: {ResNet, VGG} × {CIFAR-10, CIFAR-100, MNIST, F-MNIST} × {LGA, LPA} × {제안, 무방어}
# 제안 방법: relmat,flip,rot90,rot180 · peer · abs0 · keep0.7 · noise0.1 · probe 클래스당 20장
# 사용법:  GPU=0 SEED=1 bash scripts/run_main.sh
set -u
PY=${PY:-python}; GPU=${GPU:-0}; SEED=${SEED:-1}
cd "$(dirname "$0")/.."; mkdir -p logs
for mdl in resnet VGG; do
for ds in cifar cifar100 mnist fmnist; do
for atk in LGA LPA; do
for meth in ours nodef; do
  pn=200; lr=0.1; lep=2; tg=27
  [ "$ds" = cifar100 ] && { pn=2000; lr=0.05; lep=5; }
  if [ "$ds" = mnist ] || [ "$ds" = fmnist ]; then
    [ "$mdl" = resnet ] && tg=23 || tg=27     # ResNet 28x28 / VGG는 32x32 리사이즈
  fi
  if [ "$meth" = ours ]; then
    dets="relmat,flip,rot90,rot180"; keep=0.7; noise=0.1; extra="--pb_perturb_peer 1"
  else
    dets="relmat"; keep=1.0; noise=0.0; extra=""    # keep-all + noise0 = FedAvg
  fi
  tag="${meth}_${mdl}_${ds}_${atk}_s${SEED}"
  echo "[run] $tag"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u main_fed.py \
    --dataset "$ds" --model "$mdl" --attack "$atk" \
    --defence protobandit --pb_rule 1 --pb_rule_dets "$dets" --pb_aug_abs 0 --pb_no_anchor 1 \
    --pb_probe_n $pn --pb_rule_drop 0.0 --pb_rule_keep $keep --noise $noise \
    --epochs 300 --local_ep $lep --lr $lr \
    --malicious 0.1 --p 0.5 --seed $SEED --gpu 0 --triggerX $tg --triggerY $tg \
    $extra > "logs/${tag}.log" 2>&1
done; done; done; done
