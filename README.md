# FL_last — 관계행렬 기반 층단위 은닉 백도어 방어 (Federated Learning)

연합학습에서 **층 단위 은닉(layer-wise stealth) 백도어 공격**(LGA, LPA)을 막는 룰 기반 방어입니다.
서버가 가진 소량의 clean probe를 **원본·좌우반전·90°·180° 회전** 네 가지 뷰로 각 클라이언트 모델에
통과시키고, 거기서 나온 **클래스 간 관계행렬을 클라이언트끼리 비교**해 이상치를 걸러낸 뒤 집계합니다.

---

## 1. 방법

### 핵심 아이디어

백도어를 심은 클라이언트는 특징 추출기의 파라미터가 변형되어 있습니다. 이 변형은 clean 입력에서는
잘 드러나지 않지만, **입력을 교란하면 클래스 간 기하 구조가 정상 모델과 다르게 반응**합니다.
관계행렬은 임베딩 좌표계에 불변이므로, 독립적으로 학습된 클라이언트 모델끼리도 비교할 수 있습니다.

프로토타입 벡터를 **직접** 비교하지 않는 이유가 여기 있습니다 — 클라이언트마다 임베딩 좌표계가
제각각이라 무해한 좌표계 차이가 신호를 덮습니다. 코사인 관계행렬은 좌표계를 상쇄하고
**클래스 간 상대 기하만** 남깁니다.

### 라운드별 절차

**① Probe 준비** (최초 1회, 고정)

학습셋에서 클래스당 20장을 균등 추출하고 **모든 클라이언트의 학습 데이터에서 제외**합니다
(train-holdout). 시드로 고정되어 전 라운드 동일하며, 랜덤 augmentation 없이 결정론적으로 씁니다.

**② 클라이언트별 관계행렬**

각 업데이트 `w_i`를 템플릿 모델에 로드하고 probe를 네 가지 뷰로 통과시킵니다.

| detector | 입력 |
|---|---|
| `relmat` | 원본 |
| `flip` | 좌우 반전 |
| `rot90` | 90° 회전 |
| `rot180` | 180° 회전 |

각 뷰마다 클래스별 평균 임베딩(프로토타입)을 구하고 코사인 관계행렬을 만듭니다.

```
p_c^(i) = (1/|D_c|) Σ_{x∈D_c} f_i(x)          클래스 c 프로토타입
R_i[a,b] = cos(p_a^(i), p_b^(i))               (K × K)
```

**③ 또래 이상치 점수 (peer)**

자기 자신의 clean 행렬과 비교하지 않고, **그 라운드에 모인 클라이언트들 사이에서** 상대적
이상치 정도를 잽니다.

```
Ru  = R[:, 상삼각]                              클래스쌍 K(K-1)/2 개 (대각·하삼각 제외)
Rz  = (Ru - mean_over_clients) / std_over_clients
s_i = ‖ Rz_i - median_over_clients(Rz) ‖₂
```

기준 통계량(mean·std·median)은 자기 자신을 포함해 계산합니다. 악성 비율이 낮고 median 기반
거리를 쓰기 때문에 self-masking 영향은 실측상 무시할 수준입니다(leave-one-out 대비 ±2%p).

**④ 결합**

```
z_i^(d) = (s_i^(d) - median(s^(d))) / (MAD(s^(d)) + eps)
C_i     = z_relmat + z_flip + z_rot90 + z_rot180
```

가중치 없는 균등 합입니다. 라벨 없는 적응적 가중(gap/max, top-k, 상호 동의도, 고립도,
분할 신뢰도 등)을 여러 가지 시도했으나 모두 균등 합 이하였습니다.

**⑤ 선택 · 집계 · 노이즈**

```
sel     = argsort(C)[0 : 0.7n]                 또래와 비슷한 하위 70%만 사용
w_glob  = FedAvg({w_i}_{i∈sel})
w_glob += eps,  eps ~ N(0, sigma^2 I)
sigma   = 0.1 * median_{i∈전체} ‖w_i - w_glob‖ / sqrt(P)
```

노이즈는 **서버가 집계 후 한 번** 주입합니다(클라이언트별로 넣지 않음). `sigma`의 median은
선택된 클라이언트가 아니라 **전체 클라이언트**의 업데이트 norm 기준입니다 — 그 라운드의
전형적 업데이트 크기를 재는 것이 목적이라 걸러지기 전 분포를 씁니다. `P`는 노이즈 대상
파라미터 수로, BatchNorm의 `running_mean`/`running_var`는 제외합니다(정규화 통계라 노이즈를
넣으면 모델이 망가집니다). 파라미터마다 `N(0, sigma^2)`을 넣으면 전체 노이즈 벡터의 L2 크기가
`sigma*sqrt(P)`이므로, 위와 같이 두면 노이즈가 전형적 업데이트의 약 10%가 됩니다.

### 설정

```
--pb_rule 1 --pb_rule_dets "relmat,flip,rot90,rot180"
--pb_perturb_peer 1 --pb_aug_abs 0 --pb_no_anchor 1
--pb_probe_n <클래스당 20장>  --pb_rule_drop 0.0  --pb_rule_keep 0.7  --noise 0.1
```

### 설계 근거 (모두 실측)

| 선택 | 근거 |
|---|---|
| **또래 비교(peer)** | 자기차이 `‖R_clean − R_pert‖`보다 우수 (CIFAR-10 LGA 17.5→3.1) |
| **앵커 없음** | 앵커 상수는 robust-z에서 상쇄되어 결과에 영향이 없음 |
| **낮은 점수 선택** | 악성의 combined 순위가 상위에 몰림(평균 6.6~7.4 / 9). 반대로 하면 제거율 85%→15% |
| **균등 합** | 라벨 없는 적응 가중 13가지가 모두 균등 합 이하 |
| **probe 20장/클래스** | 40장으로 늘려도 이득 없음 — 방어자 가정이 약할수록 유리 |
| **keep 0.7** | 대안 조합(bright/contrast + keep 0.5)과 5시드 비교 결과 동등하되 main accuracy가 더 높음 |

---

## 2. 파일 구성

### 진입점
| 파일 | 내용 |
|---|---|
| `main_fed.py` | FL 메인 루프. 데이터/모델 준비 → probe 생성 → 라운드마다 로컬 학습(정상/악성) → 방어 집계 → main/BSR 평가 |

### `defense/`
| 파일 | 내용 |
|---|---|
| **`proto_bandit.py`** | **제안 방법 구현.** 핵심: `_features`, `_prototypes`, `_relmat`, `_peer_score`, `_client_scores`, `_rule_aggregate`, `add_weight_noise` |
| `Fed.py` | FedAvg |
| `flame.py`, `fltrust.py`, `flare.py`, `mkrum.py`, `RLRorigin.py`, `multimetric.py` | 비교용 기존 방어 (baseline) |

### `client/`
| 파일 | 내용 |
|---|---|
| `Update.py` | 정상 클라이언트 로컬 학습 |
| `MaliciousUpdate.py` | 악성 클라이언트 학습. `train_ours`(LGA), `train_malicious_LPA`(LPA), `train_malicious_badnet`, `train_malicious_dba` |
| `Attacker.py` | 악성 업데이트 생성 진입점 |
| `AttackerUtils.py` | 공격 대상 층 선택(`get_attack_layers_no_acc`) 등 층 단위 공격 유틸 |
| `add_trigger.py` | 트리거 삽입. `square` 계열, DBA 4분할(`--dba_size`, `--dba_gap`) |
| `test.py` | main accuracy / BSR 평가 (`test_img`) |

### `models/`
| 파일 | 내용 |
|---|---|
| `Nets.py` | ResNet-18, VGG-19-BN(`num_classes` 지원). 방어가 쓰는 `get_feature()` 포함 |
| `simple_mnsit.py`, `subnetutils.py` | 보조 모듈 |

### `utils/`
| 파일 | 내용 |
|---|---|
| `options.py` | 하이퍼파라미터 정의 (`--pb_*`가 제안 방법 옵션) |
| `sampling.py` | non-IID 클라이언트 데이터 분할 |
| `info.py` | 실험 정보 출력/기록 |

### `scripts/`
| 파일 | 내용 |
|---|---|
| `run_main.sh` | 메인 표 재현 (제안 방법 + 무방어, 16셀) |
| `run_baselines.sh` | 기존 방어 5종 실행 |

---

## 3. 주요 옵션

| 옵션 | 설명 |
|---|---|
| `--attack` | `LGA`, `LPA`, `badnet`, `dba` |
| `--defence` | `protobandit`(제안), `flame`, `fltrust`, `flare`, `mkrum`, `rlr`, `mm`, `avg` |
| `--pb_rule 1` | 룰 기반 방어 활성화 |
| `--pb_rule_dets` | detector 목록 (`relmat,flip,rot90,rot180`) |
| `--pb_perturb_peer` | 1 = perturbed 행렬 또래비교, 0 = 자기차이 |
| `--pb_aug_abs` | 1 = 양방향 \|z\|, 0 = 단방향 |
| `--pb_no_anchor` | 1 = 앵커 없이 또래비교 (권장) |
| `--pb_probe_n` | probe 총 장수 (클래스당 20장 권장) |
| `--pb_rule_drop` / `--pb_rule_keep` | band-pass 하한 / 상한 비율 |
| `--noise` | 집계 후 weight-space 노이즈 세기 |
| `--malicious` | 악성 클라이언트 비율 |
| `--p` | non-IID 정도 |
| `--triggerX` / `--triggerY` | 트리거 좌상단 위치 |
| `--dba_size` / `--dba_gap` | DBA 조각 크기 / 조각 간 간격 |
| `--pb_save_client_weights` / `--pb_save_last` | 마지막 N라운드 클라이언트 가중치 저장(사후 분석용) |
| `--pb_dump_relmat` / `--pb_dump_every` | 관계행렬 + 악성 마스크 덤프(오프라인 detector 분석용) |

**부가 실험용 옵션** (기본 off, ablation 재현용): `--pb_warmup_rounds`/`--pb_warmup_keep`(초반 깊은 컷),
`--pb_aug_weight`/`--pb_topk`(적응적 detector 가중·선택). 본 연구에서는 모두 균등 합 이하로 확인되어
최종 설정에 포함하지 않았습니다.

---

## 4. 실행

데이터는 저장소 상위의 `../data/` 에서 읽습니다 (`../data/cifar`, `../data/cifar100`,
`../data/mnist`, `../data/fmnist`).

**제안 방법**
```bash
python -u main_fed.py \
  --dataset cifar --model resnet --attack LGA \
  --defence protobandit --pb_rule 1 \
  --pb_rule_dets "relmat,flip,rot90,rot180" --pb_aug_abs 0 --pb_perturb_peer 1 \
  --pb_no_anchor 1 --pb_probe_n 200 \
  --pb_rule_drop 0.0 --pb_rule_keep 0.7 --noise 0.1 \
  --epochs 300 --malicious 0.1 --p 0.5 --seed 1 --triggerX 27 --triggerY 27
```

**무방어(FedAvg) 기준선** — 선별·노이즈를 끄면 FedAvg와 동일합니다.
```bash
python -u main_fed.py \
  --dataset cifar --model resnet --attack LGA \
  --defence protobandit --pb_rule 1 --pb_rule_dets relmat --pb_no_anchor 1 --pb_probe_n 200 \
  --pb_rule_drop 0.0 --pb_rule_keep 1.0 --noise 0.0 \
  --epochs 300 --malicious 0.1 --p 0.5 --seed 1 --triggerX 27 --triggerY 27
```

**기존 방어 baseline**
```bash
python -u main_fed.py --dataset cifar --model resnet --attack LGA \
  --defence flame --epochs 300 --malicious 0.1 --p 0.5 --seed 1 --triggerX 27 --triggerY 27
# --defence 를 fltrust / mkrum / rlr / flare 로 바꿔 실행
```

**전체 재현**
```bash
GPU=0 SEED=1 bash scripts/run_main.sh        # 제안 + 무방어 (16셀)
GPU=0 SEED=1 bash scripts/run_baselines.sh   # 기존 방어 5종
```

**로그 읽는 법**
```
Main accuracy: 72.60          글로벌 모델 정확도
Backdoor accuracy: 5.02       BSR (낮을수록 방어 성공)
[protorule] iter=... |sel|=7/10 악성통과=0/1 악성순위=8/9
                     ^집계에 쓴 클라 수  ^악성이 걸러졌는지  ^악성의 의심 순위
```

**데이터셋별 설정**

| dataset | rounds | local_ep | lr | probe | trigger |
|---|---|---|---|---|---|
| CIFAR-10 | 300 | 2 | 0.1 | 200 | 27 |
| CIFAR-100 | 300 | 5 | 0.05 | 2000 | 27 |
| MNIST / F-MNIST | 300 | 2 | 0.1 | 200 | ResNet 23 · VGG 27 |

- MNIST/F-MNIST는 28×28이라 ResNet은 트리거를 23에 두어야 5×5가 잘리지 않습니다.
  VGG-19는 maxpool 5회로 28×28에서 공간이 소멸하므로 입력을 32×32로 리사이즈하며 트리거는 27을 씁니다.
- CIFAR-100은 `lr 0.1`에서 VGG-19가 수렴하지 않아(600라운드에 정확도 9.2%) `lr 0.05 · local_ep 5`를 씁니다.

---

## 5. 실험 결과

**5 seeds · 300 rounds · malicious 10% · non-IID (p=0.5)**, 값은 마지막 10라운드 평균의 시드 간 평균±표준편차입니다.

| Model | Dataset | Attack | 무방어 Main / BSR | 제안 Main / BSR |
|---|---|---|---|---|
| ResNet-18 | CIFAR-10 | LGA | 77.9 / 97.1 | 72.6 / **5.0 ± 1.0** |
| ResNet-18 | CIFAR-10 | LPA | 78.3 / 94.7 | 73.3 / **4.1 ± 1.0** |
| ResNet-18 | CIFAR-100 | LGA | 59.1 / 94.5 | 53.0 / **0.4 ± 0.1** |
| ResNet-18 | CIFAR-100 | LPA | 59.9 / 95.1 | 52.8 / **0.5 ± 0.3** |
| ResNet-18 | MNIST | LGA | 99.4 / 100.0 | 99.4 / *40.1 ± 48.9* ✗ |
| ResNet-18 | MNIST | LPA | 99.5 / 96.0 | 99.4 / **0.1 ± 0.0** |
| ResNet-18 | F-MNIST | LGA | 92.6 / 100.0 | 91.8 / **4.7 ± 7.8** |
| ResNet-18 | F-MNIST | LPA | 92.6 / 94.4 | 91.8 / **4.6 ± 8.6** |
| VGG-19-BN | CIFAR-10 | LGA | 81.0 / 95.5 | 79.5 / *91.2 ± 11.7* ✗ |
| VGG-19-BN | CIFAR-10 | LPA | 81.2 / 94.7 | 78.7 / **4.2 ± 1.1** |
| VGG-19-BN | CIFAR-100 | LGA | 37.5 / 49.0 | 38.6 / **0.5 ± 0.1** |
| VGG-19-BN | CIFAR-100 | LPA | 37.2 / 65.3 | 38.4 / **1.5 ± 0.2** |
| VGG-19-BN | MNIST | LGA | 99.5 / 100.0 | 99.5 / **0.3 ± 0.3** |
| VGG-19-BN | MNIST | LPA | 99.5 / 74.2 | 99.5 / **0.1 ± 0.0** |
| VGG-19-BN | F-MNIST | LGA | 92.7 / 100.0 | 92.0 / **0.4 ± 0.2** |
| VGG-19-BN | F-MNIST | LPA | 92.6 / 93.0 | 92.1 / **0.2 ± 0.1** |

**안정적으로 방어되는 14개 셀 평균: BSR 89.1 → 1.9, main accuracy −1.9%p**

### 적용 범위와 한계

**대상**: 파라미터를 직접 조작해 표현 기하를 바꾸는 백도어(LGA, LPA).

**범위 밖**: badnet·DBA 등 순수 데이터 오염 공격. 트리거가 있을 때만 발동하는 지름길을 학습하고
clean 입력의 표현은 그대로 두기 때문에, clean probe 기반 신호로는 보이지 않습니다
(악성 통과율 0.81~0.96).

**실패하는 2개 셀**(VGG-19/CIFAR-10/LGA, ResNet-18/MNIST/LGA)의 원인은 정량적으로 규명했습니다.

1. **신호 축소** — LGA는 층별 업데이트를 직전 업데이트 norm 이내로 클리핑하므로, 악성 업데이트
   크기가 정상의 0.53배(VGG) / 0.60배(ResNet)까지 줄어듭니다. 모델이 덜 움직이니 표현 변화도 작습니다.
2. **잡음 증대** — VGG-19는 정상 클라이언트 간 관계행렬 산포가 ResNet의 1.5배입니다(0.536 vs 0.358).
   probe를 10배로 늘려도 15%만 감소해, 표본 오차가 아닌 **모델 간 진짜 이질성**입니다.
3. **높은 문턱** — 악성 제거율이 90% 이상이어야 방어가 유지됩니다(제거율 91% → BSR 4.7,
   85% → 81, 84% → 85, 82% → 92). 표현 기반 신호로는 62~85%가 한계였습니다.

세 조건이 겹치면 **양성 피드백**이 발생합니다. 백도어가 한 번 진입하면 정상 클라이언트도 오염된
글로벌 모델에서 학습을 시작해 또래 기준점 자체가 오염되고, 탐지가 스스로 무너집니다. 이것이 두 셀에서
관측되는 이봉 분포(BSR이 0.1 근처 아니면 100 근처)의 메커니즘입니다.

또한 VGG-19/CIFAR-100은 무방어 BSR이 49~65로 다른 셀(86~100)보다 낮습니다. 해당 설정에서
모델이 충분히 학습되지 않아(main 37%) 공격 자체가 약하게 걸리는 것으로, 이 셀의 방어 결과는
다른 셀보다 근거가 약합니다.

---

## 6. 환경

```bash
pip install -r requirements.txt
```
PyTorch, torchvision, numpy, pandas, matplotlib, wandb(선택)가 필요합니다.
`--wandb_project <name>` 을 주면 wandb에 기록되고, 생략하면 로컬 로그만 남습니다.

---

## 7. 출처 및 라이선스

본 저장소는 LGA 공격 논문의 공식 구현 [yqqhyqq/LGA](https://github.com/yqqhyqq/LGA)를 기반으로 합니다.
FL 학습 루프(`main_fed.py`), 공격 구현(`client/`), 모델(`models/`), 기존 방어 baseline
(`defense/` 중 `flame`, `fltrust`, `flare`, `mkrum`, `RLRorigin`, `multimetric`)은 원본 저장소에서
가져와 수정한 것입니다.

**본 연구의 기여**
- `defense/proto_bandit.py` — 관계행렬 기반 방어 (신규 작성)
- `utils/options.py` 의 `--pb_*` 옵션군
- `models/Nets.py` VGG `num_classes` 인자화 및 4개 데이터셋 지원
- `main_fed.py` MNIST/F-MNIST 지원(Grayscale 3채널, VGG용 32×32 리사이즈),
  FLTrust/FLARE용 root dataset 생성
- `client/add_trigger.py` DBA 트리거 파라미터화 (`--dba_size`, `--dba_gap`)

라이선스는 원본 저장소의 `LICENSE`를 따릅니다.
