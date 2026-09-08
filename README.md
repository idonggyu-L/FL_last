# FL_last — 표현·파라미터 2축 백도어 방어 (Federated Learning)

연합학습 백도어 공격을 막는 룰 기반 방어입니다. 각 클라이언트 업데이트를 **클래스 기하의 이탈**과
**파라미터 이동 크기**라는 두 축에 놓고, 그 평면에서 또래 분포에 대한 **마할라노비스 거리** 하나로
이상치를 걸러낸 뒤 집계합니다. 학습되는 요소도, 임계값 튜닝도 없습니다.

---

## 1. 방법

### 핵심 아이디어

백도어를 심은 클라이언트는 두 흔적 중 최소 하나를 남깁니다 — **깨끗한 입력에 대한 클래스 기하의
왜곡**, 또는 **업데이트 크기의 이상**. 어느 쪽이 드러나는지는 공격 방식이 결정하며, 한 축만 보는
방어는 반드시 사각지대를 갖습니다.

| 공격 유형 | `‖Δw‖` (정상 대비) | 어느 축이 잡는가 |
|---|---|---|
| 업데이트를 깎아 숨음 (LGA) | 0.11x | 파라미터 |
| 업데이트를 키움 (BadNet, DBA) | 3.05x | 파라미터 |
| 업데이트를 키움 (DAA) | 2.03x | 파라미터 |
| 층 단위 교체로 크기를 보존 (LPA, Adaptive) | 1.00x | **표현** |

층 단위 교체 공격은 정상 학습 결과의 일부 층만 백도어 모델의 층으로 갈아끼웁니다. 나머지 층이
문자 그대로 정상 업데이트이므로 `‖Δw‖`가 정상과 같아지는 것이 **구성상 필연**이며, 크기 기반
통계로는 원리적으로 잡히지 않습니다. 그런 공격도 계산하는 **함수**는 달라지므로 기하에는 드러납니다.

두 축을 각각 문턱으로 자르는 대신 `(기하 이탈, 크기 이동)` 평면에서 거리 하나로 판정합니다.
공격마다 평면의 다른 방향에 놓이므로, 한 방향만 보는 스칼라(비율·차이 등)는 반드시 하나를 놓칩니다.

관계행렬을 쓰는 이유는 **좌표계 불변성**입니다. 클라이언트마다 임베딩 기저가 달라 프로토타입을
직접 비교하면 무해한 좌표계 차이가 신호를 덮지만, 클래스 쌍 사이의 각도는 회전에 불변이라
독립적으로 학습된 모델끼리도 비교됩니다.

### 라운드별 절차

**① Probe 준비** (최초 1회, 고정)

학습셋에서 클래스당 20장을 균등 추출하고 **모든 클라이언트의 학습 데이터에서 제외**합니다
(train-holdout). 시드로 고정되어 전 라운드 동일하며, 랜덤 augmentation 없이 결정론적으로 씁니다.
서버가 별도 데이터셋을 갖는다는 가정(FLTrust의 root dataset)보다 약한 가정입니다.

**② 표현 축 — 클래스 관계 기하의 이탈**

각 업데이트 `w_i`를 템플릿 모델에 로드하고 probe를 **한 번** 통과시킵니다(입력 교란 없음).
클래스별 평균 임베딩으로 프로토타입을 만들고 코사인 관계행렬을 세웁니다.

```
p_c^(i) = (1/|D_c|) Σ_{x∈D_c} f_i(x)             클래스 c 프로토타입 (D차원)
R_i[a,b] = cos(p_a^(i), p_b^(i))                  (K × K)
```

상삼각만 씁니다 — 대각은 상수 1, 하삼각은 대칭 중복이라 고유한 클래스 쌍 `K(K-1)/2`개만 남습니다
(CIFAR-10은 45차원, CIFAR-100은 4,950차원).

자기 자신과 비교하지 않고 **그 라운드에 모인 클라이언트들 사이에서** 상대 이탈을 잽니다.

```
Ru  = R[:, 상삼각]
Rz  = (Ru - mean_over_clients) / std_over_clients      열별 또래 표준화
a_i = log ‖ Rz_i - median_over_clients(Rz) ‖₂
```

스케일은 평균·표준편차로 맞추고(차원마다 산포가 다르므로) 기준점은 이상치에 강한 중앙값을 씁니다.

**③ 파라미터 축 — 업데이트 크기**

```
b_i = log ‖ w_i - w_glob ‖₂
```

부동소수 파라미터 중 BatchNorm의 `running_mean`/`running_var`는 제외합니다 — 정규화 통계라
스케일이 완전히 달라 포함하면 신호를 덮습니다.

**④ 2축 마할라노비스**

```
x_i = (a_i, b_i)
u_i = x_i - median_j(x_j)

d0_i = ‖ u_i / std(u) ‖₂                          표준화 유클리드 거리
S    = Cov({u_j : j ∈ argsort(d0)[:n-2]})         가장 먼 2개를 빼고 추정 (trim-2)
S    = S + 1e-6 · tr(S) · I                       수치 안정화

M_i  = sqrt( u_iᵀ S⁻¹ u_i )
```

trim-2는 악성이 자기 방향의 분산을 부풀려 스스로를 정상으로 만드는 **마스킹**을 막습니다.
`n=10`이면 8명으로 공분산을 추정합니다.

**⑤ 선택 · 집계 · 노이즈**

```
sel     = argsort(M)[0 : 0.7n]                    또래와 비슷한 하위 70%만 사용
w_glob  = FedAvg({w_i}_{i∈sel})
w_glob += eps,  eps ~ N(0, sigma^2 I)
sigma   = 0.1 * median_{i∈전체} ‖w_i - w_glob‖ / sqrt(P)
```

`n=10`이면 가장 의심스러운 3명을 버립니다. 악성 비율이 30% 미만이라는 가정이 여기 들어갑니다.

노이즈는 **서버가 집계 후 한 번** 주입합니다. `sigma`의 median은 선택된 클라이언트가 아니라
**전체 클라이언트**의 업데이트 norm 기준입니다 — 그 라운드의 전형적 업데이트 크기를 재는 것이
목적이라 걸러지기 전 분포를 씁니다. `P`는 노이즈 대상 파라미터 수로 BatchNorm 통계는 제외합니다.
파라미터마다 `N(0, sigma^2)`을 넣으면 전체 노이즈 벡터의 L2 크기가 `sigma*sqrt(P)`이므로,
위와 같이 두면 노이즈가 전형적 업데이트의 약 10%가 됩니다.

> FLAME(USENIX Sec'22)의 원식은 `sigma = lambda * S`로 `sqrt(P)`로 나누지 않아 노이즈 벡터가
> `sqrt(P)`배 커집니다. 학습된 모델에 그대로 적용하면 파괴됩니다(실측 84.5% → 9.6%).
> 재현용으로 `--pb_noise_mode flame` 을 남겨 두었습니다.

### 설정

```
--pb_rule 1 --pb_rule_dets maha
--pb_perturb_peer 1 --pb_aug_abs 0 --pb_no_anchor 1
--pb_probe_n <클래스당 20장>  --pb_rule_drop 0.0  --pb_rule_keep 0.7  --noise 0.1
```

하이퍼파라미터는 셋뿐입니다 — probe 크기(클래스당 20장), keep 비율 0.7, 노이즈 계수 0.1.

축 단독 ablation용으로 `maha_a`(표현 축만) / `maha_b`(파라미터 축만) 를 제공합니다. 2축과 완전히
같은 중심화·스케일을 쓰고 공분산 결합만 뺀 것이라, 차이가 곧 두 축을 함께 보는 것의 기여분입니다.

### 설계 근거 (모두 실측)

| 선택 | 근거 |
|---|---|
| **2축 결합** | 축 단독으로는 90% 제거율 문턱을 못 넘는 칸이 생김 — 표현 4/12칸, 파라미터 1/12칸, 2축 1/12칸이며 실패 칸이 서로 다름 |
| **또래 비교(peer)** | 자기차이 `‖R_clean − R_pert‖`보다 우수 (CIFAR-10 LGA 17.5→3.1) |
| **앵커 없음** | 앵커 상수는 robust-z에서 상쇄되어 결과에 영향이 없음 |
| **log 공간** | 두 축의 스케일이 10^3배 이상 다르고 분포가 오른쪽으로 치우쳐 있음 |
| **trim-2 공분산** | 악성 1명이 자기 축 분산을 부풀려 스스로를 정상화하는 마스킹 차단 |
| **낮은 점수 선택** | 악성의 순위가 상위에 몰림. 반대로 하면 제거율 85%→15% |
| **probe 20장/클래스** | 40장으로 늘려도 이득 없음 — 방어자 가정이 약할수록 유리 |
| **keep 0.7** | 대안 조합과 5시드 비교 결과 동등하되 main accuracy가 더 높음 |
| **릿지 1e-6·tr(S)** | 축별 비례 릿지·릿지 제거와 12칸 비교 결과 제거율 동일(98.8%) — 순위 기반이라 무영향 |

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

### `observation/`

두 축 임베딩 관측 그림 도구. 방어가 런 중 기록한 덤프(`--pb_dump_relmat`)를 읽어
2D 평면·축 단독 그림과 축별 분리력 표를 만듭니다. 자체 README·설정·합성 데이터 생성기를
포함해 이 저장소 밖에서도 단독으로 동작합니다.

```bash
cd observation
python example/make_synthetic.py --out dumps   # 데이터 없이 시험
cp config.example.json config.json
bash run_all.sh
```

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
| `--pb_rule_dets` | detector 목록. 본 방법은 `maha`. 축 단독 ablation은 `maha_a`/`maha_b` |
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
  --pb_rule_dets maha --pb_aug_abs 0 --pb_perturb_peer 1 \
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

**5 seeds · 300 rounds · malicious 10% · non-IID (q=0.5 for CIFAR-10, q=0.2 for CIFAR-100)**
값은 마지막 10라운드 평균의 시드 간 평균 ± 표준편차입니다.

### BSR (%) — 낮을수록 좋음

| 셀 | 공격 | 무방어 | FLAME | FLTrust | mKrum | FLARE | **Ours** |
|---|---|---|---|---|---|---|---|
| CIFAR-10 / ResNet-18 | LGA | 97.0±0.2 | 97.8±0.3 | 93.7±7.9 | 96.7±1.6 | 97.9±0.2 | **3.8±0.9** |
| | LPA | 96.1±0.5 | 92.3±1.2 | 93.2±1.4 | 92.8±1.8 | 93.6±0.9 | **4.8±1.4** |
| | BadNet | 98.3±0.1 | 5.1±1.0 | 61.8±27.1 | 3.9±0.8 | 98.3±0.1 | **3.5±0.6** |
| | DAA | 97.7±0.1 | 4.8±0.9 | 67.7±31.2 | 3.6±0.8 | 97.5±0.4 | **3.6±0.7** |
| CIFAR-10 / VGG-19 | LGA | 96.1±0.3 | 96.3±1.8 | 84.1±27.0 | 91.1±7.3 | 96.5±0.5 | **3.0±0.5** |
| | LPA | 94.4±0.2 | 69.0±9.1 | 93.6±1.6 | 70.6±16.1 | 73.6±39.7 | 25.7±38.2 ✗ |
| | BadNet | 97.8±0.1 | 3.9±2.6 | 56.4±17.6 | 5.1±1.5 | 56.0±42.9 | **3.1±0.4** |
| | DAA | 96.7±0.2 | 5.2±2.6 | 23.5±15.4 | 6.3±1.8 | 3.1±0.4 | **2.8±0.5** |
| CIFAR-100 / ResNet-18 | LGA | 99.2±0.2 | 100.0±0.0 | 99.4±0.6 | 99.2±1.1 | 98.5±1.2 | **0.4±0.0** |
| | LPA | 95.3±3.2 | 57.1±23.1 | 78.8±8.9 | 80.1±29.4 | 90.4±6.9 | **0.5±0.1** |
| | BadNet | 100.0±0.0 | 0.5±0.1 | 79.5±21.8 | 0.4±0.0 | 97.0±6.4 | **0.4±0.1** |
| | DAA | 99.9±0.0 | 0.4±0.1 | 85.2±9.2 | 0.4±0.1 | 98.8±1.7 | **0.4±0.0** |
| **평균** | | 97.4 | 44.4 | 76.4 | 45.9 | 83.4 | **4.3** |

### Main accuracy (%) — 높을수록 좋음

| 셀 | 무방어 | FLAME | FLTrust | mKrum | FLARE | **Ours** |
|---|---|---|---|---|---|---|
| CIFAR-10 / ResNet-18 | 78.1 | 70.8 | 78.7 | 72.1 | **80.9** | 74.0 |
| CIFAR-10 / VGG-19 | 81.1 | 54.8 | 82.1 | 69.2 | **84.8** | 78.5 |
| CIFAR-100 / ResNet-18 | 64.6 | 55.8 | 63.6 | 58.7 | **65.2** | 58.7 |
| **평균** | 74.6 | 60.5 | 74.8 | 66.7 | **77.0** | 70.4 |

BSR을 실제로 낮추는 세 방법만 비교하면 Ours가 방어력에서 10배 앞서면서 정확도도 가장 높습니다.
FLARE·FLTrust의 main accuracy가 높은 것은 **방어를 거의 하지 않아 무방어에 가깝기 때문**입니다.

```
        BSR      MAIN
Ours     4.3     70.4
FLAME   44.4     60.5
mKrum   45.9     66.7
```

### 축별 분리력

각 축을 단독으로 썼을 때의 악성 제거율입니다 (`observation/report_table.py` 가 재현).

| 셀 | 공격 | `‖Δw‖` | 파라미터 단독 | `‖F‖` | 표현 단독 | 2축 |
|---|---|---|---|---|---|---|
| CIFAR-10 / VGG-19 | LGA | 0.11x | 100% | 1.95x | 78% | 100% |
| | BadNet | 3.05x | 100% | 1.48x | 56% | 100% |
| | DAA | 2.02x | 100% | 1.45x | 54% | 100% |
| | LPA | 1.00x | 66% | 3.76x | 82% | 86% |
| | Adaptive | 1.00x | **10%** | 5.11x | **100%** | 100% |

파라미터 축이 대부분을 담당하고, **업데이트 크기를 정상과 같게 유지하는 공격(LPA·Adaptive)에서만
표현 축이 구제**합니다. `Adaptive`는 ICLR 2024 *Backdoor Federated Learning by Poisoning
Backdoor-critical Layers* (Zhuang et al.) 의 적응형 변형으로, 공격자가 조작 모델이 파라미터 공간
방어에 걸리지 않음을 스스로 확인한 뒤 제출합니다. 그 최적화의 목적함수에 표현 기하는 들어 있지
않습니다.

### 적용 범위와 한계

**대상**: 백도어를 심는 과정에서 클래스 기하 또는 업데이트 크기 중 최소 하나에 흔적을 남기는 공격.
데이터 오염 계열(BadNet, DBA, DAA)은 업데이트가 커지고, 층 교체 계열(LPA, Adaptive)은 기하가
틀어지며, LGA는 업데이트가 작아지므로 모두 평면 밖으로 나갑니다.

**실패 셀**: VGG-19 / CIFAR-10 / LPA (BSR 25.7 ± 38.2). 5시드 중 4회는 방어되고(7.2~10.8)
seed 1만 실패(94.0)하는 이봉 분포입니다. 이 셀은 두 축이 모두 약합니다 — 파라미터 66%,
표현 82%, 2축 86%로 아래 문턱을 겨우 밑돕니다.

**90% 제거율 문턱.** 악성 제거율이 90% 이상이어야 방어가 유지됩니다(제거율 91% → BSR 4.7,
85% → 81, 84% → 85, 82% → 92). 문턱을 밑돌면 **양성 피드백**이 발생합니다 — 백도어가 한 번
진입하면 정상 클라이언트도 오염된 글로벌에서 학습을 시작해 또래 기준점 자체가 오염되고, 탐지가
스스로 무너져 되돌아오지 못합니다. 이봉 분포의 메커니즘이 이것입니다.

이 때문에 **탐지 성능을 무방어 런에서 측정해서는 안 됩니다.** 공격이 성공한 뒤에는 악성이 오히려
가장 전형적인 클라이언트가 되어(실측: BadNet의 파라미터 축 z가 +811σ → −0.0σ) 어떤 방어든
AUC 0.5가 나옵니다. 방어가 살아 있는 궤적에서 재야 합니다.

**회피 가능한 공격.** 이동량을 손실에 직접 규제하는 공격(`--attack daa`, `β‖θ−θ_prev‖²`)은
크기를 정상 분포 **안에** 정확히 맞추고 기하 변화도 라운드당 미미하게 유지하여 두 축을 모두
빠져나갈 수 있습니다. 크기 극단(LGA 0.11x, BadNet 3.05x)과 달리 0.998x 로 수렴하므로,
본 방법의 그물은 "크기의 양극단 + 크기 보존형 층 교체"까지이며 그 바깥이 남은 한계입니다.

---

## 6. 환경

```bash
pip install -r requirements.txt
```
PyTorch, torchvision, numpy, pandas, matplotlib, wandb(선택)가 필요합니다.
`--wandb_project <name>` 을 주면 wandb에 기록되고, 생략하면 로컬 로그만 남습니다.

---

## 7. 출처 및 라이선스

본 저장소는 LGA 공격 논문(ICCV 2025, *Stealthy Backdoor Attack in Federated Learning via
Adaptive Layer-wise Gradient Alignment*)의 공식 구현 [yqqhyqq/LGA](https://github.com/yqqhyqq/LGA)를
기반으로 합니다. 그 저장소가 포함한 `LPA` / `adaptive` 공격은 ICLR 2024 *Backdoor Federated
Learning by Poisoning Backdoor-critical Layers* (Zhuang et al.,
[openreview](https://openreview.net/pdf?id=AJBGSVSTT2)) 의 구현입니다.
FL 학습 루프(`main_fed.py`), 공격 구현(`client/`), 모델(`models/`), 기존 방어 baseline
(`defense/` 중 `flame`, `fltrust`, `flare`, `mkrum`, `RLRorigin`, `multimetric`)은 원본 저장소에서
가져와 수정한 것입니다.

**본 연구의 기여**
- `defense/proto_bandit.py` — 표현·파라미터 2축 방어 (신규 작성)
- `observation/` — 2축 관측 도구 (신규 작성, 단독 실행 가능)
- `utils/options.py` 의 `--pb_*` 옵션군
- `models/Nets.py` VGG `num_classes` 인자화 및 4개 데이터셋 지원
- `main_fed.py` MNIST/F-MNIST 지원(Grayscale 3채널, VGG용 32×32 리사이즈),
  FLTrust/FLARE용 root dataset 생성
- `client/add_trigger.py` DBA 트리거 파라미터화 (`--dba_size`, `--dba_gap`)
- `client/Attacker.py` DBA 조각 라운드 회전 수정, `--ada_assume`(공격자 상정 방어와 실제 방어 분리)
- `client/MaliciousUpdate.py` LFA 첫 라운드 `attack_layers=None` 가드

라이선스는 원본 저장소의 `LICENSE`를 따릅니다.
