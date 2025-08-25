# NuFlows 설치 및 실행 가이드

이 문서는 M1/M2 MacBook에서 NuFlows 패키지를 설치하고 실행하는 방법을 단계별로 설명합니다.

## 목차
1. [설치 과정](#1-설치-과정)
2. [설정 및 실행](#2-설정-및-실행)
3. [결과 확인](#3-결과-확인)
4. [코드 구조 및 작동 방법](#4-코드-구조-및-작동-방법)

---

## 1. 설치 과정

### 1.1 사전 요구사항
- macOS (M1/M2 MacBook 권장)
- Miniforge3 (conda 환경 관리)

### 1.2 설치 단계

#### Step 1: 가상환경 생성 및 활성화
```bash
# 가상환경 생성
conda create -n nu2flows python=3.11

# 가상환경 활성화
conda activate nu2flows
```

#### Step 2: 의존성 설치
```bash
# 프로젝트 루트 디렉토리로 이동
cd /path/to/nu2flows

# macOS용 의존성 설치
pip install -r requirements.mac.txt
pip install -r requirements.txt

# 로컬 패키지 설치 (선택사항)
pip install -e .
```

#### Step 3: 데이터 다운로드
```bash
# Zenodo에서 데이터 다운로드
# DOI: 10.5281/zenodo.8113516
# 다운로드한 파일을 data/ 디렉토리에 저장
```

#### Step 4: M1/M2 MacBook 호환성 수정
이미 적용된 수정사항들:
- ✅ Flash Attention 대체 구현
- ✅ float64 → float32 변환 (MPS 호환성)
- ✅ matplotlib 백엔드 호환성
- ✅ pin_memory 설정 최적화

---

## 2. 설정 및 실행

### 2.1 기본 설정 파일들

#### 주요 설정 디렉토리 구조:
```
configs/
├── train.yaml          # 메인 설정 파일
├── trainer/
│   └── default.yaml     # 학습 설정 (GPU/CPU, 에포크 등)
├── model/
│   └── default.yaml     # 모델 아키텍처 설정
├── datamodule/
│   └── default.yaml     # 데이터 로딩 설정
├── paths/
│   └── default.yaml     # 경로 설정
└── loggers/
    └── default.yaml     # 로깅 설정 (WandB)
```

### 2.2 실행 방법

#### 기본 실행 (MPS 가속 사용):
```bash
python3 scripts/train.py
```

#### CPU 모드로 실행:
```bash
python3 scripts/train.py trainer.accelerator=cpu
```

#### 테스트 실행 (빠른 확인):
```bash
python3 scripts/train.py +trainer.limit_train_batches=5 trainer.max_epochs=1
```

#### 시각화 없이 실행:
```bash
python3 scripts/train.py model.gen_validation=0
```

#### 설정 오버라이드 예시:
```bash
python3 scripts/train.py \
  trainer.max_epochs=10 \
  datamodule.loader_conf.batch_size=2048 \
  model.gen_validation=5
```

### 2.3 주요 설정 옵션

#### 학습 설정 (`configs/trainer/default.yaml`):
- `accelerator`: `mps` (GPU) 또는 `cpu`
- `max_epochs`: 최대 에포크 수
- `precision`: 정밀도 (32 권장)

#### 모델 설정 (`configs/model/default.yaml`):
- `gen_validation`: 검증 시각화 생성 빈도 (0=비활성화)
- `input_dimensions`: 입력 데이터 차원
- `target_dimensions`: 출력 데이터 차원

#### 데이터 설정 (`configs/datamodule/default.yaml`):
- `batch_size`: 배치 크기 (4096 기본값)
- `num_workers`: 데이터 로딩 워커 수
- `file_list`: 학습/테스트 데이터 파일 목록

---

## 3. 결과 확인

### 3.1 출력 디렉토리 구조
```
outputs/nu2flows/test_mac/
├── checkpoints/
│   ├── best_000.ckpt    # 최적 모델 체크포인트
│   └── last.ckpt        # 마지막 체크포인트
├── plots/
│   ├── hist_neutrino.png      # 뉴트리노 분포 히스토그램
│   ├── hist_antineutrino.png  # 반뉴트리노 분포 히스토그램
│   ├── corr_neutrino.png      # 뉴트리노 상관관계
│   └── corr_antineutrino.png  # 반뉴트리노 상관관계
├── wandb/               # WandB 로그 파일들
├── full_config.yaml     # 전체 실행 설정
├── train.log           # 학습 로그
└── train_finished.txt  # 완료 표시
```

### 3.2 결과 해석

#### 히스토그램 (`hist_*.png`):
- **파란색**: 실제값 (Ground Truth)
- **주황색**: 예측값 (Model Output)
- **x축**: 모멘텀 성분 (px, py, pz)
- **y축**: 빈도수

#### 상관관계 히트맵 (`corr_*.png`):
- 예측값과 실제값 간의 2D 분포
- 대각선에 가까울수록 정확한 예측
- 색상: 데이터 밀도 (로그 스케일)

#### 체크포인트:
- `best_000.ckpt`: 검증 손실 기준 최적 모델
- `last.ckpt`: 마지막 에포크 모델
- PyTorch Lightning 형식으로 저장

### 3.3 로그 모니터링

#### 실시간 로그 확인:
```bash
tail -f outputs/nu2flows/test_mac/train.log
```

#### WandB 동기화 (선택사항):
```bash
wandb sync outputs/nu2flows/test_mac/wandb/offline-run-*/
```

---

## 4. 코드 구조 및 작동 방법

### 4.1 스크립트 파일들

#### `scripts/train.py`
**목적**: 모델 학습 실행
**주요 기능**:
- Hydra 설정 로딩 및 통합
- 데이터 모듈 초기화
- 모델 아키텍처 구성
- PyTorch Lightning Trainer 설정
- 학습 실행 및 체크포인트 저장

**실행 흐름**:
1. 설정 파일들을 통합하여 전체 config 생성
2. 시드 설정으로 재현 가능성 보장
3. 데이터 모듈 인스턴스화 (H5DataModule)
4. 모델 인스턴스화 (NuFlows)
5. 콜백 및 로거 설정
6. Trainer로 학습 실행

#### `scripts/export.py`
**목적**: 학습된 모델로 예측 결과 내보내기
**주요 기능**:
- 체크포인트에서 모델 로드
- 테스트 데이터로 예측 수행
- 결과를 HDF5 파일로 저장

#### `scripts/export_onnx.py`
**목적**: 모델을 ONNX 형식으로 변환
**주요 기능**:
- PyTorch 모델을 ONNX로 내보내기
- 추론 최적화 및 배포 준비

#### `scripts/convert_root.py`
**목적**: ROOT 파일을 HDF5로 변환
**주요 기능**:
- 물리학 데이터 형식 변환
- 전처리 및 필터링

### 4.2 모델 아키텍처

#### `src/models/nuflows.py` - NuFlows 클래스
**구조**:
```
Input Data → Embedding Networks → Transformer → Normalizing Flow → Output
     ↓              ↓                ↓               ↓            ↓
  물리량 입력    차원 변환         특징 추출      확률 분포 학습   뉴트리노 예측
```

**주요 구성 요소**:
1. **Embedding Networks**: 각 물리량(jets, leptons, MET 등)을 고정 차원으로 변환
2. **Transformer Encoder**: 입력 간의 관계 학습
3. **Normalizing Flow**: 조건부 확률 분포 모델링
4. **Target Normalization**: 출력 정규화

#### `src/datamodules/dilepton.py` - H5DataModule
**기능**:
- HDF5 파일에서 데이터 로딩
- 배치 생성 및 전처리
- 학습/검증/테스트 분할

### 4.3 설정 시스템 (Hydra)

#### 설정 계층 구조:
```
train.yaml (메인)
├── trainer/default.yaml    # Lightning Trainer 설정
├── model/default.yaml      # 모델 하이퍼파라미터
├── datamodule/default.yaml # 데이터 로딩 설정
├── callbacks/default.yaml  # 학습 콜백
├── loggers/default.yaml    # 로깅 설정
└── paths/default.yaml      # 파일 경로
```

#### 오버라이드 방식:
```bash
# 명령행에서 설정 변경
python3 scripts/train.py trainer.max_epochs=50 model.gen_validation=5

# + 기호로 새 설정 추가
python3 scripts/train.py +trainer.limit_train_batches=10
```

### 4.4 데이터 플로우

```
Raw Data (ROOT) → HDF5 → DataLoader → Model → Predictions → Visualization
      ↓              ↓         ↓         ↓          ↓           ↓
   물리 시뮬레이션   변환 저장   배치 생성   신경망 학습   결과 예측   성능 분석
```

### 4.5 성능 최적화 팁

#### MPS 가속 최적화:
- `trainer.accelerator=mps` 사용
- `pin_memory=False` 설정 (이미 적용됨)
- 적절한 배치 크기 조정 (2048-4096)

#### 메모리 최적화:
```bash
# 작은 배치 크기로 메모리 사용량 줄이기
python3 scripts/train.py datamodule.loader_conf.batch_size=1024

# 워커 수 조정
python3 scripts/train.py datamodule.loader_conf.num_workers=4
```

#### 빠른 테스트:
```bash
# 적은 배치로 빠른 확인
python3 scripts/train.py +trainer.limit_train_batches=5 +trainer.limit_val_batches=2
```

---

## 문제 해결

### 일반적인 오류들:

#### 1. `ModuleNotFoundError: No module named 'hydra'`
**해결책**: 가상환경 활성화 확인
```bash
conda activate nu2flows
```

#### 2. `Cannot convert a MPS Tensor to float64`
**해결책**: 이미 수정됨 (float32 사용)

#### 3. `'FigureCanvasMac' object has no attribute 'tostring_rgb'`
**해결책**: 이미 수정됨 (호환성 있는 render_image 함수)

#### 4. 데이터 파일 없음
**해결책**: Zenodo에서 데이터 다운로드 (DOI: 10.5281/zenodo.8113516)

### 도움이 되는 명령어들:

```bash
# 전체 에러 스택 확인
HYDRA_FULL_ERROR=1 python3 scripts/train.py

# GPU 메모리 상태 확인
python3 -c "import torch; print(torch.backends.mps.is_available())"

# 설정 확인 (실제 실행 없이)
python3 scripts/train.py --config-path=configs --config-name=train.yaml --help
```

---

## 추가 자료

- **논문**: 
  - https://arxiv.org/abs/2207.00664
  - https://arxiv.org/abs/2307.02405
- **데이터**: https://doi.org/10.5281/zenodo.8113516
- **PyTorch Lightning**: https://lightning.ai/
- **Hydra**: https://hydra.cc/
- **WandB**: https://wandb.ai/

이 가이드를 통해 NuFlows 패키지를 성공적으로 실행하고 뉴트리노 모멘텀 예측 모델을 학습할 수 있습니다.
