# How to Run nu2flows on KNU Server

KNU 서버(AlmaLinux 기반)에서 nu2flows 머신러닝 패키지를 CPU 환경에서 실행하는 방법을 설명합니다.

## 환경 정보

- **OS**: AlmaLinux 9.6 (Linux 5.14.0-570.33.2.el9_6.x86_64)
- **Shell**: /bin/bash
- **권한**: 일반 사용자 (non-sudo)
- **실행 환경**: CPU 전용 (GPU 비활성화)
- **컨테이너**: Singularity/Apptainer

## 1. nu2flows 설치 및 빌드

### 1.1 프로젝트 클론
```bash
# 프로젝트 디렉토리로 이동
cd /u/user/gcho/TopPhysics/CPV/MachineLearning/

# nu2flows 저장소 클론 (또는 복사)
git clone <repository-url> nu2flows
# 또는 기존 소스코드가 있다면 해당 디렉토리 사용
```

### 1.2 Singularity 컨테이너 빌드

#### 방법 1: 정의 파일로 새로 빌드 (권장)
```bash
cd nu2flows

# 관리자 권한으로 컨테이너 빌드
sudo singularity build nu2flows.sif nu2flows_cpu.def
```

**빌드 과정:**
- 베이스 이미지 다운로드: `pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime`
- 시스템 패키지 설치: `build-essential`, `hdf5-tools`, `texlive` 등
- CPU 전용 PyTorch 설치
- Python 패키지 설치: `lightning`, `wandb`, `hydra-core` 등
- 소스코드 복사 및 환경 설정

**예상 소요 시간:** 30-60분 (네트워크 속도에 따라)

#### 방법 2: 기존 빌드된 컨테이너 사용
```bash
# 이미 빌드된 .sif 파일이 있다면
ls -la nu2flows.sif

# 파일 크기 확인 (보통 3-5GB)
du -sh nu2flows.sif
```

### 1.3 데이터 준비
```bash
# 데이터 디렉토리 생성
mkdir -p data/8113516/

# 데이터 파일 복사 (예시)
cp /path/to/your/train_1.h5 data/8113516/

# 데이터 파일 확인
ls -la data/8113516/
```

### 1.4 빌드 검증
```bash
# 컨테이너 실행 테스트
CUDA_VISIBLE_DEVICES="" singularity run nu2flows.sif

# Python 환경 확인
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import lightning; print(f'Lightning: {lightning.__version__}')"

# 컨테이너 종료
exit
```

## 2. 프로젝트 설정

### 디렉토리 구조
```
/u/user/gcho/TopPhysics/CPV/MachineLearning/
└── nu2flows/
    ├── nu2flows.sif          # Singularity 컨테이너 파일
    ├── scripts/
    │   └── train.py          # 메인 훈련 스크립트
    ├── configs/              # 설정 파일들
    ├── src/                  # 소스 코드
    ├── data/                 # 데이터 파일들
    └── mltools/              # 유틸리티 라이브러리
    ├── nu2flows_cpu.def      # 컨테이너 빌드 정의 파일
```

### 환경 변수 설정
```bash
# 프로젝트 루트 디렉토리로 이동
cd /u/user/gcho/TopPhysics/CPV/MachineLearning/nu2flows

# PYTHONPATH 설정 (선택사항)
export PYTHONPATH=/u/user/gcho/TopPhysics/CPV/MachineLearning/nu2flows:$PYTHONPATH
```

## 3. Singularity 컨테이너 실행

### CPU 전용 모드로 컨테이너 시작
```bash
# GPU 비활성화하여 컨테이너 실행 (권장 방법)
CUDA_VISIBLE_DEVICES="" singularity run ../nu2flows_cpu.sif

# 또는 현재 디렉토리에 .sif 파일이 있다면
CUDA_VISIBLE_DEVICES="" singularity run nu2flows.sif
```

**주의사항:**
- **KNU 서버는 Apptainer 사용**: `--no-gpu` 옵션 지원 안 함
- **반드시 `CUDA_VISIBLE_DEVICES=""`** 사용해야 함
- `--nv` 플래그 없이 실행하면 기본적으로 GPU 비활성화됨

## 4. 설정 파일 구성

### 주요 설정 파일들

#### `configs/trainer/default.yaml`
```yaml
_target_: lightning.Trainer
min_epochs: 1
max_epochs: 1
accelerator: cpu              # CPU 강제 사용
devices: 1
precision: 32
num_sanity_val_steps: 0       # validation 활성화
log_every_n_steps: 10         # 로그 출력 간격
default_root_dir: ${paths.output_dir}
```

#### `configs/model/default.yaml`
```yaml
_target_: src.models.nuflows.NuFlows
input_dimensions:
  misc: 2
  met: 2
  leptons: 8
  jets: 50
target_dimensions:
  neutrino: 3
  antineutrino: 3

transformer_config:
  pack_inputs: False          # 중요: 차원 문제 방지
  unpack_output: True
  num_layers: 5
  layer_config:
    attn_config:
      num_heads: 4            # CPU 환경에 맞게 축소
      dropout: 0

scheduler:
  total_steps: -1             # 자동 계산
  warmup_steps: 1             # CPU 환경에 맞게 축소
```

#### `configs/datamodule/default.yaml`
```yaml
loader_conf:
  pin_memory: False           # CPU에서는 False
  batch_size: 1024            # 적절한 배치 크기
  num_workers: 6              # CPU 코어 수에 맞게 조정
val_frac: 0.2                 # 검증 데이터 비율
```

#### `configs/loggers/default.yaml`
```yaml
wandb:
  _target_: lightning.pytorch.loggers.wandb.WandbLogger
  entity: guk-cho3223         # 사용자 계정
  offline: true               # 오프라인 모드 (권한 문제 해결)
  project: ttbar_nu2flows
```

## 5. 훈련 실행

### Weights & Biases 설정 (선택사항)
```bash
# wandb 로그인 (온라인 로깅을 원할 경우)
wandb login

# API 키 입력 후 확인
wandb whoami
```

### 훈련 시작
```bash
# 컨테이너 내에서 실행
python3 scripts/train.py
```

### 예상 출력
```
[2025-08-26 05:21:15,080][__main__][INFO] - Setting up full job config
GPU available: False, used: False
...
Epoch 0/0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 184/184 0:05:04 • 0:00:00 0.67it/s
```

## 6. 출력 파일 구조

### 메인 출력 디렉토리
```
/u/user/gcho/TopPhysics/CPV/MachineLearning/nu2flows/
└── ttbar_nu2flows/                    # project_name
    └── toCompareMac/                  # network_name
        ├── checkpoints/               # 모델 체크포인트
        │   ├── best_epoch_000.ckpt
        │   └── last.ckpt
        ├── wandb/                     # Weights & Biases 로그
        │   └── offline-run-*/
        ├── full_config.yaml           # 전체 설정 백업
        └── .hydra/                    # Hydra 설정 파일들
            ├── config.yaml
            ├── hydra.yaml
            └── overrides.yaml
```

### 파일 설명

- **`checkpoints/`**: 훈련된 모델 가중치
  - `best_epoch_000.ckpt`: 최고 성능 모델
  - `last.ckpt`: 마지막 에포크 모델

- **`wandb/`**: 실험 추적 로그
  - `offline-run-*/`: 오프라인 모드 로그
  - 온라인 동기화: `wandb sync wandb/offline-run-*/`

- **`full_config.yaml`**: 실행 시 사용된 전체 설정
- **`.hydra/`**: Hydra 프레임워크 설정 파일들

## 7. 문제 해결

### 일반적인 오류와 해결책

#### 1. CUDA/GPU 경고
```
WARNING: The NVIDIA Driver was not detected.
```
**해결책**: 정상적인 메시지. CPU 모드에서는 무시해도 됨.

#### 2. wandb 권한 오류
```
Error uploading run: returned error 403: permission denied
```
**해결책**: `configs/loggers/default.yaml`에서 `offline: true` 설정

#### 3. 차원 불일치 오류
```
ValueError: not enough values to unpack (expected 3, got 2)
```
**해결책**: `configs/model/default.yaml`에서 `pack_inputs: False` 확인

#### 4. matplotlib 호환성 오류
```
AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'
```
**해결책**: 이미 수정됨. `mltools/mltools/plotting.py`에서 fallback 메커니즘 적용

### 성능 최적화

#### CPU 환경 최적화 설정
- `num_workers: 6`: CPU 코어 수에 맞게 조정
- `batch_size: 1024`: 메모리에 맞게 조정
- `num_heads: 4`: GPU 대비 축소
- `warmup_steps: 1`: 적은 배치 수에 맞게 축소

## 8. 추가 명령어

### 로그 확인
```bash
# wandb 오프라인 로그 동기화
wandb sync /path/to/wandb/offline-run-*

# 훈련 로그 확인
tail -f .hydra/job.log
```

### 모델 내보내기/추론
```bash
# export.yaml 설정 후
python3 scripts/export.py
```

### 디스크 용량 확인
```bash
# 출력 디렉토리 크기 확인
du -sh ttbar_nu2flows/

# 체크포인트 파일 크기
ls -lh ttbar_nu2flows/toCompareMac/checkpoints/
```

## 9. 성공 확인

훈련이 성공적으로 완료되면:

1. ✅ 진행률 표시: `184/184 배치 완료`
2. ✅ 체크포인트 생성: `checkpoints/` 디렉토리에 `.ckpt` 파일들
3. ✅ wandb 로그 생성: `wandb/` 디렉토리에 실험 로그
4. ✅ 설정 백업: `full_config.yaml` 파일 생성

## 10. 빌드 시 주의사항

### 컨테이너 빌드 요구사항
- **관리자 권한**: `sudo` 권한 필요 (컨테이너 빌드 시에만)
- **디스크 공간**: 최소 10GB 이상 (빌드 과정에서 임시 파일 생성)
- **네트워크 연결**: 베이스 이미지 및 패키지 다운로드용
- **메모리**: 최소 4GB 이상 권장

### 빌드 문제 해결

#### 1. 권한 오류
```bash
FATAL: could not use fakeroot: no user namespace available
```
**해결책**: `sudo` 권한으로 빌드하거나 관리자에게 요청

#### 2. 네트워크 오류
```bash
ERROR: Failed to download base image
```
**해결책**: 
- 네트워크 연결 확인
- 프록시 설정 확인
- 방화벽 설정 확인

#### 3. 디스크 공간 부족
```bash
ERROR: No space left on device
```
**해결책**:
- `/tmp` 디렉토리 정리
- 빌드 위치를 충분한 공간이 있는 디렉토리로 변경

### nu2flows_cpu.def 파일 설명

#### 주요 구성 요소
```dockerfile
Bootstrap: docker                    # Docker 이미지 기반
From: pytorch/pytorch:2.3.1-...    # 베이스 이미지

%environment                        # 환경 변수 설정
    export CUDA_VISIBLE_DEVICES=""  # GPU 비활성화

%post                              # 빌드 시 실행할 명령어들
    apt-get update                 # 시스템 패키지 업데이트
    pip install torch --index-url  # CPU 전용 PyTorch 설치

%files                             # 호스트에서 컨테이너로 복사할 파일
    . /nu2flows                    # 현재 디렉토리를 /nu2flows로 복사

%runscript                         # 컨테이너 실행 시 기본 명령어
    cd /nu2flows
    exec /bin/bash
```

## 11. 참고사항

- **데이터 경로**: `/u/user/gcho/TopPhysics/CPV/MachineLearning/nu2flows/data/8113516/`
- **컨테이너 환경**: Python 3.10, PyTorch Lightning, Hydra 프레임워크
- **실행 시간**: CPU 환경에서 약 5-10분 (데이터 크기에 따라)
- **메모리 사용량**: 약 2-4GB (배치 크기에 따라)
- **컨테이너 크기**: 약 3-5GB (빌드 후 .sif 파일)

### 완전한 설치 체크리스트

#### 초기 설치 (한 번만 수행)
- [ ] 프로젝트 소스코드 준비
- [ ] `nu2flows_cpu.def` 파일 확인
- [ ] `sudo singularity build nu2flows.sif nu2flows_cpu.def` 실행
- [ ] 데이터 파일 준비 및 복사
- [ ] 컨테이너 실행 테스트

#### 매번 실행 시
- [ ] `cd /u/user/gcho/TopPhysics/CPV/MachineLearning/nu2flows`
- [ ] `CUDA_VISIBLE_DEVICES="" singularity run nu2flows.sif`
- [ ] `python3 scripts/train.py`

---

**작성일**: 2025-08-26  
**테스트 환경**: KNU 서버, AlmaLinux 9.6, Singularity 컨테이너  
**빌드/실행 상태**: 정상 작동 확인 완료 ✅
