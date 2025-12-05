# BlindOver Server

음료 이미지 분류를 위한 딥러닝 기반 FastAPI 서버입니다. 33종의 한국 음료를 이미지로 인식하여 분류합니다.

## 프로젝트 소개

BlindOver는 시각장애인을 위한 음료 인식 서비스입니다. 사용자가 음료 이미지를 업로드하면 딥러닝 모델이 음료를 분류하여 결과를 반환합니다.

## 기술 스택

| 구분 | 기술 |
|------|------|
| 언어 | Python 3.x |
| 웹 프레임워크 | FastAPI 0.96.0 |
| ASGI 서버 | Uvicorn 0.22.0 |
| 딥러닝 | PyTorch 2.0.1, TorchVision 0.15.2 |
| 이미지 처리 | Pillow 9.5.0 |
| 로깅 | Loguru 0.7.0 |

## 프로젝트 구조

```
blindover_server/
├── main.py                    # FastAPI 애플리케이션 진입점
├── inference.py               # 모델 추론 및 이미지 처리 로직
├── requirements.txt           # Python 의존성 패키지
├── models/                    # 신경망 모델 아키텍처
│   ├── efficientnet.py        # EfficientNetV2 (20.3M 파라미터)
│   ├── mobilenet.py           # MobileNetV3 (1.62M 파라미터)
│   ├── shufflenet.py          # ShuffleNetV2 (0.44M 파라미터) - 기본 모델
│   └── mnasnet.py             # MNASNet (5.12M 파라미터)
├── weights/                   # 사전 학습된 모델 가중치 (~134MB)
│   ├── shufflenet_weight.pt
│   ├── mobilenet_weight.pt
│   ├── efficientnet_weight.pt
│   ├── mnasnet_weight.pt
│   └── quantized_*.pt         # 양자화된 모델 (용량 감소)
├── quantization/              # 모델 양자화 유틸리티
│   ├── quantization.py
│   ├── quantized_models.py
│   └── utils.py
└── utils/
    └── dataset.py             # 데이터 로딩 및 전처리
```

## 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/BlindOver/blindover_server.git
cd blindover_server
```

### 2. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 실행 방법

### 개발 서버 실행
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 프로덕션 서버 실행
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### CLI로 이미지 분류 테스트
```bash
python inference.py --model_name shufflenet --src 이미지경로.jpg --weight ./weights/shufflenet_weight.pt
```

## API 명세

### 음료 이미지 분류

**엔드포인트**: `POST /photo`

**요청**:
```bash
curl -X POST "http://localhost:8000/photo" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@음료이미지.jpg"
```

**응답**:
```json
"코카 콜라"
```

### 헬스 체크

**엔드포인트**: `GET /`

**응답**: 서버 상태 메시지

## 지원 음료 목록 (33종)

| 번호 | 음료명 | 번호 | 음료명 |
|------|--------|------|--------|
| 0 | 2% | 17 | 펩시 |
| 1 | 박카스 | 18 | 펩시 제로 |
| 2 | 칠성 사이다 | 19 | 포카리 스웨트 |
| 3 | 칠성 사이다 제로 | 20 | 파워에이드 |
| 4 | 초코 우유 | 21 | 레드불 |
| 5 | 코카 콜라 | 22 | 식혜 |
| 6 | 데미소다 사과 | 23 | 스프라이트 |
| 7 | 데미소다 복숭아 | 24 | 스프라이트 제로 |
| 8 | 솔의눈 | 25 | 딸기 우유 |
| 9 | 환타 오렌지 | 26 | 비타 500 |
| 10 | 게토레이 | 27 | 브이톡 블루레몬 |
| 11 | 제티 | 28 | 브이톡 복숭아 |
| 12 | 맥콜 | 29 | 웰치스 포도 |
| 13 | 우유 | 30 | 웰치스 오렌지 |
| 14 | 밀키스 | 31 | 웰치스 화이트그레이프 |
| 15 | 밀키스 제로 | 32 | 제로 콜라 |
| 16 | 마운틴 듀 | | |

## 모델 정보

### 사용 가능한 모델

| 모델명 | 파라미터 수 | 특징 |
|--------|-------------|------|
| **ShuffleNetV2** | 0.44M | 기본 모델, 경량화, 빠른 추론 |
| MobileNetV3 | 1.62M | 균형잡힌 성능 |
| MNASNet | 5.12M | 모바일 최적화 |
| EfficientNetV2 | 20.3M | 높은 정확도 |

### 모델 양자화

양자화된 모델은 약 3.5~4배 용량 감소:

| 모델 | 원본 크기 | 양자화 크기 | 감소율 |
|------|-----------|-------------|--------|
| ShuffleNetV2 | 5.5 MB | 1.5 MB | 3.58x |
| MobileNetV3 | 22.1 MB | 5.6 MB | 3.93x |

## 이미지 처리 파이프라인

1. **패딩 (Padding)**: 비정사각형 이미지를 검은색으로 패딩하여 비율 유지
2. **리사이즈**: 224x224 픽셀로 크기 조정
3. **텐서 변환**: PIL 이미지를 PyTorch 텐서로 변환
4. **배치 차원 추가**: 추론을 위한 차원 추가
5. **추론**: Softmax 확률 계산 후 최고 확률 클래스 반환

## 학습 데이터셋 구조

새로운 모델 학습 시 데이터셋 구조:

```
dataset/
├── train/
│   ├── 2%/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── 박카스/
│   └── ... (33개 클래스)
├── valid/
└── test/
```

## 주요 의존성

```
fastapi==0.96.0
uvicorn==0.22.0
torch==2.0.1
torchvision==0.15.2
Pillow==9.5.0
python-multipart==0.0.6
loguru==0.7.0
pydantic==1.10.8
```

## 기여 방법

1. 이 저장소를 Fork 합니다
2. 새로운 브랜치를 생성합니다 (`git checkout -b feature/새기능`)
3. 변경사항을 커밋합니다 (`git commit -m 'Feat: 새기능 추가'`)
4. 브랜치에 Push 합니다 (`git push origin feature/새기능`)
5. Pull Request를 생성합니다

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 참고 자료

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/ko/)
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [PyTorch 양자화 문서](https://pytorch.org/docs/stable/quantization.html)
- [ShuffleNetV2 논문](https://arxiv.org/abs/1807.11164)
- [MobileNetV3 논문](https://arxiv.org/abs/1905.02244)
