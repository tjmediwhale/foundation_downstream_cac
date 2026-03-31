# CAC Image Inference (Standalone Folder)

`foundation_downstream_task/cac`에서 학습한 모델 체크포인트를 사용해,
이미지 1장을 넣으면 CAC 양성 확률(`CAC>0`)을 바로 추론하는 독립 실행 폴더입니다.

## 폴더 구조
```text
cac_inference/
├── configs/
│   └── inference.yaml
├── scripts/
│   └── infer_image.py
├── src/
│   └── cac_inference/
│       ├── model/
│       └── utils/
├── weights/
└── requirements.txt
```

## 1) 설치
```bash
cd /home/tj/Research/foundation_downstream_task/cac_inference
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 준비물
1. Downstream 체크포인트:  
`/home/tj/Research/foundation_downstream_task/cac/output/<run_name>/checkpoints/best.pt`

2. Foundation(DINOv3) 체크포인트: 학습 때 사용한 backbone 가중치 파일

3. DINOv3 코드: `third_party/dinov3`에 clone
```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
```

그리고 `configs/inference.yaml`에서 경로를 맞춰주세요.

## 3) 단일 이미지 추론
```bash
python3 scripts/infer_image.py \
  --image /absolute/path/to/fundus.jpg \
  --config configs/inference.yaml
```

경로를 CLI에서 직접 오버라이드할 수도 있습니다.
```bash
python3 scripts/infer_image.py \
  --image /absolute/path/to/fundus.jpg \
  --downstream_checkpoint /absolute/path/to/best.pt \
  --foundation_checkpoint /absolute/path/to/foundation.pt \
  --dinov3_repo /absolute/path/to/dinov3 \
  --threshold 0.5
```

JSON 파일로 저장:
```bash
python3 scripts/infer_image.py \
  --image /absolute/path/to/fundus.jpg \
  --output_json outputs/result.json
```

## 4) GitHub에 새 repo로 올리기
```bash
cd /home/tj/Research/foundation_downstream_task/cac_inference
git init
git add .
git commit -m "Add standalone CAC image inference package"
git branch -M main
git remote add origin git@github.com:<your-id>/<repo-name>.git
git push -u origin main
```

주의: `weights/*.pt`는 `.gitignore`에 포함되어 있으므로 모델 파일은 업로드되지 않습니다.

