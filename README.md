# CAC Inference (Standalone Folder)

`foundation_downstream_task/cac`에서 학습한 체크포인트를 사용해:
- CSV 기반 배치 추론
- 단일 이미지 추론

을 수행하는 독립 실행 폴더입니다.

CSV 추론 시 학습 코드와 동일한 전처리 설정(`data.preprocessing`)을 사용합니다.

## 폴더 구조
```text
cac_inference/
├── configs/
│   └── inference.yaml
├── scripts/
│   ├── infer_csv.py
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

2. DINOv3 코드:
```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
```

3. (선택) Foundation(DINOv3) 체크포인트  
보통 `best.pt`에 foundation + downstream 가중치가 함께 있어 추가 파일이 필요 없습니다.  
downstream만 저장된 체크포인트를 쓸 때만 `model.foundation.checkpoint`를 지정하세요.

4. (전처리 사용 시) DrNoon 전처리 코드
```bash
git clone <drnoon-image-transform-repo> third_party/drnoon-image-transform
```
`data.preprocessing.use_drnoon_preprocess: true`일 때 자동으로 사용됩니다.

## 3) CSV 배치 추론 (권장)
`configs/inference.yaml`의 `data.image_column`, `data.id_columns`, `data.local_prefix`,
`data.preprocessing`을 기준으로 CSV를 읽고 전처리 후 추론합니다.

```bash
python3 scripts/infer_csv.py \
  --csv /absolute/path/to/input.csv \
  --config configs/inference.yaml \
  --output_csv outputs/predictions.csv
```

자주 쓰는 오버라이드:
```bash
python3 scripts/infer_csv.py \
  --csv /absolute/path/to/input.csv \
  --downstream_checkpoint /absolute/path/to/best.pt \
  --dinov3_repo /absolute/path/to/dinov3 \
  --batch_size 16 \
  --threshold 0.5 \
  --output_csv outputs/predictions.csv
```

## 4) 단일 이미지 추론
```bash
python3 scripts/infer_image.py \
  --image /absolute/path/to/fundus.jpg \
  --config configs/inference.yaml
```

## 5) GitHub에 올리기
```bash
cd /home/tj/Research/foundation_downstream_task/cac_inference
git add .
git commit -m "Add standalone CAC CSV/image inference package"
git push
```

주의: `weights/*.pt`는 `.gitignore`에 포함되어 모델 파일은 업로드되지 않습니다.
