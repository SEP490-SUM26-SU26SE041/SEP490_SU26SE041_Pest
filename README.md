# Argo Pest

Argo Pest la ung dung nhan dien con trung gay hai. Project hien tai da tach rieng:

- `backend/`: FastAPI API, chay model ONNX bang ONNX Runtime.
- `frontend/`: Streamlit UI de upload anh va xem ket qua.
- `models/`: cache model local de test tren may.

Backend da deploy tren Render va mobile/app co the goi API truc tiep.

## Cau truc thu muc

```text
Argo_Pest/
|-- backend/
|   |-- main.py
|   |-- requirements.txt
|   |-- Procfile
|-- frontend/
|   |-- app.py
|   |-- requirements.txt
|   |-- Procfile
|-- models/
|   |-- best_pest_and_non_pest.onnx
|   |-- best_detect_pest.onnx
|   |-- best_classify_pest.onnx
|   |-- best_classify_pest.onnx.data
|-- render.yaml
|-- README.md
```

## Model dang dung

Backend can cac file:

```text
best_pest_and_non_pest.onnx
best_detect_pest.onnx
best_classify_pest.onnx
best_classify_pest.onnx.data
```

Neu local khong co model trong `models/`, backend se tai tu Hugging Face bang cac bien moi truong:

```text
GATE_MODEL_URL=https://huggingface.co/nonametd/argo_pest/resolve/main/best_pest_and_non_pest.onnx
YOLO_MODEL_URL=https://huggingface.co/nonametd/argo_pest/resolve/main/best_detect_pest.onnx
CLASSIFY_MODEL_URL=https://huggingface.co/nonametd/argo_pest/resolve/main/best_classify_pest.onnx
```

Luu y: dung link `resolve/main`, khong dung link `blob/main`.

## Tao va kich hoat venv

Chay tai thu muc root:

```powershell
cd E:\Argo_Pest
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Neu PowerShell khong cho activate, chay:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Chay backend local

Cai dependencies:

```powershell
cd E:\Argo_Pest
.\venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
```

Set model URLs:

```powershell
$env:GATE_MODEL_URL="https://huggingface.co/nonametd/argo_pest/resolve/main/best_pest_and_non_pest.onnx"
$env:YOLO_MODEL_URL="https://huggingface.co/nonametd/argo_pest/resolve/main/best_detect_pest.onnx"
$env:CLASSIFY_MODEL_URL="https://huggingface.co/nonametd/argo_pest/resolve/main/best_classify_pest.onnx"
```

Chay backend:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Neu port `8000` dang bi app khac chiem, dung port `8001`:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

Kiem tra backend:

```text
http://localhost:8000/health
```

Hoac neu chay port 8001:

```text
http://localhost:8001/health
```

Ket qua dung:

```json
{
  "ok": true,
  "runtime": "onnxruntime",
  "error": null
}
```

Swagger UI de test API:

```text
http://localhost:8000/docs
```

## Chay frontend local

Mo terminal thu hai:

```powershell
cd E:\Argo_Pest
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe -m pip install -r .\frontend\requirements.txt
```

Neu backend chay port 8000:

```powershell
$env:ARGO_PEST_API_URL="http://localhost:8000"
streamlit run .\frontend\app.py
```

Neu backend chay port 8001:

```powershell
$env:ARGO_PEST_API_URL="http://localhost:8001"
streamlit run .\frontend\app.py
```

Frontend se mo tren browser. Upload anh va bam `Analyze` de xem:

- anh da ve bounding box
- ket qua pest/non-pest
- bang detections
- JSON backend tra ve

## API backend

### Health

```http
GET /health
```

### Predict

```http
POST /predict
```

Request type:

```text
multipart/form-data
```

Fields:

```text
file: anh jpg/jpeg/png
conf: nguong confidence, mac dinh 0.40
iou: nguong IoU, mac dinh 0.45
```

Response gom:

```text
is_pest
gate_confidence
detections
annotated_image_base64
```

## Deploy len Render

Project co san `render.yaml`, co the deploy bang Render Blueprint.

Backend service:

```text
argo-pest-api
```

Frontend service:

```text
argo-pest-frontend
```

Trong Render backend `argo-pest-api`, set Environment Variables:

```text
GATE_MODEL_URL=https://huggingface.co/nonametd/argo_pest/resolve/main/best_pest_and_non_pest.onnx
YOLO_MODEL_URL=https://huggingface.co/nonametd/argo_pest/resolve/main/best_detect_pest.onnx
CLASSIFY_MODEL_URL=https://huggingface.co/nonametd/argo_pest/resolve/main/best_classify_pest.onnx
```

Sau do deploy:

```text
Manual Deploy -> Clear build cache & deploy
```

Kiem tra backend Render:

```text
https://argo-pest-api.onrender.com/health
```

Trong frontend `argo-pest-frontend`, set:

```text
ARGO_PEST_API_URL=https://argo-pest-api.onrender.com
```

## Git va file khong nen commit

Khong commit cac file local nay:

```text
venv/
models/
*.onnx
*.onnx.data
*.pt
*.pth
```

Nhung file nay da duoc cau hinh trong `.gitignore`.

## Loi thuong gap

### Backend health ok false

Mo:

```text
/health
```

Doc truong `error`. Neu thieu `.onnx.data`, can upload file `.onnx.data` len Hugging Face hoac dat file vao `models/`.

### Frontend bao khong goi duoc backend

Kiem tra backend co chay khong:

```text
http://localhost:8000/health
```

Neu backend chay port 8001, frontend phai set:

```powershell
$env:ARGO_PEST_API_URL="http://localhost:8001"
```

### Port 8000 bi chiem

Chay backend bang port 8001:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```
