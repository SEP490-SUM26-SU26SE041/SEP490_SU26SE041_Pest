import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


APP_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("MODEL_DIR", APP_DIR / "models"))
IMG_SIZE = int(os.getenv("CLASSIFIER_IMG_SIZE", "224"))
YOLO_IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))

PEST_CLASS_NAMES = [
    "Ants",
    "Bees",
    "Beetles",
    "Caterpillars",
    "Earthworms",
    "Earwigs",
    "Grasshoppers",
    "Moths",
    "Slugs",
    "Snails",
    "Wasps",
    "Weevils",
]

MODEL_FILES = {
    "gate": os.getenv("GATE_MODEL_FILENAME", "best_pest_and_non_pest.onnx"),
    "yolo": os.getenv("YOLO_MODEL_FILENAME", "best_detect_pest.onnx"),
    "classify": os.getenv("CLASSIFY_MODEL_FILENAME", "best_classify_pest.onnx"),
}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_suffix(destination.suffix + ".download")
    with requests.get(url, stream=True, timeout=(20, 600)) as response:
        response.raise_for_status()
        with temp_destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    temp_destination.replace(destination)


def download_optional_file(url: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_suffix(destination.suffix + ".download")
    try:
        with requests.get(url, stream=True, timeout=(20, 600)) as response:
            if response.status_code == 404:
                return False
            response.raise_for_status()
            with temp_destination.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
        temp_destination.replace(destination)
        return True
    except Exception:
        return False
    finally:
        if temp_destination.exists():
            temp_destination.unlink()


def ensure_onnx_external_data(path: Path, env_var: str) -> None:
    data_path = path.with_suffix(path.suffix + ".data")
    if data_path.exists():
        return
    model_url = os.getenv(env_var)
    if model_url and path.suffix == ".onnx":
        download_optional_file(model_url + ".data", data_path)


def resolve_model_path(filename: str, env_var: str) -> Path:
    local_path = APP_DIR / filename
    if local_path.exists():
        ensure_onnx_external_data(local_path, env_var)
        return local_path

    model_path = MODEL_DIR / filename
    if model_path.exists():
        ensure_onnx_external_data(model_path, env_var)
        return model_path

    model_url = os.getenv(env_var)
    if not model_url:
        raise FileNotFoundError(
            f"{filename} not found. Set {env_var} to a direct download URL."
        )

    download_file(model_url, model_path)
    ensure_onnx_external_data(model_path, env_var)
    return model_path


def create_session(path: Path) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=providers)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def preprocess_classifier(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0).astype(np.float32)


def letterbox(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    h, w = image_bgr.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, scale, (left, top)


def preprocess_yolo(image_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
    padded, scale, pad = letterbox(image_bgr, YOLO_IMG_SIZE)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0).astype(np.float32), scale, pad


def output_to_predictions(output: np.ndarray) -> np.ndarray:
    pred = np.asarray(output)
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise ValueError(f"Unsupported YOLO output shape: {output.shape}")
    if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 256:
        pred = pred.T
    return pred


def parse_yolo_output(
    output: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    original_shape: tuple[int, int],
    scale: float,
    pad: tuple[float, float],
) -> list[dict]:
    pred = output_to_predictions(output)
    img_h, img_w = original_shape
    pad_x, pad_y = pad

    boxes_xyxy = []
    scores = []
    class_ids = []

    for row in pred:
        if row.shape[0] < 5:
            continue

        box = row[:4].astype(np.float32)
        tail = row[4:].astype(np.float32)
        if tail.shape[0] == 1:
            score = float(tail[0])
            class_id = 0
        elif tail.shape[0] == 2:
            score = float(tail[0] * tail[1])
            class_id = 0
        else:
            # Ultralytics YOLOv8 ONNX commonly returns [cx, cy, w, h, class_scores...].
            cls_scores = tail
            class_id = int(np.argmax(cls_scores))
            score = float(cls_scores[class_id])

        if score < conf_thresh:
            continue

        cx, cy, bw, bh = [float(v) for v in box]
        if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 2.0:
            cx *= YOLO_IMG_SIZE
            cy *= YOLO_IMG_SIZE
            bw *= YOLO_IMG_SIZE
            bh *= YOLO_IMG_SIZE

        x1 = (cx - bw / 2 - pad_x) / scale
        y1 = (cy - bh / 2 - pad_y) / scale
        x2 = (cx + bw / 2 - pad_x) / scale
        y2 = (cy + bh / 2 - pad_y) / scale

        x1 = int(max(0, min(img_w - 1, x1)))
        y1 = int(max(0, min(img_h - 1, y1)))
        x2 = int(max(0, min(img_w - 1, x2)))
        y2 = int(max(0, min(img_h - 1, y2)))
        if x2 <= x1 or y2 <= y1:
            continue

        boxes_xyxy.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(class_id)

    if not boxes_xyxy:
        return []

    nms_boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes_xyxy]
    keep = cv2.dnn.NMSBoxes(nms_boxes, scores, conf_thresh, iou_thresh)
    if len(keep) == 0:
        return []

    detections = []
    for idx in np.array(keep).reshape(-1):
        x1, y1, x2, y2 = boxes_xyxy[int(idx)]
        detections.append(
            {
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "detection_confidence": float(scores[int(idx)]),
                "detector_class_id": int(class_ids[int(idx)]),
            }
        )
    return detections


def run_session(session: ort.InferenceSession, input_array: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_array})[0]


def encode_image_rgb(image_rgb: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode result image.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def draw_label(image: np.ndarray, x1: int, y1: int, label: str) -> None:
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    label_y1 = max(0, y1 - th - 8)
    cv2.rectangle(image, (x1, label_y1), (x1 + tw + 4, y1), (0, 255, 80), -1)
    cv2.putText(
        image,
        label,
        (x1 + 2, max(th + 2, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
    )


app = FastAPI(title="Argo Pest API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    gate_path = resolve_model_path(MODEL_FILES["gate"], "GATE_MODEL_URL")
    yolo_path = resolve_model_path(MODEL_FILES["yolo"], "YOLO_MODEL_URL")
    classify_path = resolve_model_path(MODEL_FILES["classify"], "CLASSIFY_MODEL_URL")
    gate_session = create_session(gate_path)
    yolo_session = create_session(yolo_path)
    classify_session = create_session(classify_path)
    startup_error = None
except Exception as exc:
    gate_session = None
    yolo_session = None
    classify_session = None
    startup_error = str(exc)


@app.get("/health")
def health():
    return {
        "ok": startup_error is None,
        "runtime": "onnxruntime",
        "classifier_img_size": IMG_SIZE,
        "yolo_img_size": YOLO_IMG_SIZE,
        "model_files": MODEL_FILES,
        "error": startup_error,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.40),
    iou: float = Form(0.45),
):
    if startup_error is not None:
        raise HTTPException(status_code=500, detail=f"Model load error: {startup_error}")
    if conf < 0 or conf > 1 or iou < 0 or iou > 1:
        raise HTTPException(status_code=400, detail="conf and iou must be between 0 and 1.")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read image file.") from exc

    gate_logits = run_session(gate_session, preprocess_classifier(image))
    gate_prob = softmax(gate_logits)[0]
    is_pest = int(np.argmax(gate_prob)) == 1
    gate_confidence = {
        "non_pest": float(gate_prob[0]),
        "pest": float(gate_prob[1]),
    }

    image_rgb = np.array(image)
    if not is_pest:
        return {
            "is_pest": False,
            "gate_confidence": gate_confidence,
            "detections": [],
            "annotated_image_base64": encode_image_rgb(image_rgb),
        }

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    yolo_input, scale, pad = preprocess_yolo(img_bgr)
    yolo_output = run_session(yolo_session, yolo_input)
    detections = parse_yolo_output(
        yolo_output,
        conf_thresh=conf,
        iou_thresh=iou,
        original_shape=img_bgr.shape[:2],
        scale=scale,
        pad=pad,
    )

    annotated = img_bgr.copy()
    results = []
    for det in detections:
        box = det["box"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        roi_bgr = img_bgr[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            continue

        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi_pil = Image.fromarray(roi_rgb)
        cls_logits = run_session(classify_session, preprocess_classifier(roi_pil))
        cls_prob = softmax(cls_logits)[0]
        cls_id = int(np.argmax(cls_prob))
        cls_name = PEST_CLASS_NAMES[cls_id] if cls_id < len(PEST_CLASS_NAMES) else str(cls_id)
        cls_conf = float(cls_prob[cls_id])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 80), 2)
        draw_label(annotated, x1, y1, f"{cls_name} {cls_conf * 100:.0f}%")
        results.append(
            {
                "class_id": cls_id,
                "class_name": cls_name,
                "classification_confidence": cls_conf,
                "detection_confidence": det["detection_confidence"],
                "detector_class_id": det["detector_class_id"],
                "box": box,
            }
        )

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return {
        "is_pest": True,
        "gate_confidence": gate_confidence,
        "detections": results,
        "annotated_image_base64": encode_image_rgb(annotated_rgb),
    }
