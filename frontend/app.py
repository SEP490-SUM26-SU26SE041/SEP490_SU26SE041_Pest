import base64
import os

import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("ARGO_PEST_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Argo Pest", layout="wide")
st.title("Argo Pest Detection")

with st.sidebar:
    api_url = st.text_input("Backend API URL", DEFAULT_API_URL).rstrip("/")
    conf_thresh = st.slider("Confidence (YOLO)", 0.0, 1.0, 0.40, 0.05)
    iou_thresh = st.slider("IoU (YOLO)", 0.0, 1.0, 0.45, 0.05)

col_left, col_right = st.columns([1, 2])

with col_left:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    run_btn = st.button("Analyze", use_container_width=True, type="primary")

with col_right:
    st.subheader("Result")

    if uploaded_file is None:
        st.info("Upload an image and click Analyze.")
    elif not run_btn:
        st.image(uploaded_file, caption="Uploaded image", width=480)
    else:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "image/jpeg",
            )
        }
        data = {"conf": str(conf_thresh), "iou": str(iou_thresh)}

        try:
            with st.spinner("Analyzing..."):
                response = requests.post(
                    f"{api_url}/predict",
                    files=files,
                    data=data,
                    timeout=120,
                )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Could not call backend API: {exc}")
            st.stop()

        image_bytes = base64.b64decode(result["annotated_image_base64"])
        st.image(image_bytes, width=480)

        gate = result["gate_confidence"]
        if not result["is_pest"]:
            st.success(f"Non-Pest ({gate['non_pest'] * 100:.1f}%)")
        elif not result["detections"]:
            st.warning("Pest detected, but YOLO did not locate a concrete region.")
        else:
            st.success(f"Pest ({gate['pest'] * 100:.1f}%)")
            st.dataframe(result["detections"], use_container_width=True)

        st.subheader("Backend JSON")
        st.json(result)
