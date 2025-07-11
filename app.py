import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
from datetime import timedelta
from ultralytics import YOLO
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="🚦 YOLOv11 Traffic Camera", layout="wide")

# Style Tweaks
st.markdown("""
    <style>
        .main { background-color: #f4f4f4; }
        .css-1aumxhk, .stButton>button {
            font-size: 16px;
            border-radius: 8px;
        }
        h1, h2, h3, h4 { color: #333; }
    </style>
""", unsafe_allow_html=True)


# Color Map for Classes
CLASS_COLOR_MAP = {
    'bicycle': (71, 99, 255),
    'bus': (113, 179, 60),
    'car': (255, 144, 30),
    'lorry': (0, 215, 255),
    'motorcycle': (0, 184, 202),
    'pedestrian': (180, 105, 255),
    'van': (0, 165, 255)
}

def get_yolo_color(class_name):
    return CLASS_COLOR_MAP.get(class_name, (255, 255, 255))

# Load YOLOv11 Model
model = YOLO("YOLO11- ImprovedP2.pt")
model.to('cpu')
st.sidebar.header("🔧 Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.1, 0.9, 0.4, 0.05)
st.title("🚦 YOLOv11 Traffic Camera Analysis Dashboard")

# --- VIDEO SECTION ---
uploaded_video = st.file_uploader("📤 Upload a traffic video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    class_counts = {}
    timeline_data = []
    stframe = st.empty()
    progress = st.progress(0)

    st.subheader("Processing video...")

    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, device='cpu', conf=conf_threshold, iou=iou_threshold)
        frame_classes = []

        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])
                class_id = int(cls)
                class_name = model.names[class_id]
                color = get_yolo_color(class_name)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                frame_classes.append(class_name)
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        timeline_data.append(frame_classes)
        out.write(frame)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        progress.progress((frame_num + 1) / total_frames)

    cap.release()
    out.release()
    st.success("✅ Video processing complete!")

    # Class Distribution Pie Chart
    st.subheader("Class Distribution")
    class_df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"]).sort_values(by="Count", ascending=False)
    fig1 = px.pie(class_df, values="Count", names="Class", hole=0.4, title="Object Detection Summary")
    st.plotly_chart(fig1, use_container_width=False)

    # Timeline Line Chart
    st.subheader("Detections Over Time")
    timeline_df = pd.DataFrame(columns=model.names.values())
    for classes in timeline_data:
        row = {cls: classes.count(cls) for cls in model.names.values()}
        timeline_df = pd.concat([timeline_df, pd.DataFrame([row])], ignore_index=True)
    timeline_df.fillna(0, inplace=True)
    second_df = timeline_df.groupby(timeline_df.index // int(fps)).sum()
    second_df.index.name = "Seconds"

    fig2 = px.line(second_df, title="Detections per Class (Per Second)", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    # Frame Gallery
    st.subheader("Frame Gallery")
    cap = cv2.VideoCapture(video_path)
    frame_interval = int(fps * 1)
    gallery_frames = []

    for frame_idx in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            continue

        results = model.predict(frame, device='cpu', conf=conf_threshold, iou=iou_threshold)
        det_class_count = {}

        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])
                class_id = int(cls)
                class_name = model.names[class_id]
                det_class_count[class_name] = det_class_count.get(class_name, 0) + 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), get_yolo_color(class_name), 2)

        timestamp = str(timedelta(seconds=int(frame_idx / fps)))
        caption = f"{timestamp}\n" + "\n".join([f"{k}: {v}" for k, v in det_class_count.items()])
        gallery_frames.append((frame.copy(), caption))

    cap.release()

    cols = st.columns(3)
    for i, (img, cap_text) in enumerate(gallery_frames):
        with cols[i % 3]:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=cap_text, use_container_width=True)

    # Color Legend
    st.subheader("🎨 Bounding Box Color Legend")
    legend_items = []
    for class_name, bgr in CLASS_COLOR_MAP.items():
        hex_color = '#%02x%02x%02x' % bgr[::-1]
        color_box = f"<span style='display:inline-block;width:16px;height:16px;background-color:{hex_color};margin-right:8px;border-radius:3px;'></span>"
        legend_items.append(f"{color_box} <b>{class_name}</b>")

    st.markdown("<br>".join(legend_items), unsafe_allow_html=True)

    # Download Section
    st.subheader("Download Annotated Video")
    with open(out_path, "rb") as f:
        st.download_button("⬇️ Download Processed Video", f, file_name="annotated_traffic.mp4")


# --- IMAGE SECTION ---
uploaded_images = st.file_uploader("📸 Upload image(s) for detection", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_images:
    st.subheader("Image Detection Results")
    image_class_counts = {}

    for image_file in uploaded_images:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        results = model.predict(img, device='cpu', conf=conf_threshold, iou=iou_threshold)

        frame_counts = {}
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])
                class_id = int(cls)
                class_name = model.names[class_id]
                color = get_yolo_color(class_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Count per image and overall
                frame_counts[class_name] = frame_counts.get(class_name, 0) + 1
                image_class_counts[class_name] = image_class_counts.get(class_name, 0) + 1

        # Build caption
        if frame_counts:
            caption = "\n".join([f"{k}: {v}" for k, v in frame_counts.items()])
        else:
            caption = "No detections."

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)

    if image_class_counts:
        st.subheader("📊 Image Class Distribution")
        image_df = pd.DataFrame(list(image_class_counts.items()), columns=["Class", "Count"]).sort_values(by="Count", ascending=False)
        fig = px.pie(image_df, values="Count", names="Class", hole=0.4, title="Detected Objects in Images")
        st.plotly_chart(fig, use_container_width=True)

